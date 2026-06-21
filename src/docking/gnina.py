"""
gnina.py
========
Thin, well-defined wrapper around **GNINA** for pHGFN's structurally-grounded
differential scoring.

GNINA is a neural-network molecular docking tool (built on smina / AutoDock Vina).
It scores a molecule against the *actual 3D coordinates* of a receptor, so it
returns genuinely different numbers for the folded i-motif (acidic) and the
unfolded ssRNA (neutral) — which is exactly the physical signal pHGFN needs.

Execution model
---------------
GNINA cannot run as a bare binary on this host (RHEL 9 / glibc 2.34; the binary
needs glibc 2.35). We run the official Docker image via **Singularity** with
`--nv` for GPU passthrough. See `DockingConfig` in `src/config.py`.

What GNINA reports (per pose / "mode"):
    affinity (kcal/mol)  -- Vina empirical score; MORE NEGATIVE = stronger binding
    CNNaffinity          -- learned score; HIGHER = stronger binding (protein-trained)

Because our receptor is RNA and GNINA's CNN was trained on protein pockets, the
empirical Vina `affinity` is the more trustworthy signal here. We capture BOTH and
let `DockingConfig.score_metric` choose which defines the unified "score" (higher =
better) used in the differential.

Public API
----------
    docker = GninaDocker()
    res    = docker.dock(smiles, receptor_pdb)        # -> DockResult
    diff   = docker.differential(smiles)              # -> dict with acidic/neutral/diff
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.config import BASE, cfg

# Regex for a GNINA results-table row:
#   mode | affinity | intramol | CNN pose score | CNN affinity
_ROW_RE = re.compile(
    r"^\s*(\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$"
)


@dataclass
class DockResult:
    """Parsed outcome of one GNINA run (best pose summarised)."""

    smiles: str
    receptor: str
    ok: bool
    best_affinity: Optional[float] = None      # most-negative Vina affinity (kcal/mol)
    best_cnn_affinity: Optional[float] = None  # highest CNNaffinity across poses
    n_modes: int = 0
    error: str = ""

    def score(self, metric: str) -> Optional[float]:
        """
        Unified 'higher = better' score used by the differential.
          vina -> -best_affinity   (negate kcal/mol so higher=better)
          cnn  -> best_cnn_affinity
        Returns None if docking failed.
        """
        if not self.ok:
            return None
        if metric == "vina":
            return None if self.best_affinity is None else -self.best_affinity
        if metric == "cnn":
            return self.best_cnn_affinity
        raise ValueError(f"Unknown score_metric {metric!r} (use 'vina' or 'cnn')")


class GninaDocker:
    """
    Runs GNINA (via Singularity) and computes the acidic-vs-neutral differential.

    One instance is reusable across many molecules; it holds the (immutable)
    receptor paths and docking settings from `cfg.docking`.
    """

    def __init__(self, dcfg=None) -> None:
        self.cfg = dcfg or cfg.docking
        # Fail fast if the container / receptors are missing — a clear message now
        # beats a cryptic subprocess error later.
        if self.cfg.use_singularity and not Path(self.cfg.gnina_sif).exists():
            raise FileNotFoundError(
                f"GNINA singularity image not found: {self.cfg.gnina_sif}. "
                "Pull it with: singularity pull <sif> docker://gnina/gnina:latest"
            )
        for r in (self.cfg.acidic_receptor, self.cfg.neutral_receptor):
            if not Path(r).exists():
                raise FileNotFoundError(
                    f"Receptor not found: {r}. "
                    "Build the neutral one with `python -m src.utils.make_neutral`."
                )

    # ------------------------------------------------------------------ #
    # Ligand preparation (RDKit, on host)
    # ------------------------------------------------------------------ #
    @staticmethod
    def smiles_to_sdf(smiles: str, out_sdf: Path, seed: int = 42) -> bool:
        """
        Embed a 3D conformer for `smiles` and write it as an SDF for docking.

        Uses RDKit ETKDG embedding + MMFF optimisation. Returns False if the SMILES
        is invalid or a 3D conformer could not be generated (caller treats as a
        failed dock, reward ~0).
        """
        from rdkit import Chem
        from rdkit.Chem import AllChem

        # Wrap everything: some (esp. random SELFIES-derived) SMILES make RDKit
        # raise mid-pipeline (e.g. KekulizeException). One bad molecule must never
        # crash a batched/parallel labeling run, so any failure -> False.
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
            mol = Chem.AddHs(mol)
            params = AllChem.ETKDGv3()
            params.randomSeed = seed
            if AllChem.EmbedMolecule(mol, params) != 0:
                # Retry with random coordinates for awkward systems.
                if AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=seed) != 0:
                    return False
            try:
                AllChem.MMFFOptimizeMolecule(mol)
            except Exception:
                pass  # optimisation failure is non-fatal; the embedded pose still docks
            writer = Chem.SDWriter(str(out_sdf))
            writer.write(mol)
            writer.close()
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    # Command construction + parsing
    # ------------------------------------------------------------------ #
    def _build_cmd(self, receptor: Path, ligand_sdf: Path, out_sdf: Path) -> list[str]:
        """Assemble the full `singularity exec ... gnina ...` argv."""
        c = self.cfg
        gnina_args = [
            "gnina",
            "-r", str(receptor),
            "-l", str(ligand_sdf),
            "--autobox_ligand", str(receptor),   # box the whole RNA (no co-crystal ligand)
            "--autobox_add", str(c.autobox_add),
            "--exhaustiveness", str(c.exhaustiveness),
            "--num_modes", str(c.num_modes),
            "--cnn_scoring", c.cnn_scoring,
            "--seed", str(c.seed),
            "-o", str(out_sdf),
        ]
        if not c.use_singularity:
            return [str(c.bare_binary), *gnina_args[1:]]
        sing = [c.singularity_bin, "exec"]
        if c.use_gpu:
            sing.append("--nv")
        sing += ["--bind", str(BASE), str(c.gnina_sif)]
        return sing + gnina_args

    @staticmethod
    def _parse_table(stdout: str) -> tuple[Optional[float], Optional[float], int]:
        """
        Extract (best_affinity, best_cnn_affinity, n_modes) from GNINA stdout.
        best_affinity  = min over poses (most negative kcal/mol).
        best_cnn       = max over poses (higher = better).
        """
        affinities: list[float] = []
        cnns: list[float] = []
        for line in stdout.splitlines():
            m = _ROW_RE.match(line)
            if m:
                affinities.append(float(m.group(2)))
                cnns.append(float(m.group(5)))
        if not affinities:
            return None, None, 0
        return min(affinities), max(cnns), len(affinities)

    # ------------------------------------------------------------------ #
    # Single dock
    # ------------------------------------------------------------------ #
    def dock(
        self,
        smiles: str,
        receptor: Path,
        workdir: Optional[Path] = None,
        gpu_id: Optional[int] = None,
    ) -> DockResult:
        """
        Dock one molecule into one receptor and summarise the best pose.
        `gpu_id` pins the container to a single GPU (via CUDA_VISIBLE_DEVICES) so
        offline labeling can saturate both cards.
        """
        receptor = Path(receptor)
        tmp_created = workdir is None
        # Temp files must live under a Singularity-visible path; BASE/tools is bound.
        workdir = workdir or Path(tempfile.mkdtemp(prefix="dock_", dir=BASE / "tools"))
        try:
            ligand_sdf = workdir / "ligand.sdf"
            out_sdf = workdir / "out.sdf"
            if not self.smiles_to_sdf(smiles, ligand_sdf, seed=self.cfg.seed):
                return DockResult(smiles, receptor.name, ok=False, error="ligand_embed_failed")
            cmd = self._build_cmd(receptor, ligand_sdf, out_sdf)
            env = os.environ.copy()
            if gpu_id is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True,
                    timeout=self.cfg.per_dock_timeout_s, env=env,
                )
            except subprocess.TimeoutExpired:
                return DockResult(smiles, receptor.name, ok=False, error="timeout")
            aff, cnn, n = self._parse_table(proc.stdout)
            if n == 0:
                return DockResult(
                    smiles, receptor.name, ok=False,
                    error=f"no_poses (rc={proc.returncode}): {proc.stderr[-200:]}",
                )
            return DockResult(
                smiles, receptor.name, ok=True,
                best_affinity=aff, best_cnn_affinity=cnn, n_modes=n,
            )
        finally:
            if tmp_created:
                shutil.rmtree(workdir, ignore_errors=True)

    # ------------------------------------------------------------------ #
    # Differential (the actual selectivity signal)
    # ------------------------------------------------------------------ #
    def differential(self, smiles: str, gpu_id: Optional[int] = None) -> dict:
        """
        Dock `smiles` into both conformers and compute the selectivity differential.

        Returns a dict:
          smiles, acidic_affinity, neutral_affinity,
          acidic_cnn, neutral_cnn,
          acidic_score, neutral_score,                # per cfg.docking.score_metric
          differential = acidic_score - lambda*neutral_score,
          ok (bool)
        Positive `differential` => molecule prefers the acidic i-motif (tumour-selective).
        """
        metric = self.cfg.score_metric
        lam = self.cfg.selectivity_lambda
        a = self.dock(smiles, self.cfg.acidic_receptor, gpu_id=gpu_id)
        n = self.dock(smiles, self.cfg.neutral_receptor, gpu_id=gpu_id)
        ok = a.ok and n.ok
        sa, sn = a.score(metric), n.score(metric)
        diff = (sa - lam * sn) if ok else None
        return {
            "smiles": smiles,
            "acidic_affinity": a.best_affinity,
            "neutral_affinity": n.best_affinity,
            "acidic_cnn": a.best_cnn_affinity,
            "neutral_cnn": n.best_cnn_affinity,
            "acidic_score": sa,
            "neutral_score": sn,
            "differential": diff,
            "ok": ok,
            "error": "" if ok else f"acidic:{a.error}|neutral:{n.error}",
        }


# --------------------------------------------------------------------------- #
# CLI smoke test: dock a few molecules and show the acidic-vs-neutral split.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    docker = GninaDocker()
    demo = [
        ("aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
        ("caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"),
        ("benzene", "c1ccccc1"),
    ]
    print(f"score_metric={cfg.docking.score_metric}  lambda={cfg.docking.selectivity_lambda}\n")
    hdr = f"{'name':<10}{'acidic_aff':>12}{'neutral_aff':>12}{'a_score':>10}{'n_score':>10}{'diff':>10}"
    print(hdr)
    print("-" * len(hdr))
    for name, smi in demo:
        d = docker.differential(smi)
        if d["ok"]:
            print(
                f"{name:<10}{d['acidic_affinity']:>12.2f}{d['neutral_affinity']:>12.2f}"
                f"{d['acidic_score']:>10.2f}{d['neutral_score']:>10.2f}{d['differential']:>10.2f}"
            )
        else:
            print(f"{name:<10}  FAILED: {d['error']}")
