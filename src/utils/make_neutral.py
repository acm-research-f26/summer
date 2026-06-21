"""
make_neutral.py
===============
Build the **neutral (unfolded) KRAS conformer** used as the "off-target" receptor
in pHGFN's differential docking.

Why this file exists (read the architecture note)
--------------------------------------------------
At tumour pH (~6.7) the C-rich KRAS element folds into a compact **i-motif**
(`structures/kras_acidic.pdb`). At healthy pH (~7.4) the same sequence does NOT
form the i-motif — it is an extended single strand with no intercalated C·C+ pairs.

GNINA scores a molecule against *3D coordinates*, so the two conformers must be
**genuinely different geometrically** for the differential reward
`score(acidic) - 1.5*score(neutral)` to mean anything.

The original spec proposed making the neutral state by deleting CONECT records.
That is wrong: CONECT lines are bond *annotations* — removing them does not move a
single atom, so RNA-FM/GNINA would see an identical structure. Instead we **build a
real extended ssRNA** from sequence with AmberTools `tleap`, which places residues
in a default (non-i-motif) extended conformation.

What we produce
---------------
`structures/kras_neutral.pdb` — heavy-atom-only extended ssRNA for
`CCCCGCCCCGCCCCCGCCCCC`. Hydrogens are stripped so the representation matches the
(heavy-atom-only) acidic PDB, keeping the docking comparison apples-to-apples.

Run:
    conda activate phgfn
    python -m src.utils.make_neutral
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from src.config import cfg

# --------------------------------------------------------------------------- #
# Residue-name maps
# --------------------------------------------------------------------------- #
# tleap (RNA.OL3) unit names are 1-letter (A/G/C/U) with 5'/3' terminal variants
# (A5/A3, C5/C3, ...). The old R-prefixed names (RC5 ...) are only PDB read aliases,
# not valid `sequence` units — so we build with the modern names.
_BASE_TO_AMBER = {"A": "A", "G": "G", "C": "C", "U": "U"}


def _resname_to_base(resname: str) -> str:
    """
    Map any residue-name flavour to its 1-letter RNA base.
    Handles "C", "C5", "C3", "RC", "RC5", "G", "G3", etc. -> the first A/G/C/U char.
    """
    for ch in resname.upper():
        if ch in "AGCU":
            return ch
    return "?"


# --------------------------------------------------------------------------- #
# Sequence helpers
# --------------------------------------------------------------------------- #
def _seq_to_tleap_units(seq: str) -> list[str]:
    """
    Turn an RNA sequence into the list of tleap residue tokens, giving the first
    residue a 5'-terminal cap and the last a 3'-terminal cap.

    Example: "CCG" -> ["RC5", "RC", "RG3"]
    """
    seq = seq.strip().upper().replace("T", "U")
    units: list[str] = []
    n = len(seq)
    for i, base in enumerate(seq):
        if base not in _BASE_TO_AMBER:
            raise ValueError(f"Unsupported base {base!r} at position {i} in {seq!r}")
        token = _BASE_TO_AMBER[base]
        if i == 0:
            token += "5"           # 5'-terminal residue
        elif i == n - 1:
            token += "3"           # 3'-terminal residue
        units.append(token)
    return units


def _extract_sequence_from_pdb(pdb_path: Path) -> str:
    """Read the 1-letter RNA sequence from a PDB by walking residues in order."""
    seq: list[str] = []
    seen: set[tuple[str, str]] = set()
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            chain = line[21]
            resseq = line[22:26].strip()
            key = (chain, resseq)
            if key in seen:
                continue
            seen.add(key)
            resname = line[17:20].strip()
            seq.append(_resname_to_base(resname))
    return "".join(seq)


# --------------------------------------------------------------------------- #
# tleap build
# --------------------------------------------------------------------------- #
def _run_tleap(units: list[str], raw_out: Path) -> None:
    """
    Drive AmberTools `tleap` to build an extended ssRNA and save it to `raw_out`.

    `sequence { ... }` joins residues with default internal coordinates, yielding
    an extended (non-folded) single strand — exactly our neutral model.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        script = tmp_path / "build_neutral.in"
        script.write_text(
            "source leaprc.RNA.OL3\n"
            f"mol = sequence {{ {' '.join(units)} }}\n"
            f"savepdb mol {raw_out}\n"
            "quit\n"
        )
        proc = subprocess.run(
            [str(cfg.system.ambertools_tleap), "-f", str(script)],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0 or not raw_out.exists():
            raise RuntimeError(
                "tleap failed to build the neutral structure.\n"
                f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
            )


def _strip_h_and_standardize(raw_pdb: Path, out_pdb: Path) -> int:
    """
    Write a clean heavy-atom-only PDB:
      * drop hydrogens (match the acidic heavy-atom representation),
      * rewrite Amber residue names (RC5/RC/RC3 ...) to 1-letter (C/G/...),
      * renumber atom serials.
    Returns the number of heavy atoms written.
    """
    kept = 0
    with open(raw_pdb) as fin, open(out_pdb, "w") as fout:
        fout.write("REMARK  pHGFN neutral (unfolded) ssRNA, heavy atoms only\n")
        for line in fin:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            element = line[76:78].strip()
            atom_name = line[12:16].strip()
            # Skip hydrogens (element column, or name heuristics for safety).
            if element == "H" or (not element and atom_name.startswith(("H", "1H", "2H", "3H"))):
                continue
            resname = line[17:20].strip()
            base = _resname_to_base(resname)
            kept += 1
            # Rebuild the record with standardized 1-letter resname, right-justified
            # in the 3-char residue field, and a fresh serial number.
            new_line = (
                f"{line[:6]}{kept:>5} {line[12:17]}{base:>3}{line[20:]}"
            )
            fout.write(new_line)
        fout.write("TER\nEND\n")
    return kept


# --------------------------------------------------------------------------- #
# Geometry report (proves the two conformers differ)
# --------------------------------------------------------------------------- #
def _geometry_report(pdb_path: Path) -> dict:
    """
    Compute simple shape descriptors with BioPython:
      * n_atoms
      * radius of gyration Rg (compactness; i-motif small, extended large)
      * end-to-end distance between first and last residue P/C1' atoms
    """
    import numpy as np
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("s", str(pdb_path))
    coords = np.array([atom.coord for atom in structure.get_atoms()], dtype=float)
    com = coords.mean(axis=0)
    rg = float(np.sqrt(((coords - com) ** 2).sum(axis=1).mean()))

    residues = [r for r in structure.get_residues()]

    def _anchor(res):
        for name in ("P", "C1'", "C1*", "O5'"):
            if name in res:
                return res[name].coord
        return next(res.get_atoms()).coord

    end_to_end = float(np.linalg.norm(_anchor(residues[0]) - _anchor(residues[-1])))
    return {"n_atoms": len(coords), "rg": rg, "end_to_end": end_to_end}


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def build_neutral_structure(verbose: bool = True) -> Path:
    """Build `structures/kras_neutral.pdb` and verify it differs from the acidic fold."""
    acidic = cfg.data.acidic_pdb
    neutral = cfg.data.neutral_pdb
    target_seq = cfg.data.kras_sequence.upper().replace("T", "U")

    # 1) Sanity: the acidic PDB must hold the sequence we think it does.
    acidic_seq = _extract_sequence_from_pdb(acidic)
    if verbose:
        print(f"[make_neutral] acidic PDB sequence : {acidic_seq}")
        print(f"[make_neutral] config sequence     : {target_seq}")
    if acidic_seq != target_seq:
        raise ValueError(
            f"Acidic PDB sequence {acidic_seq!r} != config sequence {target_seq!r}. "
            "Refusing to build a mismatched neutral structure."
        )

    # 2) Build the extended ssRNA with tleap, into a temp raw PDB.
    units = _seq_to_tleap_units(target_seq)
    if verbose:
        print(f"[make_neutral] tleap units         : {' '.join(units)}")
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "neutral_raw.pdb"
        _run_tleap(units, raw)
        n_heavy = _strip_h_and_standardize(raw, neutral)
    if verbose:
        print(f"[make_neutral] wrote {neutral} ({n_heavy} heavy atoms)")

    # 3) Confirm the neutral structure carries the same sequence.
    neutral_seq = _extract_sequence_from_pdb(neutral)
    if neutral_seq != target_seq:
        raise ValueError(
            f"Built neutral sequence {neutral_seq!r} != target {target_seq!r}."
        )

    # 4) Geometry comparison — the whole point: neutral must be far less compact.
    g_acidic = _geometry_report(acidic)
    g_neutral = _geometry_report(neutral)
    if verbose:
        print("\n[make_neutral] geometry comparison (proves conformers differ):")
        print(f"  {'metric':<16}{'acidic(i-motif)':>18}{'neutral(ssRNA)':>18}")
        print(f"  {'heavy atoms':<16}{g_acidic['n_atoms']:>18}{g_neutral['n_atoms']:>18}")
        print(f"  {'radius_gyration':<16}{g_acidic['rg']:>18.2f}{g_neutral['rg']:>18.2f}")
        print(f"  {'end_to_end (A)':<16}{g_acidic['end_to_end']:>18.2f}{g_neutral['end_to_end']:>18.2f}")

    if g_neutral["rg"] <= g_acidic["rg"]:
        print(
            "\n[make_neutral] WARNING: neutral Rg is not larger than acidic Rg. "
            "The extended strand should be less compact than the i-motif — inspect the structures."
        )
    else:
        if verbose:
            print(
                f"\n[make_neutral] OK: neutral is "
                f"{g_neutral['rg'] / g_acidic['rg']:.1f}x less compact than the i-motif fold."
            )
    return neutral


if __name__ == "__main__":
    build_neutral_structure()
