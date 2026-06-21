"""
config.py
=========
Single source of truth for every hyperparameter, path, and switch in pHGFN
(pH-conditioned Generative Flow Network for RNA-targeted drug design).

Design rule
-----------
Change values *here* and nowhere else. No other file should hardcode a number,
a path, or a threshold — they import `cfg` from this module and read from it.

Architecture note (important — read before changing anything)
-------------------------------------------------------------
This config reflects the *structurally-grounded* pHGFN design:

  1. ORACLE (fast surrogate)      RNA-FM + ChemBERTa (both frozen) + a small
                                  trainable fusion head. It is trained on two
                                  signals:
                                    (a) HARIBOSS RNA-ligand binding proxy, and
                                    (b) the GNINA acidic-vs-neutral *differential*.
                                  Use (b) as the cheap, millisecond in-loop reward.

  2. GNINA (expensive truth)      Neural-network docking against the *actual 3D
                                  coordinates* of the two KRAS conformers:
                                    - acidic   = folded i-motif  (structures/kras_acidic.pdb)
                                    - neutral  = unfolded ssRNA   (structures/kras_neutral.pdb)
                                  Run OFFLINE to label a molecule library (-> trains
                                  the proxy) and to VALIDATE the final top-K. It is
                                  far too slow to call inside the GFlowNet loop.

  3. GFlowNet                     Generates pH-conditioned molecules; its reward is
                                  the proxy-predicted GNINA differential (softplus
                                  shaped), gated by ADMET.

Why this is scientifically defensible: the selectivity signal originates from
real 3D physics (GNINA on two genuinely different conformers), not from an
untrained scalar. The proxy merely *amortizes* that signal so generation is fast.

Lines marked `# [redesign]` differ from the original sequence-only spec.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Root path. Everything else is derived from this so the project is relocatable
# by changing one line.
# --------------------------------------------------------------------------- #
BASE = Path("/mnt/Data/Tushaar/acmr-fall26")


# =========================================================================== #
# DATA
# =========================================================================== #
@dataclass
class DataConfig:
    """Filesystem locations of inputs and the sequences/derived data we produce."""

    # --- Pre-existing inputs (already on disk; never re-download) ----------- #
    hariboss_dir: Path = BASE / "hariboss"        # 98 PDB files (RNA-ligand complexes)
    rnafm_dir: Path = BASE / "rnafm"              # local RNA-FM weights + tokenizer
    chemberta_dir: Path = BASE / "chemberta"      # local ChemBERTa weights + tokenizer
    structures_dir: Path = BASE / "structures"    # kras_acidic.pdb (+ kras_neutral.pdb, generated)
    processed_dir: Path = BASE / "data" / "processed"

    # --- KRAS i-motif target sequence -------------------------------------- #
    # 21-nt C-rich KRAS promoter element that forms an i-motif at acidic pH.
    kras_sequence: str = "CCCCGCCCCGCCCCCGCCCCC"

    # --- Conformer structure files ----------------------------------------- #
    acidic_pdb: Path = BASE / "structures" / "kras_acidic.pdb"     # folded i-motif (given)
    neutral_pdb: Path = BASE / "structures" / "kras_neutral.pdb"   # [redesign] built via AmberTools NAB

    # --- HARIBOSS processed dataset (oracle binding head) ------------------- #
    hariboss_csv: Path = BASE / "data" / "processed" / "hariboss_processed.csv"

    # --- GNINA differential label library (proxy head) --------------------- #  [redesign]
    # A library of molecules (HARIBOSS ligands + ZINC/ChEMBL-style fragments +
    # GFlowNet warm-start samples) docked offline into BOTH conformers. Each row:
    # smiles, gnina_acidic, gnina_neutral, gnina_diff. This trains the proxy.
    proxy_library_csv: Path = BASE / "data" / "processed" / "gnina_labels.csv"
    proxy_library_size: int = 2000   # # molecules to dock offline for proxy training

    # --- Splits ------------------------------------------------------------ #
    train_split: float = 0.70
    val_split: float = 0.15
    test_split: float = 0.15


# =========================================================================== #
# DOCKING (GNINA)                                                    [redesign]
# =========================================================================== #
@dataclass
class DockingConfig:
    """
    GNINA docking settings.

    GNINA runs inside a Singularity container (host is RHEL 9 / glibc 2.34; the
    prebuilt GNINA binary needs glibc 2.35, so we use the official Docker image
    converted to a .sif and run with `--nv` for GPU passthrough).

    Invocation shape:
        singularity exec --nv <gnina_sif> gnina \
            -r <receptor.pdb> -l <ligand.sdf> \
            --autobox_ligand <receptor.pdb> --autobox_add <pad> \
            --exhaustiveness <e> --num_modes <n> \
            --cnn_scoring <mode> --seed <seed>
    """

    # --- How GNINA is executed --------------------------------------------- #
    use_singularity: bool = True
    singularity_bin: str = "singularity"
    gnina_sif: Path = BASE / "tools" / "gnina.sif"
    bare_binary: Path = BASE / "tools" / "gnina"   # fallback only (won't run on glibc 2.34)
    use_gpu: bool = True                            # adds `--nv` to singularity exec

    # --- Receptors (the two conformers) ------------------------------------ #
    acidic_receptor: Path = BASE / "structures" / "kras_acidic.pdb"
    neutral_receptor: Path = BASE / "structures" / "kras_neutral.pdb"

    # --- Search box -------------------------------------------------------- #
    # The i-motif has no co-crystallised ligand to autobox around, so we box the
    # entire RNA (autobox_ligand = the receptor itself) plus padding.
    autobox_whole_receptor: bool = True
    autobox_add: float = 4.0          # Angstrom padding around the receptor box

    # --- Search / scoring quality ------------------------------------------ #
    exhaustiveness: int = 8           # Vina search effort (higher = slower, better)
    num_modes: int = 9                # poses to generate per ligand
    cnn_scoring: str = "rescore"      # none | rescore | refinement | all  (rescore = CNN on top poses)
    cnn_model: str = "default"        # GNINA CNN ensemble
    seed: int = 42
    per_dock_timeout_s: int = 600     # kill a single docking that hangs

    # --- Differential reward ----------------------------------------------- #
    # reward_raw = score(acidic) - lambda * score(neutral),  higher = better binding.
    # score_metric picks which GNINA output defines "score":
    #   "vina" -> -affinity (kcal/mol negated so higher=better). Empirical, geometry
    #             driven; DEFAULT because GNINA's CNN is protein-trained and our
    #             target is RNA (CNN is out-of-distribution here).
    #   "cnn"  -> CNNaffinity (higher=better). Kept for comparison/ablation.
    score_metric: str = "vina"
    selectivity_lambda: float = 1.5   # penalise neutral binding 50% harder

    # --- Concurrency for offline labeling ---------------------------------- #
    n_parallel_docks: int = 2         # docks to run at once (we have 2 GPUs)


# =========================================================================== #
# ORACLE  (HARIBOSS binding-affinity head — fast surrogate filter)
# =========================================================================== #
@dataclass
class OracleConfig:
    """
    The frozen-encoder + trainable-fusion oracle.

    Encoder dims are fixed by the pre-trained checkpoints and must not change:
      RNA-FM  -> 640,  ChemBERTa -> 768.
    """

    # --- Encoder dimensions (fixed by pre-trained models) ------------------ #
    rna_embed_dim: int = 640
    mol_embed_dim: int = 768
    ph_embed_dim: int = 64

    # --- Fusion head (the only trainable part) ----------------------------- #
    fusion_hidden_dim: int = 512      # 48GB VRAM comfortably allows this
    fusion_heads: int = 8
    fusion_layers: int = 3
    dropout: float = 0.1
    max_rna_len: int = 128
    max_mol_len: int = 128

    # --- Training (binding-affinity regression on HARIBOSS) ---------------- #
    lr: float = 3e-4
    batch_size: int = 32
    epochs: int = 80
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    use_fp16: bool = True
    patience: int = 15                # early-stopping patience (epochs)


# =========================================================================== #
# PROXY  (GNINA-differential head — the fast in-loop reward)         [redesign]
# =========================================================================== #
@dataclass
class ProxyConfig:
    """
    Head that predicts the GNINA acidic-vs-neutral differential from
    (molecule, pH) so the GFlowNet gets a millisecond reward instead of paying
    for a real dock every step.

    It reuses the oracle's frozen encoders + fusion trunk and adds a regression
    head trained on `DataConfig.proxy_library_csv` (the offline GNINA labels).
    """

    hidden_dim: int = 512
    dropout: float = 0.1
    lr: float = 3e-4
    batch_size: int = 64
    epochs: int = 120
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    use_fp16: bool = True
    patience: int = 20
    # Standardise targets (GNINA differential) for stable regression; we store
    # the mean/std at train time and invert at inference.
    standardize_targets: bool = True


# =========================================================================== #
# GFLOWNET
# =========================================================================== #
@dataclass
class GFlowNetConfig:
    """pH-conditioned SELFIES GFlowNet trained with Trajectory Balance."""

    # --- Molecule / sequence sizing ---------------------------------------- #
    selfies_max_length: int = 72
    policy_hidden_dim: int = 512
    policy_layers: int = 6
    policy_heads: int = 8
    ph_embed_dim: int = 64

    # --- Training ---------------------------------------------------------- #
    lr: float = 1e-4
    batch_size: int = 64
    epochs: int = 500
    # Reward = exp(differential / reward_temperature). Lower T = sharper preference
    # for high-selectivity molecules (1.0 spreads our differential range well).
    reward_temperature: float = 1.0
    selectivity_weight: float = 1.5   # mirrors DockingConfig.selectivity_lambda
    replay_buffer_size: int = 20000
    n_samples_per_epoch: int = 128
    grad_clip: float = 1.0
    use_fp16: bool = True

    # --- pH conditioning --------------------------------------------------- #
    target_ph: float = 6.7            # tumour micro-environment (design FOR this)
    comparison_ph: float = 7.4        # healthy tissue (design AGAINST this)

    # --- Reward source ----------------------------------------------------- #  [redesign]
    # "proxy"  -> use the trained GNINA-differential proxy (fast; default)
    # "gnina"  -> call real GNINA in-loop (correct but ~weeks; debugging only)
    reward_source: str = "proxy"


# =========================================================================== #
# EVALUATION
# =========================================================================== #
@dataclass
class EvalConfig:
    """Filtering, diversity, and Pareto settings for generated candidates."""

    n_final_candidates: int = 10000
    top_k: int = 200

    # ADMET thresholds (Lipinski Ro5 + extensions)
    qed_min: float = 0.5
    sa_max: float = 6.0
    mw_max: float = 500.0
    logp_max: float = 5.0
    hbd_max: int = 5
    hba_max: int = 10
    rotatable_bonds_max: int = 10
    tanimoto_novelty_threshold: float = 0.4
    # Lower bounds — reject trivial molecules (e.g. Br2, ClB(Br)Br) that pass the   # [redesign]
    # Ro5 UPPER limits but are not real drug-like scaffolds.
    mw_min: float = 150.0     # keeps real small drugs (aspirin 180, caffeine 194)
    min_heavy_atoms: int = 12  # the main guard against trivial fragments (Br2, etc.)
    min_rings: int = 1
    require_carbon: bool = True

    # Number of top candidates to re-score with REAL GNINA for final validation.  [redesign]
    n_gnina_validate: int = 200


# =========================================================================== #
# SYSTEM
# =========================================================================== #
@dataclass
class SystemConfig:
    """Hardware placement, seeding, and output locations."""

    seed: int = 42
    primary_gpu: int = 0              # oracle / proxy training + inference
    secondary_gpu: int = 1           # GFlowNet training
    use_multi_gpu: bool = True
    checkpoint_dir: Path = BASE / "checkpoints"
    results_dir: Path = BASE / "results"
    log_wandb: bool = False

    # NVIDIA pip-wheel lib dirs are prepended to LD_LIBRARY_PATH when shelling
    # out to the bare GNINA binary. (Unused when running GNINA via Singularity.)
    env_dir: Path = Path("/home/013/a/af/dal058778/miniconda3/envs/phgfn")

    # AmberTools `tleap` lives in its own isolated env (keeps heavy Amber deps   # [redesign]
    # out of the ML env). Used as a CLI to build the neutral ssRNA structure.
    ambertools_tleap: Path = Path("/home/013/a/af/dal058778/miniconda3/envs/ambertools/bin/tleap")


# =========================================================================== #
# TOP-LEVEL CONFIG
# =========================================================================== #
@dataclass
class Config:
    """Aggregates every sub-config. Import the singleton `cfg` below."""

    data: DataConfig = field(default_factory=DataConfig)
    docking: DockingConfig = field(default_factory=DockingConfig)      # [redesign]
    oracle: OracleConfig = field(default_factory=OracleConfig)
    proxy: ProxyConfig = field(default_factory=ProxyConfig)            # [redesign]
    gflownet: GFlowNetConfig = field(default_factory=GFlowNetConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    system: SystemConfig = field(default_factory=SystemConfig)

    def ensure_dirs(self) -> None:
        """Create all output directories we write to (idempotent)."""
        for p in (
            self.data.processed_dir,
            self.system.checkpoint_dir,
            self.system.results_dir,
        ):
            Path(p).mkdir(parents=True, exist_ok=True)


# Global, import-everywhere singleton.
cfg = Config()


# --------------------------------------------------------------------------- #
# Smoke test: `python -m src.config` prints a readable summary.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    cfg.ensure_dirs()
    print("pHGFN configuration loaded OK")
    print(f"  BASE                : {BASE}")
    print(f"  HARIBOSS dir        : {cfg.data.hariboss_dir}")
    print(f"  acidic / neutral    : {cfg.data.acidic_pdb.name} / {cfg.data.neutral_pdb.name}")
    print(f"  GNINA sif           : {cfg.docking.gnina_sif}")
    print(f"  GNINA cnn_scoring   : {cfg.docking.cnn_scoring}  (lambda={cfg.docking.selectivity_lambda})")
    print(f"  Oracle batch/epochs : {cfg.oracle.batch_size} / {cfg.oracle.epochs}")
    print(f"  GFlowNet reward src : {cfg.gflownet.reward_source}")
    print(f"  pH target/compare   : {cfg.gflownet.target_ph} / {cfg.gflownet.comparison_ph}")
    print(f"  GPUs (oracle/gfn)   : cuda:{cfg.system.primary_gpu} / cuda:{cfg.system.secondary_gpu}")
