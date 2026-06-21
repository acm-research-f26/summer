#!/bin/bash
# =============================================================================
# pHGFN — complete training & evaluation pipeline
# Hardware: 2x RTX 6000 Ada (48GB).  GPU 0 = oracle, GPU 1 = GFlowNet.
#
# Structurally-grounded design:
#   - GNINA docks molecules into the acidic i-motif vs neutral ssRNA conformers.
#   - A proxy oracle (RNA-FM + ChemBERTa + fusion) is trained on those GNINA
#     differentials and used as the fast in-loop GFlowNet reward.
#   - Real GNINA re-validates the final top candidates.
#
# Usage:
#   nohup bash run_pipeline.sh > results/nohup.log 2>&1 &
#   tail -f results/nohup.log
# =============================================================================
set -euo pipefail

BASE="/mnt/Data/Tushaar/acmr-fall26"
CONDA_SH="/home/013/a/af/dal058778/miniconda3/etc/profile.d/conda.sh"
LIBRARY_SIZE="${LIBRARY_SIZE:-2000}"     # GNINA label-library size (matches DataConfig.proxy_library_size)
N_PARALLEL="${N_PARALLEL:-6}"            # parallel docks during labeling
LOG="$BASE/results/pipeline_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$BASE/results" "$BASE/checkpoints" "$BASE/data/processed"
cd "$BASE"

# Activate env + force offline model loading (weights are already on disk).
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate phgfn
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

echo "=================================================="
echo "pHGFN pipeline start: $(date)"
echo "logging to: $LOG"
echo "=================================================="

# ---- 1. Integration test (must pass) ----
echo "[1/7] Integration test ..."
python test_integration.py | tee -a "$LOG"

# ---- 2. Preprocess HARIBOSS ----
echo "[2/7] Preprocessing HARIBOSS ..."
python -c "from src.oracle.dataset import preprocess_hariboss; preprocess_hariboss()" | tee -a "$LOG"

# ---- 3. Build the neutral (unfolded) conformer ----
echo "[3/7] Building neutral ssRNA structure ..."
python -m src.utils.make_neutral | tee -a "$LOG"

# ---- 4. Offline GNINA labeling (resumable) ----
echo "[4/7] GNINA labeling (library=$LIBRARY_SIZE, parallel=$N_PARALLEL) ..."
python -c "from src.docking.label_library import dock_library; dock_library(n_total=$LIBRARY_SIZE, n_parallel=$N_PARALLEL, gpu_ids=(0,1))" | tee -a "$LOG"

# ---- 5. Train the oracle / proxy (GPU 0) ----
echo "[5/7] Training oracle (proxy of GNINA differential) on cuda:0 ..."
python -c "from src.oracle.train import train_oracle; train_oracle()" | tee -a "$LOG"

# ---- 6. Pretrain (behaviour-clone) then train the GFlowNet (GPU 1) ----
# Random SELFIES decode to tiny non-drug-like fragments, so the policy is first
# behaviour-cloned on a drug-like corpus (ZINC250k subset) before TB fine-tuning.
ZINC="$BASE/data/processed/zinc250k.csv"
if [ ! -f "$ZINC" ]; then
  echo "[6/7] Downloading ZINC250k drug-like corpus ..."
  curl -sL -o "$ZINC" "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc/250k_rndm_zinc_drugs_clean_3.csv"
fi
echo "[6/7] Pretraining policy (behaviour cloning) on cuda:1 ..."
python -c "from src.gflownet.pretrain import build_corpus, pretrain_policy; build_corpus(); pretrain_policy()" | tee -a "$LOG"
echo "[6/7] Training GFlowNet (Trajectory Balance) on cuda:1 ..."
python -c "from src.gflownet.train import train_gflownet; train_gflownet()" | tee -a "$LOG"

# ---- 7. Generate, filter, validate, analyse ----
echo "[7/7] Generating + evaluating candidates ..."
python -c "
from src.evaluation.generate import generate_candidates, validate_with_gnina
from src.evaluation.admet import filter_candidates
from src.evaluation.diversity import compute_all_metrics
from src.evaluation.pareto import compute_pareto_frontier
generate_candidates()
filter_candidates()
compute_all_metrics()
compute_pareto_frontier()
validate_with_gnina()        # real GNINA on the top-K
" | tee -a "$LOG"

echo "=================================================="
echo "pHGFN pipeline complete: $(date)"
echo "key outputs:"
echo "  results/pareto_frontier.png            <- main result figure"
echo "  results/pareto_optimal.csv             <- best trade-off candidates"
echo "  results/candidates_gnina_validated.csv <- real-GNINA selectivity of top-K"
echo "  results/diversity_metrics.json"
echo "  checkpoints/oracle_best.pt  checkpoints/policy_best.pt"
echo "=================================================="
