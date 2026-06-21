"""
test_integration.py
====================
End-to-end smoke test for pHGFN. Runs a TINY slice of every component to confirm
the pieces connect BEFORE committing hours to full training. Fast (< ~2 min).

If any check fails the script exits non-zero — do not start full training until
all checks pass.

Checks (incl. the GNINA-grounded redesign):
  1  RNA-FM loads from local disk (no download)
  2  ChemBERTa loads from local disk (no download)
  3  RNA-FM frozen (0 trainable params)
  4  ChemBERTa frozen (0 trainable params)
  5  Oracle forward pass runs (finite output)
  6  Oracle differential reward is finite (sane)
  7  GFlowNet samples a trajectory -> valid SMILES
  8  Trajectory Balance loss computes (finite, non-zero)
  9  Policy gradients exist after backward()
 10  Oracle encoder gradients do NOT exist (frozen confirmed)
 11  Neutral conformer exists and is geometrically distinct from the i-motif
 12  Real GNINA differential runs in the container (acidic vs neutral)
"""

import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch  # noqa: E402

RESULTS = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append(ok)
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}" + (f": {detail}" if detail else ""))


def main() -> int:
    from src.config import cfg
    from src.gflownet.environment import SELFIESEnvironment
    from src.gflownet.losses import trajectory_balance_loss
    from src.gflownet.policy import pHConditionedPolicy
    from src.oracle.model import pHGFNOracle

    dev = f"cuda:{cfg.system.primary_gpu}" if torch.cuda.is_available() else "cpu"

    # --- Oracle / encoders (checks 1-4) ---
    oracle = pHGFNOracle(device=dev)
    check("RNA-FM loaded from local disk", True)
    check("ChemBERTa loaded from local disk", True)
    rna_train = sum(p.numel() for p in oracle.rna_encoder.parameters() if p.requires_grad)
    mol_train = sum(p.numel() for p in oracle.mol_encoder.parameters() if p.requires_grad)
    check("RNA-FM frozen (0 trainable params)", rna_train == 0, f"{rna_train} trainable")
    check("ChemBERTa frozen (0 trainable params)", mol_train == 0, f"{mol_train} trainable")

    # --- Oracle forward + reward (checks 5-6) ---
    smi = ["CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1"]
    ph = torch.tensor([cfg.gflownet.target_ph] * 2, device=dev)
    out = oracle.forward([cfg.data.kras_sequence] * 2, smi, ph)
    check("Oracle forward pass runs", torch.isfinite(out["score"]).all().item(),
          f"score shape {tuple(out['score'].shape)}")
    reward = oracle.differential_reward(smi)
    check("Oracle differential reward finite", torch.isfinite(reward).all().item(),
          f"values {[round(v,3) for v in reward.tolist()]}")

    # --- GFlowNet sample + TB + grads (checks 7-10) ---
    env = SELFIESEnvironment()
    pol_dev = f"cuda:{cfg.system.secondary_gpu}" if torch.cuda.device_count() > 1 else dev
    policy = pHConditionedPolicy(env=env).to(pol_dev)
    actions, lengths, smiles = policy.sample(8, cfg.gflownet.target_ph, device=pol_dev)
    n_valid = sum(s is not None for s in smiles)
    example = next((s for s in smiles if s), None)
    check("GFlowNet trajectory -> valid SMILES", n_valid > 0, f"{n_valid}/8 valid, e.g. {example}")

    lp = policy.trajectory_log_prob(actions, lengths, cfg.gflownet.target_ph)
    rewards = torch.rand(8, device=pol_dev) + 0.01
    loss = trajectory_balance_loss(policy.log_Z, lp, rewards)
    check("TB loss computes (finite, non-zero)", bool(torch.isfinite(loss)) and loss.item() != 0,
          f"{loss.item():.3f}")

    loss.backward()
    pol_grad = sum(p.grad.abs().sum().item() for p in policy.parameters() if p.grad is not None)
    check("Policy gradients exist after backward()", pol_grad > 0, f"sum|grad|={pol_grad:.1f}")
    enc_grad = any(p.grad is not None for p in oracle.rna_encoder.parameters()) or \
        any(p.grad is not None for p in oracle.mol_encoder.parameters())
    check("Oracle encoder gradients: None (frozen)", not enc_grad)

    # --- Structures + GNINA (checks 11-12) ---
    try:
        from src.utils.make_neutral import _geometry_report
        g_a = _geometry_report(cfg.data.acidic_pdb)
        g_n = _geometry_report(cfg.data.neutral_pdb)
        check("Neutral conformer distinct from i-motif", g_n["rg"] > g_a["rg"],
              f"Rg neutral {g_n['rg']:.1f} > acidic {g_a['rg']:.1f}")
    except Exception as exc:
        check("Neutral conformer distinct from i-motif", False, str(exc))

    try:
        from src.docking.gnina import GninaDocker
        d = GninaDocker().differential("c1ccncc1")  # pyridine
        finite = d["ok"] and d["acidic_score"] is not None and d["neutral_score"] is not None
        check("Real GNINA differential runs", bool(finite),
              f"acidic {d['acidic_score']}, neutral {d['neutral_score']}, diff {d['differential']}")
    except Exception as exc:
        check("Real GNINA differential runs", False, str(exc))

    # --- Summary ---
    print("\n" + "=" * 44)
    passed = sum(RESULTS)
    print(f"INTEGRATION TEST: {passed}/{len(RESULTS)} CHECKS PASSED")
    print("=" * 44)
    if passed == len(RESULTS):
        print("ALL CHECKS PASSED — READY FOR FULL TRAINING")
        return 0
    print("SOME CHECKS FAILED — fix before full training")
    return 1


if __name__ == "__main__":
    sys.exit(main())
