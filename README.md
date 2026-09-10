# Axon — Exact-Character Stateful AI Architecture

[![Tests](https://img.shields.io/badge/tests-594%20passing-brightgreen)](tests/)
[![Substrate](https://img.shields.io/badge/substrate-16D%20frozen%20exact-blue)](substrate/)
[![Lineage](https://img.shields.io/badge/lineage-SHA--256%20content--addressed-violet)](roundtable/)
[![Cloud Ready](https://img.shields.io/badge/compute-Spot%20%2F%20Preemptible%20Native-orange)](scripts/axon_kaggle.py)
[![Website](https://img.shields.io/badge/website-gliksbot.com-cyan)](https://gliksbot.com)

Axon is a stateful, organ-governed AI runtime under active development by DexterGliksbot LLC. 
Unlike conventional LLMs that discard session state and approximate memory through lossy summaries, 
Axon treats **learned state as the primary asset**: bit-exact character preservation, transactional governance, 
and interruptible compute tranches designed for spot cloud fleets.

---

## 🏛️ The Four Core Architectural Organs

1. **16D Frozen Substrate (The Nervous System)**:
   Zero tokenization amnesia. Unicode scalars are represented as exact 16-dimensional coordinate vectors. Malformed UTF-8 or invalid byte sequences fail closed at the compiler boundary.
2. **Heart Transaction Guard (The Sole Writer)**:
   Reasoning cores can only submit typed proposals. The Heart OS is the sole authority permitted to mutate canonical state, enforcing strict ACID boundaries and live fail-closed rejection of adversarial injections.
3. **Dormant-in-Place Memory (Attention Without Amnesia)**:
   Sliding the regional attention mask changes what the reasoning rail attends to, but never deletes data. 59,875 autobiographical records remain addressable with cryptographic SHA-256 provenance.
4. **Spot-Native Compute Tranches (Built for Cloud Fleets)**:
   Optimizer states, gradients, and model parameters checkpoint every 15 steps. Runs pause and resume across spot/preemptible GPU evictions with zero state loss.

---

## 📊 Verifiable Cloud Receipts (D64 Receipt Continuation)

| Metric | Pre-Receipt Baseline | Kaggle Ablation v1 | Kaggle Ablation v2 | Status |
|---|---:|---:|---:|:---:|
| **Source Position Accuracy** | 64.5% | **100.0%** | **100.0%** | **SOLVED** |
| **Payload Content Accuracy** | 64.5% | **100.0%** | **100.0%** | **SOLVED** |
| **Copy-Route Decision Gate** | 100.0% | 0.0% | **100.0%** | **RECOVERED** |
| **Heldout Loss Reduction** | 2.382 | 0.732 | **0.446** | **-81.3%** |

*All audit reports and evidence bundles are immutably recorded in [`roundtable/reports/`](roundtable/reports/).*

---

## ⚡ 30-Second Quickstart

Verify the substrate and control plane locally:

```powershell
# 1. Verify exact 16D Unicode substrate (10/10 conformance gates)
python substrate/substrate.py

# 2. Verify Heart transactional commit & fail-closed injection rejection
python -m pytest tests/test_heart_control_plane.py -q

# 3. Inspect Trainer state & cloud job lineage
python scripts/axon_trainer.py status
```

---

## 🤝 Cloud Partnership: Why Google Cloud?

Axon is applying for the **Google Cloud for Startups** program. Our architecture maps directly onto Google Cloud infrastructure:
* **Preemptible TPU v5e / v6e Slices**: Axon's tranche execution makes preemptible compute 100% loss-free.
* **Vertex AI & GKE Spot**: Content-addressed deployment packets allow distributed worker pools to train without centralized state drift.
* **Scale Path**: Transitioning from current 64D proof geometries to 512D/1024D packed rails on Google Cloud accelerators.

---

## 🛠️ Developer & Agent Protocol

Before contributing or modifying repository state, every human or autonomous agent must read:

1. `AGENTS.md`
2. `docs/WORKING_CONTRACT.md`
3. `docs/SOURCE_OF_TRUTH.md` (Architecture doctrine)
4. `docs/DAY_ZERO.md`
5. `roundtable/ENGINEERS_LEDGER_PROTOCOL.md`
6. `roundtable/ENGINEERS_LEDGER.md` (Current context)

Historical event authority resides immutably in `roundtable/ENGINEERS_LEDGER_CANONICAL.jsonl`.

### Canonical State Roots
* `State/active/`: Active branch and HEAD pointer.
* `State/dormant/`: 59,875-record autobiographical memory body.
* `State/training/`: Content-addressed curricula, checkpoints, and cloud job artifacts.
* `archive/`: Historical Day Zero legacy evidence (never imported in active runtime).

---

© 2026 DexterGliksbot LLC · [gliksbot.com](https://gliksbot.com)
