# BoostFlag — Code & Data Artifact

Anonymous code and data artifact accompanying the ICEICT 2026 submission
*"Boost and Flag: Predicting the Next Attack Target on Enterprise Authentication Graphs."*

This repository contains **only code and experimental data** — no paper source or figures. Every "recomputable" claim in the paper is mechanically verifiable from these files, without refitting any model.

## 1. Layout

```
├── README.md                     ← this file
├── results/
│   ├── golden_master.json        ← frozen result master (98 scored keys)
│   ├── exp*.json                 ← 14 experiment JSONs (with correction annotations)
│   ├── exp4_gains_model_vs_*.npy ← 8 per-event gain series (hop-1 vs each reference)
│   └── exp12/16/21_margin_*.npy  ← 16 per-event margin series (24 series total)
├── scripts/
│   ├── golden_master_check.py    ← mechanical checker: 98 keys zero-drift
│   ├── exp*.py                   ← experiment scripts (fitting, protocol, diagnostics)
│   └── exp8_neural.sh            ← neural comparison pipeline (DyGLib)
├── src/
│   └── lanl_pp.py                ← core library: intensity, Omori kernel, IGPE, simulator
├── logs/neural/                  ← DyGLib logs (EdgeBank/TGAT/TGN AP ± SD, 5/2 runs)
└── data/
    └── lanl_smoke.csv.*.part     ← 1,488,080-event NTLM subset (split; see §5)
```

## 2. Claim-to-artifact mapping

| Paper claim (section) | Artifact | How to verify |
|---|---|---|
| Event-level IGPE: hybrid vs uniform **+5.87 bits** (§V-A) | `results/exp12_boost_flag.json::test_IGPE` | key lookup |
| Boost/flag margins **+2.32 / +0.90** (§V-A) | `results/exp12_boost_flag.json::margins` | key lookup |
| Boost share **0.72** (§V-A) | same JSON: `R3_share_boost` | key lookup |
| Optimism-adjusted **[4.72, 5.59]** (§IV-B) | `results/exp37_adaptive_optimism.json::deflation` | key lookup |
| **98-key frozen master** zero-drift (§VI) | `scripts/golden_master_check.py` | run: `python3 scripts/golden_master_check.py` — exit 0 |
| **57 logged comparisons** → K_eff=3 (§IV-B) | `results/exp37_adaptive_optimism.json::registry` | registry audit |
| **24 per-event series** (§VI) | `results/*.npy` | `ls results/*.npy \| wc -l` → 24 |
| Decision rule thresholds (§IV-A) | `results/exp12_boost_flag.json::preregistered_rules` | rules R1–R4 |
| Six-face reconciliation (Table III) | `results/exp18_skill_scale_protocol.json`, `exp20_hod_true_integral.json` | key lookup per face |
| Neural AP ± SD (EdgeBank/TGAT/TGN, Fig. 3) | `logs/neural/exp8_*.log` | grep `average test average_precision` |
| Margin histogram data (Fig. 2) | `results/exp12_margin_M_hybrid_W24_c_gt_M_flag.npy` | load with numpy |

## 3. Quick verification (numpy + scipy only)

```bash
# 1. Frozen master check
python3 scripts/golden_master_check.py

# 2. Boost/flag decomposition (mirror statistics in §V-A)
python3 -c "
import numpy as np, json
from scipy.stats import wilcoxon
mf = np.load('results/exp12_margin_M_hybrid_W24_c_gt_M_flag.npy')
mb = np.load('results/exp12_margin_M_hybrid_W24_c_gt_M_boost_W24_c.npy')
print(f'vs flag:  median={np.median(mf):.4f}  pos/neg={int((mf>0).sum())}/{int((mf<0).sum())}')
print(f'vs boost: median={np.median(mb):.4f}  pos/neg={int((mb>0).sum())}/{int((mb<0).sum())}  p={wilcoxon(mb).pvalue:.1e}')
d = json.load(open('results/exp12_boost_flag.json'))
print(f'share_boost={d[\"R3_share_boost\"]:.4f}  verdict={d[\"verdict\"]}')
"

# 3. Marginal distribution shape (Fig. 2 data)
python3 -c "
import numpy as np
mf = np.load('results/exp12_margin_M_hybrid_W24_c_gt_M_flag.npy')
pos = np.sort(mf[mf>0])[::-1]
print(f'top-10 events carry {pos[:10].sum()/mf.sum()*100:.1f}% of net gain')
"
```

## 4. Model specification

All parameters are in `results/exp12_boost_flag.json::model_family` (hybrid) and `results/exp4b_rescored.json` (hop-1 reference). Fitting is constrained L-BFGS-B (`p≤2`, `c∈[10,3600] s`) via `src/lanl_pp.py`. The Omori kernel is not unit-normalized; the branching ratio `a = κ∫₀ᵂ g` uses the finite-support convention (hop-1) or the window-constant variant (hybrid).

## 5. Neural comparison subset

`data/lanl_smoke.csv.*.part` — the 1,488,080-event NTLM subset used for Fig. 3 (DyGLib protocol). Reassemble:
```bash
cat data/lanl_smoke.csv.*.part > lanl_smoke.csv  # 26 MB, 1,488,081 lines (incl. header)
```
**Provenance:** the subset was extracted during the experimental campaign; the exact filter script is not separately archived. The subset file and its DyGLib preprocessing are shipped as-is; `scripts/exp8_neural.sh` reproduces the full neural evaluation pipeline on this file.

## 6. Environment

| Component | Version |
|---|---|
| Python | ≥ 3.9 |
| numpy / scipy | ≥ 1.24 / ≥ 1.10 |
| DyGLib (neural comparison only) | NeurIPS 2023, arXiv:2303.13047 |
| Data | LANL cyber1 (Kent 2015, DOI 10.17021/1179829) — publicly available |

## 7. License

- Code: MIT License.
- Data: LANL cyber1 is publicly released by Los Alamos National Laboratory. This repository is not an official LANL product.
- Internal experiment IDs (exp12, exp37, ...) in JSON key names map to paper sections via the table in §2.
