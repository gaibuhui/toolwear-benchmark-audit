# toolwear-benchmark-audit

Reproducibility package for the paper **"Is the Value of Sensor Signals in Tool-Wear Prediction Overstated? A Protocol, Label-Granularity and Baseline Audit"** (IEEE TIM submission).

The paper audits five public machine-condition-monitoring corpora and separates three causes of the gap between published accuracy and industrial reality: missing zero-training probes, protocol bias, and label measurement granularity.

## Key findings

| # | Finding | Evidence (script) |
|---|---------|-------------------|
| N1 | A zero-sensor 4-variable regression (cumulative cutting time, depth of cut, feed, material) reaches R² = 0.927 on NASA milling, above two published deep sensor-based models (0.9069–0.9105); adding 66 sensor features to a Taylor-form physics model *increases* MAE by 34% | `nasa_baselines.py`, `nasa_taylor.py`, `nasa_cnn.py` |
| N2 | The strength of the persistence baseline is an identity: MAE ≡ E\|ΔW\| = label range × normalized step size. Four corpora label the *same physical quantity* (flank wear), yet persistence MAE spans 0.047–77.8 µm (a factor of 1660) and R² spans 0.850–1.0000, purely as a function of how the label was measured and resampled | `label_granularity.py` |
| N3 | Horizon-increment protocol: sensor signals do not buy accuracy; they buy a longer permissible interval between measurements (−37…−53% error for horizons ≥ 25 cuts) | `paper_protocol_sweep.py` (P5) |
| N4 | Zero-training probes: persistence wins all 28 protocol×model cells on the Mendeley milling corpus (MAE 0.212–0.264 µm = 0.14% of range); kNN(1) = 1.000 under a random split on CWRU, 0.999 leave-one-load, 0.519 leave-one-severity | `paper_protocol_sweep.py`, `cwru_binary.py` |

Protocol bias: the same multilayer perceptron moves from 4.73 µm (random split) to 69.83 µm (leave-one-machine-out), **+1376%**.

## Repository layout

```
scripts/   experiment code (run from the repository root; data lands in data/)
data/      download instructions; the corpora themselves are NOT committed
```

## Data

All five corpora are public. See [`data/README.md`](data/README.md) for exact download
commands, sizes, licenses and caveats (one is 12 GB; the PHM 2010 original is access-gated
and is used here through the Katulu Uniwear redistribution).

## Reproduction

Run from the repository root after downloading the data.

| Paper table | Script(s) | Notes |
|---|---|---|
| D1 features (6,418 × 138) | `ab_labels.py` → `ab_extract_all.py` | `data/toolwear/ab_features.npz` |
| Tab. III protocol×baseline matrix (P1–P4) | `paper_protocol_sweep.py`, `persist_ref.py` | persistence conventions; `fix_ridge.py` documents the intercept fix |
| Tab. IV GRU variants (B4a/B4b/B4c) | `paper_gru_v2.py`, `paper_gru_delta.py` | hidden-state warm-up, target standardisation, residual anchoring |
| Tab. V / §NASA | `nasa_baselines.py`, `nasa_taylor.py`, `nasa_cnn.py` | zero-sensor regression, Taylor law, raw-waveform CNN |
| Tab. VI CWRU | `dl_cwru12k.py` → `cwru_protocol.py`, `cwru_protocol2.py`, `cwru_binary.py` | P1/P2/P6 protocol gradient with kNN(1) probe |
| Tab. VII label granularity | `dl_uniwear.py` → `label_granularity.py` | D1–D4 persistence identity |

`diag_nasa.py` and `inspect_uniwear.py` are small dataset-inspection utilities.

## Environment

- Python 3.9+
- `numpy>=1.24,<2.0` (numpy 2.x breaks the pinned stack used in development)
- `pyarrow`, `torch>=2.0` (CPU is sufficient; every result in the paper runs on a single RTX 4060 8 GB or CPU)
- `huggingface_hub` for the download scripts

```bash
pip install -r requirements.txt
```

## Revision additions (v1.1, 2026-09-13)

- `scripts/d2_horizon.py` — horizon-increment protocol replicated on D2 (NASA milling):
  persistence 77.8 µm vs ΔCT 45.9 µm (−41%) vs ΔCT×condition 34.3 µm (−56%) at h=1; −70% at h=4.
- `scripts/d1_fill_sensitivity.py` — forward-fill sensitivity: persistence MAE 0.204 → 1.110 µm
  (5.5×) under a time-to-event framing; R² 0.9997 → 0.9991.
- `scripts/paper_stats_dump.py` → `data/toolwear/stats_dump.npz` — per-sample signed errors for
  every (protocol, model) cell, one seed, exact original configurations.
- `scripts/paper_stats_test.py` → `stats_results.md` — Wilcoxon signed-rank p and bootstrap 95% CI
  for all 28 protocol×model comparisons vs persistence (all significant; margins +0.019 to +67.7 µm).

## Revision additions (v1.2, 2026-09-13)

Addressing the second-round review on D2 (NASA milling, 146 labelled records):

- `scripts/nasa_feat_cache.py` — builds `nasa_feat.npz` (66 sensor features, 4 process variables,
  grouping ids, and a per-record data-quality probe `ROWMAX`).
- `scripts/nasa_feat_ablation.py` — three checks:
  1. **feature-count-matched ablation**: random sensor subsets of size k = 4/10/20 (ten draws each)
     give MAE 202.2 ± 25.3 / 193.8 ± 116.1 / 137.1 ± 59.0 µm against **50.6 µm** for the four
     process variables under the identical estimator — so the sensor arm does not lose merely because
     it has more features;
  2. **data quality**: exactly one of the 146 records carries samples of order 1e34 on all six
     channels, which alone makes 23 of the 66 feature columns numerically inert; excluding it changes
     no ordering (115.3 vs 50.6 µm);
  3. **importance and partial correlation**: the features are not noise (median |ρ| with the label
     0.352, LASSO retains 25/66) but their information is largely a proxy for the process variables
     (median |ρ| with the residual drops to 0.163; 3 of 66 exceed 0.3).

  The two scripts need different environments: `nasa_feat_cache.py` requires `pyarrow`,
  `nasa_feat_ablation.py` requires `scikit-learn` and `scipy`.

## License

MIT (code). Dataset licenses and citation requirements are listed in `data/README.md`.
