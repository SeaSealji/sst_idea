# Validation record — 2026-11-03

Status: **CPU algebra/interface PASS; training and SST skill untested.**

Environment: Python 3.10, PyTorch 2.2.2+cu118, CPU only, two Torch threads. Unit suite: 12 tests, zero skips, 0.532 s. Private synthetic/static integration: 3.028 s, zero optimizer updates. CUDA build availability is not GPU verification.

| Check | Evidence |
|---|---|
| Linear sum vs concatenated feature head | Maximum float64 difference 2.220446049250313e−15; unit suite also checks all parameter/input gradients |
| Projected residual Dδ=c | Global wet core float64 maximum 8.881784197001252e−16 |
| Whole-support area SSE partition | Absolute residual 7.275957614183426e−12; cross term 9.592326932761353e−14 |
| Capacity | Sum and project: 448 parameters each; inactive constant fine bias omitted |
| Actual static support | 180×360 grid, 43,467 wet cells → 11,424 coarse components in 11,412 occupied bins |
| Synthetic full-model shape | [1,7,1,244,424] → [1,7,1,244,424], includes halos |
| Fine/coarse gradients | All fine-head parameters finite/nonzero; coarse weight gradient L1=0.00706354 |
| Full-model residual consistency | Float32 maximum 2.9802322387695312e−8 |
| No-grad replay | Maximum difference 0 |
| Full-model parameter count | 138,631 baseline; 138,848 diagnostic (−7 fine bias +224 coarse weights) |

The unit suite additionally checks double-precision gradcheck, preserved fine anomalies, one anchor addition, normalized-versus-Celsius counterexample, subset-mask loss of orthogonality, ineffective fine bias, invalid shapes/nonfinite wet residuals, disconnected wet-component and batch gradient isolation, and dry-NaN handling in geometry. Geometry isolation is local to the declared graph; it does not prove whole-model land isolation.

The full-model harness uses legal pooling of already fused features, not the original independent coarse context path. No learned checkpoint or SST arrays were loaded. The inherited static mask's train-only provenance is not established by this diagnostic. No data leakage or metric error was observed because no predictive experiment was run; future training still requires data/metric provenance gates.

Protocol identity was rechecked against the active registry before the harness:

- Protocol SHA256: `d81150be609e7feeda89e124d546c975ec37491308c11a96305fcf2cdb5c24fe`
- T1-no-expert config SHA256: `ae6a121ab7e23ffe8a59e370a1d1df409534741c5cd291a89e267923a8f6a3c1`
- Model source SHA256: `3bfc23769d103712f6dcacf321e76e6ab587918950451cc8f6004baa92c4887c`

No D1–D7, spring, coastal or offshore performance result exists for this candidate. These numerical tolerances are structural checks, not forecasting scores. Historical test exposure must not be described as a new untouched test; this run read no test values. Current best models and previous negative results remain retained.
