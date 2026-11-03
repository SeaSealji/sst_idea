# WMLF decoder Z0: distinguish output fusion from coarse/detail constraints

This is a zero-training diagnostic for [Issue #5](https://github.com/SeaSealji/sst_idea/issues/5), including its 2026-11-03 STORM/WMLF supplement. It refines the earlier STORM discussion rather than treating the paper as a new discovery. The research hypothesis is that independently forecast coarse residuals plus explicitly zero-mean local detail may improve D1–D7 SST errors; simple summation by itself provides neither this constraint nor new linear expressivity.

## What the reference implements

`geometry.py` reuses our wet-component hierarchy from the prior wet-graph diagnostic (PR #23). Within each 2×2 bin, disconnected wet components remain separate. D is wet-area pooling, U is piecewise-constant prolongation, and DU=I. Features entering the decoder must already have legal support. This module cannot undo upstream cross-land convolution or certify polar topology.

For fine residual f and coarse residual c:

- Sum control: δ = f + Uc. In general Dδ = Df + c, not c.
- Projection candidate: δ = Uc + (I−UD)f, so Dδ = c in the supplied field coordinates.
- Add the persistence anchor a exactly once outside both heads: Y = a + δ, hence DY = Da + c. The coarse head predicts a residual, not the whole future field.

`LinearScaleDecoder` maps fine/coarse features of width 32 into seven future residuals. Sum and projection each have 448 trainable parameters, with both output biases omitted. A constant fine bias would be annihilated by I−UD; counting it as effective capacity would be misleading. Equal parameter counts alone do not establish equal FLOPs or memory.

For linear heads with fixed U, Wf F + U(Wc C) = [Wf,Wc][F,U(C)]. The tests check both outputs and input/parameter derivatives. This rules out an expressivity claim for this specific linear decomposition; it does not establish equivalence of nonlinear multiscale encoders/heads, different feature taps, losses, optimization, or full STORM.

With the same area weights and complete declared aggregates, weighted squared error partitions into coarse/background and zero-mean detail terms. The identity need not hold for the primary uniform-grid metric, coastal subsets cutting through aggregates, a different mask, or a different coordinate system. `error_partition` deliberately scores the complete supplied support.

## Units are part of the hypothesis

Correction (2026-11-03): the active protocol uses a grid-cell/day-of-year climatology and a **global scalar** σ_train=0.7119533415268037 °C, as verified in the frozen protocol and cache window metadata. The earlier wording incorrectly called σ per-grid. For this scalar normalization, D(σr)=σDr, so normalized residual consistency transfers exactly to Celsius residual consistency; a coarse head predicting c in normalized units represents σc in Celsius. Full-field consistency also includes the once-added anchor and target-day climatology. This is an algebraic unit statement, not physical heat conservation.

The retained synthetic counterexample applies only to a possible future spatially varying σ: two equal-area cells with r=[1,−1], c=0, σ=[1,3] give Dr=0 but D(σr)=−1. It is not a diagnosed defect in the current SST protocol. Any future local-normalization revision must define coarse units explicitly. STORM output summation still does not impose a coarse/detail projection. Keep uniform wet-grid RMSE primary and area-weighted scores separate.

## Related work and evidence boundaries

| Source | Relevant overlap | Differentiation to test |
|---|---|---|
| [STORM, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/file/9ad996b5c45130de2bc00b60d8607904-Paper-Conference.pdf), §§4.1–4.3 | Spatial pyramid, nonlinear cross-scale encoder, independent T→L heads, depth-dependent upsampling, shared variable projection, EarthMix direct sum | Wet-support geometry, explicit coarse/detail residual constraint, units audit, and matched concat controls on SST-only 7→7 |
| [NeST, AISTATS 2026](https://proceedings.mlr.press/v306/ai26f.html), official abstract | Future macro-region trends guiding micro-region forecasts with spectral regions | No claim that coarse-to-fine forecasting itself is new; isolate local wet-area constraints rather than copying future-trend conditioning |
| [SimVP, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Gao_SimVP_Simpler_Yet_Better_Video_Prediction_CVPR_2022_paper.html), official abstract | Direct multi-frame prediction with CNNs and MSE | Direct seven-day heads and convolution alone are not novelty |

STORM's Table 1 T2M RMSE 0.675 averages 6/12/18/24 h; it is not an isolated 24 h score. Table 2 T2M 2.596 versus SimVP 3.168 averages 168/192/216/240 h. These atmospheric, multi-variable results cannot be compared numerically to our SST-only validation. Its `w/o M` ablation jointly changes embedder, encoder and decoder, so it does not isolate EarthMix. No SST coastal benefit is established there. The PDF's Eq. 5 has a notation/dimension ambiguity (paragraph uses cross-scale features while the displayed input is spatial features); no author code was available to resolve it. The linked STORM GitHub repository returned 404 on this run; no author implementation audit is claimed.

Reading scope: focused full-PDF sections above, §5.3 and Tables 1–2; Figure 2 and pages 5/7 visually checked. NeST and SimVP comparisons are abstract-level only. Private source hashes and fetch records are archived separately.

## Integration status

A private CPU harness connects projection to the active T1-no-expert interface with synthetic inputs, static wet mask, and randomly initialized weights. Its coarse features are **D(fused features)**. The original stride-2 context tensor is not tapped: it has different support from component-split ragged wet nodes. This harness is not a complete WMLF implementation or an independent coarse representation. Ordinary upstream/downstream convolutions and reflected halos retain their inherited limitations.

Current baseline is T1-no-expert under the active global 1° SST-only 7→7 protocol, not historical T1-Fused. No model/checkpoint was replaced. See [VALIDATION.md](VALIDATION.md) and [ablation_plan.json](ablation_plan.json) for results and gates. No SST targets, validation targets, sealed tests, learned checkpoints, GPU training or optimizer updates were used.

Run from this directory with NumPy and PyTorch:

```sh
python -m unittest -v test_decoder
```
