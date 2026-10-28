# WAMAC macro-anchor diagnostic MVP

Offline, dependency-free diagnostic for proposal SeaSealji/sst_idea#5. This is
not a trained SST model, a NeST reproduction, or evidence of ocean-model gains.
No original data, checkpoints, private infrastructure details or credentials.

For disjoint regions, with the **same positive weights for pooling and loss**:

`MSE(base-truth) = within-region anomaly MSE + regional mean-error MSE`.

Replacing each base regional mean with a predicted guide keeps the within-region
anomaly term unchanged; only the regional mean-error term changes. A guide that
beats persistence may still be worse than pooled base predictions. The tool
scores fixed correction gains, their analytical loss derivatives, and a clearly
labelled validation-only oracle bound. It does not select a gain or write a model.

The existing ocean score weights wet cells equally, while existing auxiliary
regional training uses cosine-latitude weights. Run these protocols separately;
never mix their pooling means in an asserted orthogonal loss decomposition.

## Run

`python3 -m unittest -v`

`python3 macro_anchor_diagnostic.py explicit_validation_export.jsonl`

Input is one flattened spatial frame per JSON line, for one origin and D1-D7:

```json
{"split":"validation","guide_source":"history_only_prediction","forecast_origin":"2018-01-01","guide_input_max_date":"2018-01-01","target_date":"2018-01-02","lead":1,"weighting":"uniform_wet_cells","base":[2,4,8,10],"truth":[1,3,9,11],"region":[0,0,1,1],"guide":{"0":2,"1":10}}
```

For a separate area-weighted diagnostic use `weighting=explicit_area_weights`
and a positive weight per wet cell. Region -1 denotes ignored land. Regions
must be constructed and checked for wet connectivity upstream; this tool accepts
an explicit region map and does not infer coastline geometry from temperatures.

CLI records are restricted to forecast origins and targets in 2017-2021, with
guide input dates no later than the forecast origin. Exporter metadata alone
cannot prove absence of leakage: independently verify exporter input slicing,
guide training, checkpoint provenance and masks. Never feed oracle corrections
back into inference or use held-out test values for these diagnostics.

The current local experiment archive contains per-window scalar MSE/MAE, not
spatial forecasts/truth/guide means. Real-data diagnostics therefore remain
pending an authorized validation-only export. Synthetic tests verify algebra,
analytical derivatives, shape checks, masks and time guards; they do not validate
a forecast model's forward/backward passes or its skill.
