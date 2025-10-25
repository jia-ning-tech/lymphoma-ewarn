## Ablation Studies

This section unifies **keep-only** and **drop-one** ablation; all points are 5-fold CV **mean ± std**.

### Ablation (h=24)

| Setting | Group | #Features | AUROC(mean±std) | AP(mean±std) |
|---|---:|---:|---:|---:|
| baseline_all | - | 372 | 0.7974 ± 0.0228 | 0.1303 ± 0.0398 |
| drop-one | labs | 260 | 0.7646 ± 0.0274 | 0.1260 ± 0.0447 |
| drop-one | others | 224 | 0.8021 ± 0.0262 | 0.1249 ± 0.0360 |
| drop-one | vent | 358 | 0.7225 ± 0.0465 | 0.0773 ± 0.0148 |
| drop-one | vitals | 274 | 0.7711 ± 0.0276 | 0.0696 ± 0.0197 |
| keep-only | labs | 112 | 0.6662 ± 0.0444 | 0.0488 ± 0.0091 |
| keep-only | others | 148 | 0.4985 ± 0.0503 | 0.0296 ± 0.0076 |
| keep-only | vent | 14 | 0.7719 ± 0.0236 | 0.0686 ± 0.0156 |
| keep-only | vitals | 98 | 0.6604 ± 0.0494 | 0.0602 ± 0.0197 |
