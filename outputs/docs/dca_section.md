## Decision Curve Analysis (DCA)

### Decision Curve Analysis — h=24h, split=val

- Thresholds shown at ~0.05 / 0.10 / 0.20 (nearest grid from CSV).
- `net benefit` per-patient (or per 100 patients if you ran with `--per-100`).

| Variant | Threshold | Net benefit (Model) | Treat-all | Treat-none |
|---|---:|---:|---:|---:|
| raw | 0.050 | 0.1803 | -3.4450 | 0.0000 |
| raw | 0.100 | 0.1909 | -9.1920 | 0.0000 |
| raw | 0.200 | 0.0283 | -22.8410 | 0.0000 |
| isotonic | 0.050 | 0.3388 | -3.4450 | 0.0000 |
| isotonic | 0.100 | 0.2244 | -9.1920 | 0.0000 |
| isotonic | 0.200 | 0.1628 | -22.8410 | 0.0000 |
| sigmoid | 0.050 | 0.1724 | -3.4450 | 0.0000 |
| sigmoid | 0.100 | 0.1279 | -9.1920 | 0.0000 |
| sigmoid | 0.200 | 0.1345 | -22.8410 | 0.0000 |

**Curves**
![h=24, val, raw](/public/home/aojiang/mimic/lymphoma-ewarn/outputs/figures/dca_h24_val.png)
![h=24, val, isotonic](/public/home/aojiang/mimic/lymphoma-ewarn/outputs/figures/dca_h24_val_cal_isotonic.png)
![h=24, val, sigmoid](/public/home/aojiang/mimic/lymphoma-ewarn/outputs/figures/dca_h24_val_cal_sigmoid.png)

### Decision Curve Analysis — h=24h, split=test

- Thresholds shown at ~0.05 / 0.10 / 0.20 (nearest grid from CSV).
- `net benefit` per-patient (or per 100 patients if you ran with `--per-100`).

| Variant | Threshold | Net benefit (Model) | Treat-all | Treat-none |
|---|---:|---:|---:|---:|
| raw | 0.050 | 1.2371 | 0.3376 | 0.0000 |
| raw | 0.100 | 0.2109 | -5.1992 | 0.0000 |
| raw | 0.200 | -0.0540 | -18.3491 | 0.0000 |
| isotonic | 0.050 | 2.1992 | 0.3376 | 0.0000 |
| isotonic | 0.100 | -0.1134 | -5.1992 | 0.0000 |
| isotonic | 0.200 | -0.3753 | -18.3491 | 0.0000 |
| sigmoid | 0.050 | 1.3452 | 0.3376 | 0.0000 |
| sigmoid | 0.100 | 0.2936 | -5.1992 | 0.0000 |
| sigmoid | 0.200 | -0.3543 | -18.3491 | 0.0000 |

**Curves**
![h=24, test, raw](/public/home/aojiang/mimic/lymphoma-ewarn/outputs/figures/dca_h24_test.png)
![h=24, test, isotonic](/public/home/aojiang/mimic/lymphoma-ewarn/outputs/figures/dca_h24_test_cal_isotonic.png)
![h=24, test, sigmoid](/public/home/aojiang/mimic/lymphoma-ewarn/outputs/figures/dca_h24_test_cal_sigmoid.png)

### Decision Curve Analysis — h=48h, split=val

- Thresholds shown at ~0.05 / 0.10 / 0.20 (nearest grid from CSV).
- `net benefit` per-patient (or per 100 patients if you ran with `--per-100`).

| Variant | Threshold | Net benefit (Model) | Treat-all | Treat-none |
|---|---:|---:|---:|---:|
| raw | 0.050 | 0.2330 | -2.9880 | 0.0000 |
| raw | 0.100 | 0.1290 | -8.7096 | 0.0000 |
| raw | 0.200 | 0.1062 | -22.2983 | 0.0000 |
| isotonic | 0.050 | 0.3487 | -2.9880 | 0.0000 |
| isotonic | 0.100 | 0.2192 | -8.7096 | 0.0000 |
| isotonic | 0.200 | 0.1840 | -22.2983 | 0.0000 |
| sigmoid | 0.050 | 0.2295 | -2.9880 | 0.0000 |
| sigmoid | 0.100 | 0.1657 | -8.7096 | 0.0000 |
| sigmoid | 0.200 | 0.1109 | -22.2983 | 0.0000 |

**Curves**
![h=48, val, raw](/public/home/aojiang/mimic/lymphoma-ewarn/outputs/figures/dca_h48_val.png)
![h=48, val, isotonic](/public/home/aojiang/mimic/lymphoma-ewarn/outputs/figures/dca_h48_val_cal_isotonic.png)
![h=48, val, sigmoid](/public/home/aojiang/mimic/lymphoma-ewarn/outputs/figures/dca_h48_val_cal_sigmoid.png)

### Decision Curve Analysis — h=48h, split=test

- Thresholds shown at ~0.05 / 0.10 / 0.20 (nearest grid from CSV).
- `net benefit` per-patient (or per 100 patients if you ran with `--per-100`).

| Variant | Threshold | Net benefit (Model) | Treat-all | Treat-none |
|---|---:|---:|---:|---:|
| raw | 0.050 | 3.2239 | 3.8269 | 0.0000 |
| raw | 0.100 | 1.6348 | -1.5160 | 0.0000 |
| raw | 0.200 | 0.7657 | -14.2055 | 0.0000 |
| isotonic | 0.050 | 4.3478 | 3.8269 | 0.0000 |
| isotonic | 0.100 | 1.1650 | -1.5160 | 0.0000 |
| isotonic | 0.200 | 0.5495 | -14.2055 | 0.0000 |
| sigmoid | 0.050 | 2.4969 | 3.8269 | 0.0000 |
| sigmoid | 0.100 | 1.5801 | -1.5160 | 0.0000 |
| sigmoid | 0.200 | 0.7326 | -14.2055 | 0.0000 |

**Curves**
![h=48, test, raw](/public/home/aojiang/mimic/lymphoma-ewarn/outputs/figures/dca_h48_test.png)
![h=48, test, isotonic](/public/home/aojiang/mimic/lymphoma-ewarn/outputs/figures/dca_h48_test_cal_isotonic.png)
![h=48, test, sigmoid](/public/home/aojiang/mimic/lymphoma-ewarn/outputs/figures/dca_h48_test_cal_sigmoid.png)

