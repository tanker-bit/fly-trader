# Evaluation suite   (2026-11-08 01:25)

33 signals, 120 non-overlapping monthly decision dates (2013-12-12 to 2023-11-16); holdout untouched.

## Signal tests (mean monthly rank-IC; p from a sign-flip test on the monthly IC series)

| signal | IC | IC sd | t | p | p Holm | p BH | p maxT | ICs>0 |
|---|---|---|---|---|---|---|---|---|
| random dense + ridge | +0.0160 | 0.100 | +1.76 | 0.083 | 1.000 | 0.596 | 0.988 | 55% |
| dist_52w_high | -0.0362 | 0.226 | -1.75 | 0.080 | 1.000 | 0.596 | 0.313 | 49% |
| ma50_dist | -0.0285 | 0.180 | -1.74 | 0.087 | 1.000 | 0.596 | 0.626 | 51% |
| vol_20 | +0.0302 | 0.216 | +1.53 | 0.130 | 1.000 | 0.596 | 0.551 | 52% |
| ma200_dist | -0.0293 | 0.218 | -1.47 | 0.146 | 1.000 | 0.596 | 0.589 | 48% |
| ret_60 | -0.0253 | 0.189 | -1.47 | 0.146 | 1.000 | 0.596 | 0.762 | 52% |
| random KC code + ridge | +0.0127 | 0.104 | +1.34 | 0.181 | 1.000 | 0.596 | 0.999 | 56% |
| ret_20 | -0.0213 | 0.183 | -1.28 | 0.205 | 1.000 | 0.596 | 0.895 | 48% |
| accruals | -0.0104 | 0.092 | -1.23 | 0.222 | 1.000 | 0.596 | 1.000 | 47% |
| ridge raw | +0.0173 | 0.155 | +1.22 | 0.224 | 1.000 | 0.596 | 0.976 | 56% |
| ret_120 | -0.0226 | 0.205 | -1.21 | 0.227 | 1.000 | 0.596 | 0.858 | 48% |
| mom_12_1 | -0.0234 | 0.219 | -1.17 | 0.246 | 1.000 | 0.596 | 0.831 | 45% |
| momentum 12-1 | -0.0234 | 0.219 | -1.17 | 0.246 | 1.000 | 0.596 | 0.831 | 45% |
| rsi_14 | -0.0185 | 0.178 | -1.14 | 0.253 | 1.000 | 0.596 | 0.957 | 52% |
| book_to_price | -0.0152 | 0.171 | -0.97 | 0.334 | 1.000 | 0.706 | 0.992 | 46% |
| days_since_filing | +0.0057 | 0.066 | +0.95 | 0.342 | 1.000 | 0.706 | 1.000 | 50% |
| eps_growth_yoy | -0.0073 | 0.106 | -0.76 | 0.453 | 1.000 | 0.753 | 1.000 | 49% |
| gross_margin | +0.0067 | 0.107 | +0.68 | 0.494 | 1.000 | 0.753 | 1.000 | 47% |
| quality (ROE+ROA+GM) | +0.0096 | 0.157 | +0.67 | 0.508 | 1.000 | 0.753 | 1.000 | 47% |
| ret_1 | -0.0115 | 0.190 | -0.66 | 0.508 | 1.000 | 0.753 | 1.000 | 45% |
| reversal (-ret_1) | +0.0115 | 0.190 | +0.66 | 0.508 | 1.000 | 0.753 | 1.000 | 55% |
| earnings_yield | -0.0083 | 0.156 | -0.58 | 0.561 | 1.000 | 0.753 | 1.000 | 48% |
| fcf_yield | +0.0067 | 0.127 | +0.58 | 0.568 | 1.000 | 0.753 | 1.000 | 47% |
| rev_growth_yoy | +0.0072 | 0.145 | +0.54 | 0.588 | 1.000 | 0.753 | 1.000 | 54% |
| value (EY+BP+FCF) | -0.0086 | 0.182 | -0.52 | 0.610 | 1.000 | 0.753 | 1.000 | 46% |
| roe | +0.0062 | 0.134 | +0.50 | 0.611 | 1.000 | 0.753 | 1.000 | 46% |
| roa | +0.0076 | 0.165 | +0.50 | 0.616 | 1.000 | 0.753 | 1.000 | 48% |
| volume_ratio | -0.0030 | 0.080 | -0.40 | 0.688 | 1.000 | 0.810 | 1.000 | 46% |
| debt_to_equity | +0.0026 | 0.090 | +0.32 | 0.749 | 1.000 | 0.823 | 1.000 | 54% |
| low leverage (-D/E) | -0.0026 | 0.090 | -0.32 | 0.749 | 1.000 | 0.823 | 1.000 | 46% |
| vol_ratio | -0.0009 | 0.120 | -0.08 | 0.933 | 1.000 | 0.941 | 1.000 | 47% |
| ret_5 | -0.0014 | 0.195 | -0.08 | 0.937 | 1.000 | 0.941 | 1.000 | 52% |
| growth (rev+eps yoy) | +0.0009 | 0.137 | +0.07 | 0.941 | 1.000 | 0.941 | 1.000 | 53% |

Read the corrected columns (Holm, maxT), not raw p: with this many signals a few will look good by chance.

## Power: what could this data have detected? (real month-to-month IC dispersion)

| planted IC | detection rate |
|---|---|
| 0.00 | 4% |
| 0.01 | 17% |
| 0.02 | 40% |
| 0.03 | 71% |
| 0.04 | 86% |
| 0.05 | 95% |
| 0.07 | 100% |
| 0.10 | 100% |

False-positive rate at IC 0 (placebo): 4% (should be near 5%). Smallest planted IC found in >= 80% of trials: 0.04.

## Decile long-short (net of costs, monthly, equal weight)

| signal | gross/m | net Sharpe @0bp | @10bp | @25bp | maxDD @10bp | turnover |
|---|---|---|---|---|---|---|
| random dense + ridge | +0.56% | +0.51 | +0.37 | +0.16 | -26.5% | 76% |
| dist_52w_high | -1.40% | -0.68 | -0.72 | -0.80 | -189.2% | 51% |
| ma50_dist | -1.14% | -0.66 | -0.74 | -0.87 | -163.3% | 72% |
| vol_20 | +0.98% | +0.55 | +0.48 | +0.37 | -41.1% | 65% |
| ma200_dist | -1.05% | -0.54 | -0.58 | -0.64 | -136.1% | 39% |
| ret_60 | -0.92% | -0.52 | -0.58 | -0.67 | -137.4% | 55% |
| random KC code + ridge | +0.43% | +0.48 | +0.30 | +0.04 | -26.0% | 78% |
| ret_20 | -0.70% | -0.42 | -0.53 | -0.69 | -113.7% | 88% |
| accruals | -0.38% | -0.56 | -0.59 | -0.64 | -55.7% | 11% |
| ridge raw | +0.54% | +0.39 | +0.31 | +0.20 | -27.9% | 52% |

Caveats: the panel is not fully survivorship-free (delisted names are missing), costs are a flat bps per unit traded with no market impact or borrow cost on shorts, and the top-10 list above was chosen AFTER seeing the results, so its Sharpe ratios are optimistic.