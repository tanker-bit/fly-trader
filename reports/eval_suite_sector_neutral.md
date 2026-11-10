# Evaluation suite   (2026-11-08 10:15)

33 signals, 120 non-overlapping monthly decision dates (2013-12-12 to 2023-11-16); holdout untouched. SECTOR-NEUTRAL (labels and scores demeaned within GICS sector).

## Signal tests (mean monthly rank-IC; p from a sign-flip test on the monthly IC series)

| signal | IC | IC sd | t | p | p Holm | p BH | p maxT | ICs>0 |
|---|---|---|---|---|---|---|---|---|
| random dense + ridge | +0.0160 | 0.076 | +2.29 | 0.024 | 0.777 | 0.777 | 0.353 | 59% |
| accruals | -0.0103 | 0.075 | -1.51 | 0.132 | 1.000 | 0.831 | 0.902 | 39% |
| vol_20 | +0.0217 | 0.180 | +1.32 | 0.195 | 1.000 | 0.831 | 0.966 | 49% |
| dist_52w_high | -0.0228 | 0.202 | -1.24 | 0.218 | 1.000 | 0.831 | 0.981 | 52% |
| ridge raw | +0.0136 | 0.122 | +1.22 | 0.228 | 1.000 | 0.831 | 0.984 | 55% |
| debt_to_equity | +0.0067 | 0.063 | +1.15 | 0.245 | 1.000 | 0.831 | 0.991 | 58% |
| low leverage (-D/E) | -0.0067 | 0.063 | -1.15 | 0.245 | 1.000 | 0.831 | 0.991 | 42% |
| earnings_yield | -0.0147 | 0.140 | -1.15 | 0.252 | 1.000 | 0.831 | 0.991 | 46% |
| rsi_14 | -0.0131 | 0.136 | -1.06 | 0.292 | 1.000 | 0.831 | 0.997 | 44% |
| ma200_dist | -0.0187 | 0.194 | -1.05 | 0.293 | 1.000 | 0.831 | 0.997 | 51% |
| value (EY+BP+FCF) | -0.0142 | 0.162 | -0.96 | 0.338 | 1.000 | 0.831 | 0.999 | 43% |
| ret_120 | -0.0151 | 0.181 | -0.92 | 0.355 | 1.000 | 0.831 | 1.000 | 52% |
| ret_60 | -0.0131 | 0.158 | -0.91 | 0.367 | 1.000 | 0.831 | 1.000 | 54% |
| days_since_filing | +0.0051 | 0.063 | +0.90 | 0.363 | 1.000 | 0.831 | 1.000 | 54% |
| ma50_dist | -0.0122 | 0.152 | -0.88 | 0.378 | 1.000 | 0.831 | 1.000 | 57% |
| book_to_price | -0.0115 | 0.153 | -0.83 | 0.409 | 1.000 | 0.837 | 1.000 | 41% |
| ret_20 | -0.0106 | 0.147 | -0.79 | 0.436 | 1.000 | 0.837 | 1.000 | 54% |
| mom_12_1 | -0.0122 | 0.189 | -0.71 | 0.482 | 1.000 | 0.837 | 1.000 | 50% |
| momentum 12-1 | -0.0122 | 0.189 | -0.71 | 0.482 | 1.000 | 0.837 | 1.000 | 50% |
| rev_growth_yoy | +0.0071 | 0.132 | +0.59 | 0.561 | 1.000 | 0.849 | 1.000 | 55% |
| ret_5 | -0.0076 | 0.149 | -0.56 | 0.579 | 1.000 | 0.849 | 1.000 | 50% |
| random KC code + ridge | +0.0041 | 0.081 | +0.55 | 0.587 | 1.000 | 0.849 | 1.000 | 52% |
| vol_ratio | -0.0048 | 0.100 | -0.53 | 0.592 | 1.000 | 0.849 | 1.000 | 48% |
| eps_growth_yoy | -0.0040 | 0.091 | -0.48 | 0.631 | 1.000 | 0.868 | 1.000 | 51% |
| roe | -0.0036 | 0.105 | -0.38 | 0.709 | 1.000 | 0.894 | 1.000 | 48% |
| fcf_yield | -0.0033 | 0.114 | -0.31 | 0.754 | 1.000 | 0.894 | 1.000 | 48% |
| roa | -0.0040 | 0.140 | -0.31 | 0.757 | 1.000 | 0.894 | 1.000 | 53% |
| ret_1 | -0.0032 | 0.149 | -0.23 | 0.819 | 1.000 | 0.894 | 1.000 | 48% |
| reversal (-ret_1) | +0.0032 | 0.149 | +0.23 | 0.819 | 1.000 | 0.894 | 1.000 | 52% |
| gross_margin | +0.0019 | 0.091 | +0.23 | 0.821 | 1.000 | 0.894 | 1.000 | 43% |
| growth (rev+eps yoy) | +0.0023 | 0.123 | +0.20 | 0.840 | 1.000 | 0.894 | 1.000 | 55% |
| volume_ratio | +0.0009 | 0.071 | +0.13 | 0.893 | 1.000 | 0.908 | 1.000 | 52% |
| quality (ROE+ROA+GM) | -0.0014 | 0.135 | -0.12 | 0.908 | 1.000 | 0.908 | 1.000 | 53% |

Read the corrected columns (Holm, maxT), not raw p: with this many signals a few will look good by chance.

## Power: what could this data have detected? (real month-to-month IC dispersion)

| planted IC | detection rate |
|---|---|
| 0.00 | 5% |
| 0.01 | 15% |
| 0.02 | 45% |
| 0.03 | 70% |
| 0.04 | 87% |
| 0.05 | 96% |
| 0.07 | 100% |
| 0.10 | 100% |

False-positive rate at IC 0 (placebo): 5% (should be near 5%). Smallest planted IC found in >= 80% of trials: 0.04.

## Decile long-short (net of costs, monthly, equal weight)

| signal | gross/m | net Sharpe @0bp | @10bp | @25bp | maxDD @10bp | turnover |
|---|---|---|---|---|---|---|
| random dense + ridge | +0.42% | +0.57 | +0.36 | +0.05 | -15.8% | 76% |
| accruals | -0.27% | -0.51 | -0.55 | -0.62 | -42.8% | 12% |
| vol_20 | +0.69% | +0.52 | +0.42 | +0.27 | -25.7% | 66% |
| dist_52w_high | -1.08% | -0.70 | -0.76 | -0.85 | -144.4% | 49% |
| ridge raw | +0.36% | +0.38 | +0.26 | +0.09 | -18.9% | 54% |
| debt_to_equity | +0.24% | +0.43 | +0.41 | +0.38 | -12.7% | 5% |
| low leverage (-D/E) | -0.24% | -0.43 | -0.45 | -0.48 | -35.9% | 5% |
| earnings_yield | -0.23% | -0.28 | -0.31 | -0.36 | -55.5% | 12% |
| rsi_14 | -0.54% | -0.50 | -0.66 | -0.90 | -91.4% | 88% |
| ma200_dist | -0.76% | -0.52 | -0.57 | -0.66 | -106.0% | 39% |

Caveats: the panel is not fully survivorship-free (delisted names are missing), costs are a flat bps per unit traded with no market impact or borrow cost on shorts, and the top-10 list above was chosen AFTER seeing the results, so its Sharpe ratios are optimistic.