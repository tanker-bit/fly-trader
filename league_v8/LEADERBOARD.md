# Round 18   train through 2023-01-12   validate 2023-04-14 -> 2024-04-15
agents scored 20, agents tried so far 200, validation noise (se of mean IC) about +/-0.054

rank id     gen parent   IC now 3-rd avg rounds     lr0  tau_an kc_sp drive   view  fate
   1 a0192    0 -        +0.046   +0.046      1   0.005    5402 0.066  1.72    r+f  KEPT
   2 a0184    4 a0155    +0.035   +0.038      2   0.006   22000 0.073  2.17  p+r+f  KEPT
   3 a0187    4 a0155    +0.031   +0.036      2   0.007   22394 0.073  1.92  p+r+f  KEPT
   4 a0195    2 a0189    +0.035   +0.035      1   0.004    5142 0.063  1.71    r+f  KEPT
   5 a0185    1 a0160    +0.022   +0.035      2   0.024    2000 0.032  1.08    p+r  KEPT
   6 a0196    5 a0184    +0.033   +0.033      1   0.008   28732 0.068  1.99  p+r+f  KEPT
   7 a0180    0 -        +0.019   +0.027      2   0.038   48375 0.084  1.04    p+f  KEPT
   8 a0191    0 -        +0.026   +0.026      1   0.021   67594 0.034  1.38      f  KEPT
   9 a0194    5 a0184    +0.026   +0.026      1   0.006   13437 0.077  2.17  p+r+f  culled
  10 a0193    0 -        +0.025   +0.025      1   0.005   14219 0.079  1.66    p+f  KEPT
  11 a0182    0 -        +0.003   +0.020      2   0.020   86281 0.056  2.63    p+r  KEPT
  12 a0199    1 a0182    +0.020   +0.020      1   0.020   86281 0.062  2.63    p+r  culled
  13 a0197    5 a0184    +0.019   +0.019      1   0.006   25644 0.084  2.17    p+f  culled
  14 a0160    0 -        +0.021   +0.018      4   0.021    2085 0.032  1.16      p  culled
  15 a0189    1 a0103    -0.003   +0.016      2   0.007    5760 0.066  1.62      f  culled
  16 a0181    0 -        -0.006   +0.013      2   0.066   27325 0.056  2.80      p  culled
  17 a0162    0 -        +0.028   +0.013      4   0.025    9841 0.055  1.35    p+r  culled
  18 a0153    0 -        +0.026   +0.013      5   0.010   17673 0.091  2.76  p+r+f  culled
  19 a0198    2 a0189    +0.003   +0.003      1   0.007    5760 0.066  1.45      f  culled
  20 a0190    0 -        -0.001   -0.001      1   0.005   66610 0.033  1.94    r+f  culled

Controls on the same validation window (no learning from this window):
  ridge                IC +0.055
  momentum 12-1        IC +0.014
  reversal (-ret_1)    IC +0.008
  random               IC +0.002
  fly untrained (default) IC -0.002
  Committee (kept last round) IC +0.023   (its 10 members average +0.018 individually)
  Committee (all scored) IC +0.029   (its 20 members average +0.020 individually)

Luck bar for the best of 200 agents: 0.174.  Best IC this round: +0.046  -> NOT above the bar: this round's ranking is mostly noise.