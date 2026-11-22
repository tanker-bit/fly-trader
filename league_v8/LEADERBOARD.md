# Round 1   train through 2006-01-05   validate 2006-04-06 -> 2007-04-09
agents scored 20, agents tried so far 30, validation noise (se of mean IC) about +/-0.045

rank id     gen parent   IC now 3-rd avg rounds     lr0  tau_an kc_sp drive   view  fate
   1 a0003    0 -        +0.037   +0.045      2   0.094   29224 0.068  2.19    r+f  KEPT
   2 a0019    0 -        +0.022   +0.038      2   0.054   10157 0.023  1.75  p+r+f  KEPT
   3 a0010    0 -        +0.011   +0.033      2   0.015   38760 0.048  2.20    p+r  KEPT
   4 a0004    0 -        +0.025   +0.029      2   0.019    6732 0.072  1.72    p+r  KEPT
   5 a0001    0 -        -0.006   +0.026      2   0.061    2281 0.059  2.86    r+f  KEPT
   6 a0008    0 -        +0.005   +0.026      2   0.009   27716 0.046  1.32  p+r+f  KEPT
   7 a0013    0 -        -0.028   +0.022      2   0.050   74145 0.090  2.63  p+r+f  KEPT
   8 a0011    0 -        -0.039   +0.017      2   0.079    3135 0.081  1.77    p+f  KEPT
   9 a0000    0 -        -0.023   +0.014      2   0.052   71084 0.067  1.47    p+f  KEPT
  10 a0015    0 -        -0.026   +0.013      2   0.004   53753 0.011  2.49    r+f  KEPT
  11 a0029    1 a0008    +0.005   +0.005      1   0.009   27716 0.046  1.32  p+r+f  culled
  12 a0020    0 -        +0.000   +0.000      1   0.074   21463 0.034  0.99      f  culled
  13 a0021    0 -        +0.000   +0.000      1   0.061    2281 0.059  2.86      f  culled
  14 a0022    0 -        +0.000   +0.000      1   0.003    3252 0.088  2.04      f  culled
  15 a0023    0 -        +0.000   +0.000      1   0.094   29224 0.068  2.19      f  culled
  16 a0027    1 a0001    -0.003   -0.003      1   0.038    2000 0.059  3.00    r+f  culled
  17 a0028    1 a0001    -0.019   -0.019      1   0.061    2281 0.064  2.86    r+f  culled
  18 a0025    1 a0015    -0.022   -0.022      1   0.003   53753 0.017  2.49    r+f  culled
  19 a0024    1 a0019    -0.024   -0.024      1   0.054   13788 0.012  2.07    p+f  culled
  20 a0026    1 a0013    -0.037   -0.037      1   0.087  100000 0.086  2.70  p+r+f  culled

Controls on the same validation window (no learning from this window):
  ridge                IC +0.002
  momentum 12-1        IC -0.054
  reversal (-ret_1)    IC +0.031
  random               IC -0.001
  fly untrained (default) IC +0.001
  Committee (kept last round) IC -0.004   (its 10 members average -0.002 individually)
  Committee (all scored) IC -0.011   (its 20 members average -0.006 individually)

Luck bar for the best of 30 agents: 0.116.  Best IC this round: +0.037  -> NOT above the bar: this round's ranking is mostly noise.