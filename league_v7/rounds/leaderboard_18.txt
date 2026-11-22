# Round 18   train through 2023-01-12   validate 2023-04-14 -> 2024-04-15
agents scored 20, agents tried so far 200, validation noise (se of mean IC) about +/-0.052

rank id     gen parent   IC now 3-rd avg rounds     lr0  tau_an kc_sp drive  fate
   1 a0187    7 a0150    +0.033   +0.040      2   0.008    5323 0.060  0.90  KEPT
   2 a0189    8 a0163    +0.031   +0.040      2   0.010    4782 0.018  1.11  KEPT
   3 a0188    8 a0163    +0.031   +0.040      2   0.006   11045 0.015  1.09  KEPT
   4 a0181    7 a0159    +0.031   +0.039      2   0.004    5782 0.063  1.58  KEPT
   5 a0180    7 a0155    +0.031   +0.039      2   0.009    3674 0.060  1.20  KEPT
   6 a0184    8 a0165    +0.031   +0.038      2   0.010   14696 0.040  1.44  KEPT
   7 a0195    9 a0189    +0.032   +0.032      1   0.008    4782 0.015  1.11  KEPT
   8 a0191    8 a0187    +0.032   +0.032      1   0.008    7147 0.060  1.03  KEPT
   9 a0192    8 a0181    +0.031   +0.031      1   0.004    5782 0.057  1.00  KEPT
  10 a0182    7 a0152    +0.016   +0.031      2   0.009   10314 0.051  1.11  KEPT
  11 a0193    8 a0187    +0.031   +0.031      1   0.011    6951 0.056  0.90  culled
  12 a0186    7 a0152    +0.016   +0.031      2   0.011   10314 0.058  1.32  culled
  13 a0199    9 a0189    +0.029   +0.029      1   0.010    4782 0.013  0.99  culled
  14 a0198    8 a0181    +0.029   +0.029      1   0.004    5782 0.056  1.58  culled
  15 a0183    8 a0161    +0.030   +0.029      2   0.010   17311 0.040  0.90  culled
  16 a0190    8 a0181    +0.026   +0.026      1   0.005    5782 0.063  1.73  culled
  17 a0185    7 a0159    +0.010   +0.023      2   0.004    7500 0.063  1.40  culled
  18 a0194    8 a0181    +0.019   +0.019      1   0.004    5782 0.060  1.85  culled
  19 a0197    8 a0186    +0.017   +0.017      1   0.013   10314 0.063  1.49  culled
  20 a0196    8 a0182    +0.016   +0.016      1   0.009   18287 0.051  1.25  culled

Controls on the same validation window (no learning from this window):
  ridge                IC +0.055
  momentum 12-1        IC +0.014
  reversal (-ret_1)    IC +0.008
  random               IC +0.002
  fly untrained (default) IC -0.002
  Committee (kept last round) IC +0.029   (its 10 members average +0.026 individually)
  Committee (all scored) IC +0.028   (its 20 members average +0.026 individually)

Luck bar for the best of 200 agents: 0.169.  Best IC this round: +0.033  -> NOT above the bar: this round's ranking is mostly noise.