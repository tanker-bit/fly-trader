# Cross-round summary (IC = mean rank correlation with the 21-day excess return)

 rd   n     all    surv    kids    best dead    comm   ridge     mom     rev    rand    untr  persist
  0  20  +0.043    +nan  +0.043  +0.073    0    +nan  +0.068  +0.061  +0.021  -0.005  +0.015     +nan
  1  20  +0.001  -0.005  +0.006  +0.030    0  -0.004  +0.002  -0.054  +0.031  -0.001  +0.001    -0.26
  2  20  -0.019  -0.017  -0.020  +0.048    0  -0.019  -0.017  +0.101  +0.028  +0.004  +0.041    +0.02
  3  20  -0.052  -0.054  -0.051  +0.019    0  -0.072  -0.045  -0.063  +0.026  +0.011  -0.032    -0.08
  4  20  -0.008  -0.013  -0.004  +0.082    0  -0.012  +0.041  -0.083  +0.007  -0.005  +0.000    +0.26
  5  20  +0.035  +0.036  +0.034  +0.041    0  +0.037  +0.030  -0.010  +0.004  +0.001  +0.024    -0.13
  6  20  -0.018  -0.019  -0.017  -0.005    0  -0.019  -0.031  +0.044  -0.013  +0.001  +0.000    -0.13
  7  20  +0.006  +0.005  +0.007  +0.011    0  +0.006  -0.037  +0.008  -0.001  +0.004  -0.004    +0.21
  8  20  +0.042  +0.040  +0.043  +0.048    0  +0.042  +0.055  +0.014  +0.039  -0.010  +0.023    +0.75
  9  20  -0.009  -0.009  -0.009  -0.004    0  -0.010  +0.007  -0.011  -0.009  -0.002  -0.006    +0.64
 10  20  -0.038  -0.038  -0.038  -0.021    0  -0.039  -0.026  +0.061  -0.028  +0.004  -0.026    +0.18
 11  20  +0.035  +0.035  +0.034  +0.041    0  +0.037  +0.015  -0.075  -0.001  -0.009  -0.013    +0.13
 12  20  -0.011  -0.012  -0.011  +0.014    0  -0.013  +0.024  +0.055  +0.009  +0.005  -0.001    -0.52
 13  20  -0.002  -0.002  -0.003  +0.004    0  -0.002  -0.002  -0.026  +0.015  +0.006  -0.011    +0.64
 14  20  +0.032  +0.028  +0.035  +0.054    0  +0.030  +0.025  +0.037  +0.006  +0.009  +0.022    +0.54
 15  20  +0.026  +0.029  +0.023  +0.049    0  +0.030  +0.071  -0.125  +0.025  -0.007  -0.022    -0.93
 16  20  -0.028  -0.029  -0.027  -0.020    0  -0.031  -0.070  -0.026  +0.005  +0.007  +0.008    -0.60
 17  20  +0.044  +0.045  +0.043  +0.048    0  +0.047  +0.034  +0.001  +0.001  -0.007  +0.028    -0.62
 18  20  +0.026  +0.026  +0.026  +0.033    0  +0.029  +0.055  +0.014  +0.008  +0.002  -0.002    +0.56

all = mean IC of every agent scored; surv = agents kept before the round; kids = new that round; dead = IC exactly 0 (silent);
comm = committee of last round's survivors (needs scores saved by v6 code; blank for older runs); persist = rank corr of agents' IC with the previous round.

Levels (mean over rounds):
fly, all agents                        mean +0.0055  se 0.0068  t +0.81  rounds>0 10/19
ridge                                  mean +0.0104  se 0.0092  t +1.12  rounds>0 12/19
momentum 12-1                          mean -0.0041  se 0.0134  t -0.31  rounds>0 10/19
reversal                               mean +0.0090  se 0.0038  t +2.35  rounds>0 14/19
random                                 mean +0.0004  se 0.0014  t +0.27  rounds>0 11/19
fly untrained                          mean +0.0024  se 0.0045  t +0.55  rounds>0 10/19
committee of survivors                 mean +0.0020  se 0.0077  t +0.26  rounds>0 8/18

Differences (what matters):
fly - ridge                            mean -0.0049  se 0.0059  t -0.84  rounds>0 7/19
fly - untrained                        mean +0.0030  se 0.0062  t +0.49  rounds>0 10/19
fly - random                           mean +0.0051  se 0.0077  t +0.66  rounds>0 10/19
fly - momentum                         mean +0.0096  se 0.0153  t +0.63  rounds>0 11/19
committee - mean individual fly        mean -0.0007  se 0.0011  t -0.61  rounds>0 11/18
committee - ridge                      mean -0.0051  se 0.0062  t -0.83  rounds>0 7/18

Does the culling find skill?
survivors - children (selection value) mean -0.0014  se 0.0010  t -1.38  rounds>0 6/18
persistence (rank corr, round to round) mean +0.0364  se 0.1156  t +0.31  rounds>0 10/18

Reading guide: |t| < 2 means no evidence. Persistence near 0 or negative and selection value near 0 mean the
ranking inside a round is noise, so evolving among similar flies adds nothing. 'dead' agents scoring 0 should be 0 after v6.