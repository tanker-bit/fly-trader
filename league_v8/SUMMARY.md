# Cross-round summary (IC = mean rank correlation with the 21-day excess return)

 rd   n     all    surv    kids    best dead    comm   ridge     mom     rev    rand    untr  persist
  0  20  +0.035    +nan  +0.035  +0.073    4    +nan  +0.068  +0.061  +0.021  -0.005  +0.015     +nan
  1  20  +0.003  -0.002  +0.008  +0.037    0  -0.004  +0.002  -0.054  +0.031  -0.001  +0.001    -0.50
  2  20  -0.019  -0.019  -0.020  +0.008    2  -0.024  -0.017  +0.101  +0.028  +0.004  +0.041    -0.61
  3  20  -0.025  -0.027  -0.022  +0.039    0  -0.062  -0.045  -0.063  +0.026  +0.011  -0.032    -0.53
  4  20  +0.006  +0.009  +0.003  +0.079    0  +0.018  +0.041  -0.083  +0.007  -0.005  +0.000    +0.39
  5  20  +0.029  +0.030  +0.027  +0.041    1  +0.039  +0.030  -0.010  +0.004  +0.001  +0.024    +0.66
  6  20  -0.019  -0.020  -0.018  +0.003    0  -0.021  -0.031  +0.044  -0.013  +0.001  +0.000    -0.16
  7  20  +0.009  +0.014  +0.003  +0.034    0  +0.015  -0.037  +0.008  -0.001  +0.004  -0.004    +0.10
  8  20  +0.033  +0.039  +0.027  +0.058    1  +0.045  +0.055  +0.014  +0.039  -0.010  +0.023    +0.33
  9  20  -0.001  -0.005  +0.003  +0.035    0  -0.007  +0.007  -0.011  -0.009  -0.002  -0.006    +0.32
 10  20  +0.007  +0.006  +0.008  +0.056    0  +0.020  -0.026  +0.061  -0.028  +0.004  -0.026    +0.31
 11  20  -0.029  -0.037  -0.021  +0.036    0  -0.044  +0.015  -0.075  -0.001  -0.009  -0.013    +0.07
 12  20  -0.008  -0.014  -0.002  +0.042    0  -0.017  +0.024  +0.055  +0.009  +0.005  -0.001    +0.65
 13  20  +0.010  +0.017  +0.004  +0.043    0  +0.025  -0.002  -0.026  +0.015  +0.006  -0.011    +0.08
 14  20  +0.041  +0.042  +0.040  +0.084    0  +0.070  +0.025  +0.037  +0.006  +0.009  +0.022    +0.94
 15  20  -0.002  -0.004  +0.000  +0.071    0  -0.006  +0.071  -0.125  +0.025  -0.007  -0.022    -0.68
 16  20  -0.024  -0.026  -0.021  -0.005    0  -0.038  -0.070  -0.026  +0.005  +0.007  +0.008    +0.19
 17  20  +0.022  +0.024  +0.019  +0.047    0  +0.036  +0.034  +0.001  +0.001  -0.007  +0.028    +0.20
 18  20  +0.020  +0.018  +0.023  +0.046    0  +0.023  +0.055  +0.014  +0.008  +0.002  -0.002    +0.47

all = mean IC of every agent scored; surv = agents kept before the round; kids = new that round; dead = IC exactly 0 (silent);
comm = committee of last round's survivors (needs scores saved by v6 code; blank for older runs); persist = rank corr of agents' IC with the previous round.

Levels (mean over rounds):
fly, all agents                        mean +0.0046  se 0.0050  t +0.93  rounds>0 11/19
ridge                                  mean +0.0104  se 0.0092  t +1.12  rounds>0 12/19
momentum 12-1                          mean -0.0041  se 0.0134  t -0.31  rounds>0 10/19
reversal                               mean +0.0090  se 0.0038  t +2.35  rounds>0 14/19
random                                 mean +0.0004  se 0.0014  t +0.27  rounds>0 11/19
fly untrained                          mean +0.0024  se 0.0045  t +0.55  rounds>0 10/19
committee of survivors                 mean +0.0038  se 0.0081  t +0.47  rounds>0 9/18

Differences (what matters):
fly - ridge                            mean -0.0058  se 0.0073  t -0.79  rounds>0 8/19
fly - untrained                        mean +0.0022  se 0.0051  t +0.42  rounds>0 13/19
fly - random                           mean +0.0042  se 0.0054  t +0.78  rounds>0 13/19
fly - momentum                         mean +0.0087  se 0.0131  t +0.67  rounds>0 14/19
committee - mean individual fly        mean +0.0013  se 0.0031  t +0.42  rounds>0 9/18
committee - ridge                      mean -0.0034  se 0.0085  t -0.40  rounds>0 8/18

Does the culling find skill?
survivors - children (selection value) mean -0.0009  se 0.0020  t -0.43  rounds>0 8/18
persistence (rank corr, round to round) mean +0.1232  se 0.1092  t +1.13  rounds>0 13/18

Reading guide: |t| < 2 means no evidence. Persistence near 0 or negative and selection value near 0 mean the
ranking inside a round is noise, so evolving among similar flies adds nothing. 'dead' agents scoring 0 should be 0 after v6.