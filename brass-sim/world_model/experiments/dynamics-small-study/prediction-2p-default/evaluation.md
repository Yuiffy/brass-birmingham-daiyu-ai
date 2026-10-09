# World Model Evaluation

Dataset: 400 complete games; held out: 40 games / 3200 transitions.

Current player accuracy: 75.47%
Change precision / recall / F1: 51.41% / 75.63% / 61.21%

| Field group | MAE in native units | Rounded accuracy | MAE on changed fields |
|---|---:|---:|---:|
| cards | 0.057 | 96.38% | 0.9644736647605896 |
| coalMarket | 0.281 | 68.84% | 0.0996779203414917 |
| currentPlayer | 0.123 | 87.73% | 0.13833075761795044 |
| era | 0.003 | 99.72% | 0.10000000149011612 |
| flip | 0.008 | 99.15% | 0.5584880709648132 |
| income | 0.229 | 94.19% | 2.825878381729126 |
| industry | 0.009 | 99.46% | 0.7254375219345093 |
| ironMarket | 0.066 | 94.50% | 0.44637811183929443 |
| merchant | 0.005 | 99.50% | 0.7009050846099854 |
| money | 3.565 | 68.54% | 10.732303619384766 |
| network | 0.014 | 99.09% | 0.6782901287078857 |
| resources | 0.024 | 97.97% | 0.39375779032707214 |
| supply | 0.017 | 98.91% | 0.15316972136497498 |
| turn | 1.121 | 66.41% | 3.3312129974365234 |
| vp | 0.647 | 97.64% | 35.79483413696289 |

| Rollout depth | Normalized MAE | Persistence MAE |
|---|---:|---:|
| 1 | 0.00642 | 0.00460 |
| 2 | 0.01265 | 0.00938 |
| 3 | 0.01542 | 0.01285 |
| 4 | 0.01797 | 0.01504 |
| 5 | 0.02023 | 0.01882 |

See evaluation.json for action breakdowns, sample field differences and limitations.