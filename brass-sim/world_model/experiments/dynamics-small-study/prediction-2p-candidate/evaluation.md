# World Model Evaluation

Dataset: 400 complete games; held out: 40 games / 3200 transitions.

Current player accuracy: 94.78%
Change precision / recall / F1: 86.71% / 87.65% / 87.18%

| Field group | MAE in native units | Rounded accuracy | MAE on changed fields |
|---|---:|---:|---:|
| cards | 0.016 | 99.00% | 0.44139114022254944 |
| coalMarket | 0.025 | 97.97% | 0.09759429842233658 |
| currentPlayer | 0.026 | 97.39% | 0.05486862361431122 |
| era | 0.008 | 99.22% | 0.02500000037252903 |
| flip | 0.005 | 99.56% | 0.539833664894104 |
| income | 0.129 | 95.95% | 2.5594823360443115 |
| industry | 0.004 | 99.80% | 0.3835907280445099 |
| ironMarket | 0.023 | 98.31% | 0.08721234649419785 |
| merchant | 0.004 | 99.60% | 0.5269386768341064 |
| money | 1.730 | 71.52% | 5.044417858123779 |
| network | 0.005 | 99.76% | 0.2265653908252716 |
| resources | 0.011 | 99.18% | 0.35955461859703064 |
| supply | 0.002 | 99.84% | 0.060719624161720276 |
| turn | 0.328 | 88.01% | 1.4188871383666992 |
| vp | 0.136 | 98.42% | 8.209590911865234 |

| Rollout depth | Normalized MAE | Persistence MAE |
|---|---:|---:|
| 1 | 0.00078 | 0.00460 |
| 2 | 0.00281 | 0.00938 |
| 3 | 0.00353 | 0.01285 |
| 4 | 0.00514 | 0.01504 |
| 5 | 0.00580 | 0.01882 |

See evaluation.json for action breakdowns, sample field differences and limitations.