# World Model Evaluation

Dataset: 1000 complete games; held out: 100 games / 12800 transitions.

Current player accuracy: 21.64%
Change precision / recall / F1: 34.35% / 80.49% / 48.15%

| Field group | MAE in native units | Rounded accuracy | MAE on changed fields |
|---|---:|---:|---:|
| cards | 0.083 | 96.24% | 0.48317766189575195 |
| coalMarket | 0.225 | 89.62% | 0.28842809796333313 |
| currentPlayer | 0.453 | 63.86% | 0.32479819655418396 |
| era | 0.002 | 100.00% | 0.04116455093026161 |
| flip | 0.026 | 99.60% | 0.6453747153282166 |
| income | 0.804 | 45.84% | 1.9749008417129517 |
| industry | 0.093 | 97.58% | 0.7400966286659241 |
| ironMarket | 0.196 | 94.12% | 0.2791191041469574 |
| merchant | 0.010 | 99.90% | 0.6221279501914978 |
| money | 6.140 | 5.75% | 7.0513081550598145 |
| network | 0.066 | 98.89% | 0.5952566266059875 |
| resources | 0.061 | 98.43% | 0.4245251417160034 |
| supply | 0.122 | 98.43% | 0.4528934955596924 |
| turn | 1.291 | 65.34% | 1.9937822818756104 |
| vp | 0.285 | 95.02% | 8.24343204498291 |

| Rollout depth | Normalized MAE | Persistence MAE |
|---|---:|---:|
| 1 | 0.01907 | 0.00459 |
| 2 | 0.03024 | 0.00962 |
| 3 | 0.04099 | 0.01368 |
| 4 | 0.05660 | 0.01767 |
| 5 | 0.06836 | 0.02137 |

See evaluation.json for action breakdowns, sample field differences and limitations.