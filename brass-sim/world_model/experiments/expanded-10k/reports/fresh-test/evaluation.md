# World Model Evaluation

Dataset: 10000 complete games; held out: 100 games / 12800 transitions.

Current player accuracy: 21.98%
Change precision / recall / F1: 34.36% / 80.49% / 48.16%

| Field group | MAE in native units | Rounded accuracy | MAE on changed fields |
|---|---:|---:|---:|
| cards | 0.083 | 96.26% | 0.4820708632469177 |
| coalMarket | 0.227 | 89.38% | 0.3046690821647644 |
| currentPlayer | 0.453 | 64.13% | 0.32520896196365356 |
| era | 0.002 | 100.00% | 0.03429068624973297 |
| flip | 0.026 | 99.62% | 0.6487417221069336 |
| income | 0.796 | 46.57% | 2.0278308391571045 |
| industry | 0.093 | 97.59% | 0.7299575805664062 |
| ironMarket | 0.194 | 94.43% | 0.28900837898254395 |
| merchant | 0.011 | 99.90% | 0.6426337361335754 |
| money | 6.070 | 5.50% | 6.947600841522217 |
| network | 0.067 | 98.87% | 0.6098134517669678 |
| resources | 0.060 | 98.45% | 0.4242139458656311 |
| supply | 0.121 | 98.48% | 0.4462350010871887 |
| turn | 1.289 | 65.32% | 2.0040323734283447 |
| vp | 0.298 | 94.86% | 8.156556129455566 |

| Rollout depth | Normalized MAE | Persistence MAE |
|---|---:|---:|
| 1 | 0.01838 | 0.00455 |
| 2 | 0.02953 | 0.00936 |
| 3 | 0.04028 | 0.01357 |
| 4 | 0.05725 | 0.01742 |
| 5 | 0.06934 | 0.02111 |

See evaluation.json for action breakdowns, sample field differences and limitations.