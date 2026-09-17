# World Model Evaluation

Dataset: 400 complete games; held out: 40 games / 4320 transitions.

Current player accuracy: 93.59%
Change precision / recall / F1: 86.08% / 85.16% / 85.61%

| Field group | MAE in native units | Rounded accuracy | MAE on changed fields |
|---|---:|---:|---:|
| cards | 0.015 | 99.05% | 0.4279671907424927 |
| coalMarket | 0.020 | 98.45% | 0.08163455128669739 |
| currentPlayer | 0.032 | 96.79% | 0.07057701796293259 |
| era | 0.007 | 99.26% | 0.17499999701976776 |
| flip | 0.007 | 99.37% | 0.5835995078086853 |
| income | 0.142 | 95.62% | 2.8772335052490234 |
| industry | 0.007 | 99.67% | 0.4741823971271515 |
| ironMarket | 0.029 | 97.45% | 0.14004217088222504 |
| merchant | 0.004 | 99.59% | 0.6142325401306152 |
| money | 1.993 | 68.89% | 4.856207847595215 |
| network | 0.007 | 99.67% | 0.3076249063014984 |
| resources | 0.011 | 99.08% | 0.35737624764442444 |
| supply | 0.002 | 99.84% | 0.060446158051490784 |
| turn | 0.371 | 88.22% | 1.4328263998031616 |
| vp | 0.310 | 98.14% | 12.111132621765137 |

| Rollout depth | Normalized MAE | Persistence MAE |
|---|---:|---:|
| 1 | 0.00213 | 0.00788 |
| 2 | 0.00286 | 0.01076 |
| 3 | 0.00560 | 0.01501 |
| 4 | 0.00635 | 0.01767 |
| 5 | 0.00935 | 0.02152 |

See evaluation.json for action breakdowns, sample field differences and limitations.