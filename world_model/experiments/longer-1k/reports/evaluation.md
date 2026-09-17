# World Model Evaluation

Dataset: 1000 complete games; held out: 100 games / 12800 transitions.

Current player accuracy: 21.62%
Change precision / recall / F1: 33.54% / 80.15% / 47.29%

| Field group | MAE in native units | Rounded accuracy | MAE on changed fields |
|---|---:|---:|---:|
| cards | 0.085 | 95.97% | 0.49989986419677734 |
| coalMarket | 0.227 | 89.29% | 0.29515567421913147 |
| currentPlayer | 0.455 | 63.69% | 0.32722851634025574 |
| era | 0.002 | 100.00% | 0.06373288482427597 |
| flip | 0.026 | 99.59% | 0.6825206875801086 |
| income | 0.812 | 44.19% | 2.0018248558044434 |
| industry | 0.096 | 97.49% | 0.7692266702651978 |
| ironMarket | 0.197 | 94.14% | 0.27339789271354675 |
| merchant | 0.011 | 99.91% | 0.6691266894340515 |
| money | 6.133 | 5.93% | 7.239922523498535 |
| network | 0.069 | 98.89% | 0.5944586396217346 |
| resources | 0.062 | 98.38% | 0.43590280413627625 |
| supply | 0.123 | 98.41% | 0.45456811785697937 |
| turn | 1.311 | 65.12% | 2.0190513134002686 |
| vp | 0.350 | 94.09% | 8.581741333007812 |

| Rollout depth | Normalized MAE | Persistence MAE |
|---|---:|---:|
| 1 | 0.01962 | 0.00459 |
| 2 | 0.02944 | 0.00962 |
| 3 | 0.03893 | 0.01368 |
| 4 | 0.05166 | 0.01767 |
| 5 | 0.06105 | 0.02137 |

See evaluation.json for action breakdowns, sample field differences and limitations.