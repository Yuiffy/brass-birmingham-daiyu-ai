# 冻结估值公式

所有公式仅改变竞拍，仍使用相同的购股 / 部署启发式。原公式等价对照不视为一个独立的新方法。

| ID | 名称 | 参数 |
| --- | --- | --- |
| formula-b10-o100-c0 | 线性 b10/股1/控0 | `{"kind":"linear","base":10,"optionWeight":1,"controlWeight":0,"horizon":0.8,"buffer":6,"name":"线性 b10/股1/控0"}` |
| formula-b10-o100-c25 | 线性 b10/股1/控0.25 | `{"kind":"linear","base":10,"optionWeight":1,"controlWeight":0.25,"horizon":0.8,"buffer":6,"name":"线性 b10/股1/控0.25"}` |
| formula-b10-o100-c75 | 线性 b10/股1/控0.75 | `{"kind":"linear","base":10,"optionWeight":1,"controlWeight":0.75,"horizon":0.8,"buffer":6,"name":"线性 b10/股1/控0.75"}` |
| formula-b10-o35-c0 | 线性 b10/股0.35/控0 | `{"kind":"linear","base":10,"optionWeight":0.35,"controlWeight":0,"horizon":0.8,"buffer":6,"name":"线性 b10/股0.35/控0"}` |
| formula-b10-o35-c25 | 线性 b10/股0.35/控0.25 | `{"kind":"linear","base":10,"optionWeight":0.35,"controlWeight":0.25,"horizon":0.8,"buffer":6,"name":"线性 b10/股0.35/控0.25"}` |
| formula-b10-o35-c75 | 线性 b10/股0.35/控0.75 | `{"kind":"linear","base":10,"optionWeight":0.35,"controlWeight":0.75,"horizon":0.8,"buffer":6,"name":"线性 b10/股0.35/控0.75"}` |
| formula-b10-o70-c0 | 线性 b10/股0.7/控0 | `{"kind":"linear","base":10,"optionWeight":0.7,"controlWeight":0,"horizon":0.8,"buffer":6,"name":"线性 b10/股0.7/控0"}` |
| formula-b10-o70-c25 | 线性 b10/股0.7/控0.25 | `{"kind":"linear","base":10,"optionWeight":0.7,"controlWeight":0.25,"horizon":0.8,"buffer":6,"name":"线性 b10/股0.7/控0.25"}` |
| formula-b10-o70-c75 | 线性 b10/股0.7/控0.75 | `{"kind":"linear","base":10,"optionWeight":0.7,"controlWeight":0.75,"horizon":0.8,"buffer":6,"name":"线性 b10/股0.7/控0.75"}` |
| formula-b4-o100-c0 | 线性 b4/股1/控0 | `{"kind":"linear","base":4,"optionWeight":1,"controlWeight":0,"horizon":0.8,"buffer":6,"name":"线性 b4/股1/控0"}` |
| formula-b4-o100-c25 | 线性 b4/股1/控0.25 | `{"kind":"linear","base":4,"optionWeight":1,"controlWeight":0.25,"horizon":0.8,"buffer":6,"name":"线性 b4/股1/控0.25"}` |
| formula-b4-o100-c75 | 线性 b4/股1/控0.75 | `{"kind":"linear","base":4,"optionWeight":1,"controlWeight":0.75,"horizon":0.8,"buffer":6,"name":"线性 b4/股1/控0.75"}` |
| formula-b4-o35-c0 | 线性 b4/股0.35/控0 | `{"kind":"linear","base":4,"optionWeight":0.35,"controlWeight":0,"horizon":0.8,"buffer":6,"name":"线性 b4/股0.35/控0"}` |
| formula-b4-o35-c25 | 线性 b4/股0.35/控0.25 | `{"kind":"linear","base":4,"optionWeight":0.35,"controlWeight":0.25,"horizon":0.8,"buffer":6,"name":"线性 b4/股0.35/控0.25"}` |
| formula-b4-o35-c75 | 线性 b4/股0.35/控0.75 | `{"kind":"linear","base":4,"optionWeight":0.35,"controlWeight":0.75,"horizon":0.8,"buffer":6,"name":"线性 b4/股0.35/控0.75"}` |
| formula-b4-o70-c0 | 线性 b4/股0.7/控0 | `{"kind":"linear","base":4,"optionWeight":0.7,"controlWeight":0,"horizon":0.8,"buffer":6,"name":"线性 b4/股0.7/控0"}` |
| formula-b4-o70-c25 | 线性 b4/股0.7/控0.25 | `{"kind":"linear","base":4,"optionWeight":0.7,"controlWeight":0.25,"horizon":0.8,"buffer":6,"name":"线性 b4/股0.7/控0.25"}` |
| formula-b4-o70-c75 | 线性 b4/股0.7/控0.75 | `{"kind":"linear","base":4,"optionWeight":0.7,"controlWeight":0.75,"horizon":0.8,"buffer":6,"name":"线性 b4/股0.7/控0.75"}` |
| formula-b7-o100-c0 | 线性 b7/股1/控0 | `{"kind":"linear","base":7,"optionWeight":1,"controlWeight":0,"horizon":0.8,"buffer":6,"name":"线性 b7/股1/控0"}` |
| formula-b7-o100-c25 | 线性 b7/股1/控0.25 | `{"kind":"linear","base":7,"optionWeight":1,"controlWeight":0.25,"horizon":0.8,"buffer":6,"name":"线性 b7/股1/控0.25"}` |
| formula-b7-o100-c75 | 线性 b7/股1/控0.75 | `{"kind":"linear","base":7,"optionWeight":1,"controlWeight":0.75,"horizon":0.8,"buffer":6,"name":"线性 b7/股1/控0.75"}` |
| formula-b7-o35-c0 | 线性 b7/股0.35/控0 | `{"kind":"linear","base":7,"optionWeight":0.35,"controlWeight":0,"horizon":0.8,"buffer":6,"name":"线性 b7/股0.35/控0"}` |
| formula-b7-o35-c25 | 线性 b7/股0.35/控0.25 | `{"kind":"linear","base":7,"optionWeight":0.35,"controlWeight":0.25,"horizon":0.8,"buffer":6,"name":"线性 b7/股0.35/控0.25"}` |
| formula-b7-o35-c75 | 线性 b7/股0.35/控0.75 | `{"kind":"linear","base":7,"optionWeight":0.35,"controlWeight":0.75,"horizon":0.8,"buffer":6,"name":"线性 b7/股0.35/控0.75"}` |
| formula-b7-o70-c0 | 线性 b7/股0.7/控0 | `{"kind":"linear","base":7,"optionWeight":0.7,"controlWeight":0,"horizon":0.8,"buffer":6,"name":"线性 b7/股0.7/控0"}` |
| formula-b7-o70-c25 | 线性 b7/股0.7/控0.25 | `{"kind":"linear","base":7,"optionWeight":0.7,"controlWeight":0.25,"horizon":0.8,"buffer":6,"name":"线性 b7/股0.7/控0.25"}` |
| formula-b7-o70-c75 | 线性 b7/股0.7/控0.75 | `{"kind":"linear","base":7,"optionWeight":0.7,"controlWeight":0.75,"horizon":0.8,"buffer":6,"name":"线性 b7/股0.7/控0.75"}` |
| formula-buffer0 | 线性参考/预留0 | `{"kind":"linear","base":7,"optionWeight":0.7,"controlWeight":0.25,"horizon":0.8,"buffer":0,"name":"线性参考/预留0"}` |
| formula-buffer12 | 线性参考/预留12 | `{"kind":"linear","base":7,"optionWeight":0.7,"controlWeight":0.25,"horizon":0.8,"buffer":12,"name":"线性参考/预留12"}` |
| formula-fixed10 | 固定上限10 | `{"kind":"fixed","cap":10,"buffer":6,"name":"固定上限10"}` |
| formula-fixed14 | 固定上限14 | `{"kind":"fixed","cap":14,"buffer":6,"name":"固定上限14"}` |
| formula-fixed18 | 固定上限18 | `{"kind":"fixed","cap":18,"buffer":6,"name":"固定上限18"}` |
| formula-fixed22 | 固定上限22 | `{"kind":"fixed","cap":22,"buffer":6,"name":"固定上限22"}` |
| formula-fixed26 | 固定上限26 | `{"kind":"fixed","cap":26,"buffer":6,"name":"固定上限26"}` |
| formula-fixed30 | 固定上限30 | `{"kind":"fixed","cap":30,"buffer":6,"name":"固定上限30"}` |
| formula-h120 | 剩余涨幅系数1.2 | `{"kind":"linear","base":7,"optionWeight":0.7,"controlWeight":0.25,"horizon":1.2,"buffer":6,"name":"剩余涨幅系数1.2"}` |
| formula-h40 | 剩余涨幅系数0.4 | `{"kind":"linear","base":7,"optionWeight":0.7,"controlWeight":0.25,"horizon":0.4,"buffer":6,"name":"剩余涨幅系数0.4"}` |
| formula-legacy | 原公式+3（等价对照） | `{"kind":"legacy","name":"原公式+3（等价对照）"}` |
| formula-mcfull-relative | 终局模拟/领先 | `{"kind":"mc","horizon":0,"objective":"relative","samples":8,"buffer":6,"quote":"two paired evaluations; local unit-cost correction","name":"终局模拟/领先"}` |
| formula-mcfull-wealth | 终局模拟/财富 | `{"kind":"mc","horizon":0,"objective":"wealth","samples":8,"buffer":6,"quote":"two paired evaluations; local unit-cost correction","name":"终局模拟/财富"}` |
| formula-mcround-relative | 单航次模拟/领先 | `{"kind":"mc","horizon":1,"objective":"relative","samples":8,"buffer":6,"quote":"two paired evaluations; local unit-cost correction","name":"单航次模拟/领先"}` |
| formula-mcround-wealth | 单航次模拟/财富 | `{"kind":"mc","horizon":1,"objective":"wealth","samples":8,"buffer":6,"quote":"two paired evaluations; local unit-cost correction","name":"单航次模拟/财富"}` |
| formula-relative10 | 公开持股优势 b10 | `{"kind":"relative","base":10,"optionWeight":0.7,"controlWeight":0.75,"horizon":0.8,"buffer":6,"name":"公开持股优势 b10"}` |
| formula-relative4 | 公开持股优势 b4 | `{"kind":"relative","base":4,"optionWeight":0.7,"controlWeight":0.75,"horizon":0.8,"buffer":6,"name":"公开持股优势 b4"}` |
| formula-relative7 | 公开持股优势 b7 | `{"kind":"relative","base":7,"optionWeight":0.7,"controlWeight":0.75,"horizon":0.8,"buffer":6,"name":"公开持股优势 b7"}` |
