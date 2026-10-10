(function(root){
    'use strict';
    const names={human:'人类玩家',heuristic:'基础 AI',search:'搜索 AI',neural:'神经网络 AI',world:'世界模型 AI',guided:'学习增强 AI'};
    function measured(type,{players=4,profile='',rules='legacy-v1'}={}) {
        return root.BrassBenchmarkScores?.scores.find(row=>row.rules===rules&&row.players===players&&
            row.profile===(rules==='economy-v2'?profile:'')&&row.type===type);
    }
    function suffix(row){return Number.isFinite(row?.meanVP)?`（平均约${Math.round(row.meanVP)}分）`:'（评分不可用）';}
    function optionLabel(type,context) {
        return type==='human'?names.human:names[type]+suffix(measured(type,context));
    }
    function profileLabel(profile,context) {
        const name={'':'标准版','human-guide-v1':'攻略版','teacher-trained-v2':'教师＋保牌版'}[profile];
        if(context.rules!=='economy-v2')return name+'（需切换经济规则）';
        return name+suffix(measured('guided',{...context,profile}));
    }
    function evidence(context) {
        const rows=Object.keys(names).filter(type=>type!=='human').map(type=>({type,row:measured(type,context)}));
        const games=rows.every(({row})=>row)?rows[0].row.games:null;
        const text=games?`${context.players} 人局，各 AI 均跑 ${games} 个独立种子的同类自对弈；每局各座位平均后统计。`+
            rows.map(({type,row})=>`${names[type]}：${row.meanVP.toFixed(2)} 分（95% 区间 ${row.meanVP95CI.map(v=>v.toFixed(2)).join('～')}）`).join('；')+'。':
            '均分报告未能加载，请刷新页面。';
        return {text,url:'docs/browser-ai-benchmarks.md'};
    }
    root.BrassAILabels=Object.freeze({optionLabel,profileLabel,evidence,measured});
})(globalThis);
