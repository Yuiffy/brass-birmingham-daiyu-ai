(function(root){
    'use strict';
    const names={human:'人类玩家',heuristic:'基础 AI',search:'搜索 AI',neural:'神经网络 AI',world:'世界模型 AI',guided:'学习增强 AI'};
    // Saved simulator benchmarks, not predictions of the user's next game.
    // Keep rule sets, player counts, profiles and matchup sources separate.
    // Economy mixed tables: world_model/experiments/economy-20260920/{2,3,4}p/runtime-tournament.json.
    // Four-player baseline/guide self-play: docs/human-strategy-research.md.
    // Teacher self-play/world mixed: world_model/experiments/public-human-20261010/summary.json.
    // Legacy tables: world_model/reports/tournament.json, tournament-4p-default-120-new.json,
    // and tournament-{2,3}p-specialized-active-120-new.json.
    const economy={
        2:{guided:83.875,world:27.216666666666665},
        3:{guided:71.98333333333333,world:24.566666666666666},
        4:{guided:78.28,world:27.833333333333332}
    };
    const legacy={
        2:{guided:140.78333333333333,world:92.86666666666666},
        3:{guided:139.81666666666666,world:108.4},
        4:{heuristic:87.1275,search:104.4725,neural:91.8625,guided:132.25833333333333,world:119.81666666666666}
    };
    function score(type,{players=4,profile='',rules='legacy-v1'}={}) {
        if(rules!=='economy-v2')return legacy[players]?.[type];
        if(type==='guided'&&profile==='human-guide-v1')return players===4?124.38:undefined;
        if(['guided','world'].includes(type)&&profile==='teacher-trained-v2')
            return players===4?(type==='guided'?127.76666666666667:50.3):undefined;
        return economy[players]?.[type];
    }
    function suffix(value){return Number.isFinite(value)?`（平均约${Math.round(value)}分）`:'（均分待测）';}
    function optionLabel(type,context) {
        if(type==='human')return names.human;
        if(context.rules==='economy-v2'&&context.profile==='teacher-trained-v2'&&context.players!==4&&['guided','world'].includes(type))
            return `${names[type]}（仅支持四人）`;
        return names[type]+suffix(score(type,context));
    }
    function profileLabel(profile,context) {
        const name={'':'标准版','human-guide-v1':'攻略版','teacher-trained-v2':'教师＋保牌版'}[profile];
        if(context.rules!=='economy-v2')return name+'（需切换经济规则）';
        if(profile==='teacher-trained-v2'&&context.players!==4)return name+'（仅支持四人）';
        return name+suffix(score('guided',{...context,profile}));
    }
    function evidence(context) {
        if(context.rules==='economy-v2') {
            if(context.profile==='teacher-trained-v2')return {
                text:'四人教师＋保牌：学习增强 AI 为 120 局同策略自对弈均分 127.77；世界模型 AI 为 40 局混合桌均分 50.30。两项对手不同，不能作为同桌排名。其他 AI 暂无本设置下的均分。',
                url:'docs/public-human-training-results.md'
            };
            if(context.profile==='human-guide-v1')return {
                text:'攻略版：四人学习增强 AI 为 120 局同策略自对弈均分 124.38；二、三人未测。世界模型仍使用标准版权重，其均分取自此前混合桌测试。其他 AI 暂无本设置下的均分。',
                url:'docs/human-strategy-research.md'
            };
            return {
                text:'标准版：四人学习增强 AI 为 120 局同策略自对弈均分 78.28；二、三人及世界模型均分取自各人数的 120 局混合桌历史测试。对手不同，不能作为同桌排名。其他 AI 暂无本设置下的均分。',
                url:context.players===4?'docs/human-strategy-research.md':'docs/economy-training-results.md'
            };
        }
        return {
            text:'历史规则含额外终局收入分。学习增强 / 世界模型取自各人数的 120 局混合桌；基础 / 搜索 / 神经网络取自四人 400 局混合桌，二、三人未测。不同测试的对手不同，均分不能作为同桌排名，也不能与经济规则直接比较。',
            url:'docs/higher-scores.md'
        };
    }
    root.BrassAILabels=Object.freeze({optionLabel,profileLabel,evidence});
})(globalThis);
