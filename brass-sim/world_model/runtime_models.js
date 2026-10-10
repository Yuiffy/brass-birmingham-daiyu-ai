(function(root){
    'use strict';
    // Rule versions and player counts never silently fall back to incompatible weights.
    const corrected=(typeof process!=='undefined'?process.env.BRASS_RULES:root.BRASS_RULES)==='economy-v2';
    const directories=Object.freeze(corrected?{
        "2": "experiments/economy-20260920/2p/direct",
        "3": "experiments/economy-20260920/3p/direct",
        "4": "experiments/economy-20260920/4p/direct"
}:{
        2:'experiments/dynamics-2p-active/models',
        3:'experiments/dynamics-3p-active/models',
        4:'models'
    });
    const guidedDirectories=Object.freeze(corrected?{
        "2": "experiments/economy-20260920/2p/direct",
        "3": "experiments/economy-20260920/3p/direct",
        "4": "experiments/economy-20260920/4p/direct"
}:{...directories});
    function directoryForPlayers(players){
        if(!Number.isInteger(players)||players<2||players>4)throw Error('Use 2–4 players');
        return directories[players];
    }
    function guidedDirectoryForPlayers(players){directoryForPlayers(players);return guidedDirectories[players];}
    function strategyForProfile(profile){return profile==='teacher-trained-v2'?'human-card-v2':profile==='human-guide-v1'?'human-guide-v1':null;}
    function teacherDataForPlayers(players,base,teacher){
        directoryForPlayers(players);
        if(!corrected||base.schemaVersion!==teacher.schemaVersion||teacher.schemaVersion!=='brass-wm-economy-v2'||
            base.stateDim!==teacher.stateDim||base.actionDim!==teacher.actionDim||
            base.valueModel?.featureVersion!==teacher.valueModel?.featureVersion||!teacher.valueModel?.layers)
            throw Error('Incompatible teacher transfer');
        if(players===4)return teacher;
        // Guided search uses the real engine, so only the value head transfers.
        // World prediction retains native per-player dynamics and action encoding.
        return {...base,valueModel:teacher.valueModel,
            teacherTransfer:{players,valueSource:'human-teacher-20261009',nativeDynamics:true,
                note:'Four-player value head transferred without additional training; measured separately.'}};
    }
    const api=Object.freeze({directories,directoryForPlayers,guidedDirectories,guidedDirectoryForPlayers,strategyForProfile,teacherDataForPlayers});
    if(typeof module!=='undefined'&&module.exports)module.exports=api;
    else root.BrassRuntimeModels=api;
})(globalThis);
