(function(root){
    'use strict';
    // Paths are relative to world_model, shared by the game and arena worker.
    const directories=Object.freeze({
        2:'experiments/dynamics-2p-active/models',
        3:'experiments/dynamics-3p-active/models',
        4:'models'
    });
    function directoryForPlayers(players){
        if(!Number.isInteger(players)||players<2||players>4)throw Error('Use 2–4 players');
        return directories[players];
    }
    const api=Object.freeze({directories,directoryForPlayers});
    if(typeof module!=='undefined'&&module.exports)module.exports=api;
    else root.BrassRuntimeModels=api;
})(globalThis);
