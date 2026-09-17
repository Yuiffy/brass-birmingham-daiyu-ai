importScripts('../js/gameData.js','../js/gameState.js','../js/gameLogic.js','../scripts/autorun.js',
    'simulator.js','encoding.js','value_features.js','inference.js','planner.js','tournament.js','runtime_models.js');
let cancelled=false;
onmessage=async({data})=>{
    if(data.type==='cancel'){cancelled=true;return;}
    if(data.type!=='run')return;
    cancelled=false;
    try{
        const directory=data.config.currentModels?BrassRuntimeModels.directoryForPlayers(data.config.types?.length||4):data.config.models||'models';
        if(!['models','experiments/v1/models','experiments/longer-1k/models','experiments/expanded-10k/models','experiments/structured-value-10k/models','experiments/score-v2/models',...Object.values(BrassRuntimeModels.directories)].includes(directory))throw Error('Unknown model version');
        const load=async file=>{const r=await fetch(`${directory}/${file}`);if(!r.ok)throw Error(`Missing ${file}`);return new BrassWorldModel.Network(await r.json());};
        const [world,policy]=await Promise.all([load('world-model.json'),load('neural-policy.json')]);
        const result=await BrassTournament.tournament({...data.config,world,policy,cancelled:()=>cancelled,
            onProgress:p=>postMessage({type:'progress',progress:p})});
        result.runtimeModels={players:data.config.types?.length||4,directory};
        postMessage({type:'result',result});
    }catch(e){postMessage({type:'error',message:e.message});}
};
