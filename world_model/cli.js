#!/usr/bin/env node
const fs=require('node:fs');
const path=require('node:path');
const {spawnSync}=require('node:child_process');
const root=path.resolve(__dirname,'..');
const [command,...args]=process.argv.slice(2);
const numeric=new Set(['games','players','seed','depth','width','exploration','workers','stride']);
function options(argv){const result={};for(let i=0;i<argv.length;i++){if(!argv[i].startsWith('--'))throw Error('Expected --option');const [key,inline]=argv[i].slice(2).split('=');const value=inline??argv[++i];if(value===undefined)throw Error(`Missing --${key} value`);result[key]=numeric.has(key)?Number(value):value;}return result;}
function python(script){
    const candidates=[process.env.BRASS_PYTHON,'python','python3','py'].filter(Boolean);
    const bundled=process.env.USERPROFILE&&path.join(process.env.USERPROFILE,'.cache','codex-runtimes','codex-primary-runtime','dependencies','python','python.exe');
    if(bundled&&fs.existsSync(bundled))candidates.push(bundled);
    for(const bin of candidates){const probe=spawnSync(bin,['-c','import numpy'],{timeout:8000,windowsHide:true,stdio:'ignore'});if(probe.status!==0)continue;
        const run=spawnSync(bin,[path.join(__dirname,script),...args],{cwd:root,stdio:'inherit',windowsHide:true});
        if(run.error)throw run.error;process.exitCode=run.status||0;return;
    }
    throw Error('Python + NumPy required. Install world_model/requirements.txt, or set BRASS_PYTHON to your Python executable.');
}
async function main(){
    process.chdir(root);
    if(command==='generate'){require('./generate').generate(options(args));return;}
    if(command==='collect-value'){await require('./collect_value').collect(options(args));return;}
    if(command==='train'){python('train.py');return;}
    if(command==='train-structured'){python('train_structured.py');return;}
    if(command==='train-value'){python('train_value.py');return;}
    if(command==='compose'){python('compose.py');return;}
    if(command==='evaluate'){python('evaluate.py');return;}
    if(command==='compare'){python('compare_runs.py');return;}
    if(command==='compare-scores'){python('compare_scores.py');return;}
    if(command==='compare-geography'){python('compare_geography.py');return;}
    if(command==='merge-value'){python('merge_value_replay.py');return;}
    if(command==='evaluate-value'){python('evaluate_value.py');return;}
    if(command==='demo'){require('./server').start(options(args));return;}
    if(command==='benchmark'){
        const {Network}=require('./inference');const {tournament}=require('./tournament');
        const config=options(args),models=config.models||'world_model/models';
        if(config['guided-models'])config.guidedModels=config['guided-models'];
        if(typeof config.types==='string')config.types=config.types.split(',');
        if(config.workers!==undefined&&(!Number.isInteger(config.workers)||config.workers<1||config.workers>16))throw Error('Use 1–16 benchmark workers');
        const world=new Network(JSON.parse(fs.readFileSync(path.join(models,'world-model.json'),'utf8')));
        const policy=new Network(JSON.parse(fs.readFileSync(path.join(models,'neural-policy.json'),'utf8')));
        const run=config.workers>1?require('./benchmark_parallel').parallelTournament:tournament;
        const guidedWorld=config.guidedModels?new Network(JSON.parse(fs.readFileSync(path.join(config.guidedModels,'world-model.json'),'utf8'))):null;
        const report=await run({...config,models,world,policy,guidedWorld,onProgress:p=>{if(p.completed&&(p.game%20===0||p.game===p.games))console.log(`Game ${p.game}/${p.games}`,p.scores.map(s=>`${s.type}: ${s.vp}`).join(' | '));}});
        const crypto=require('node:crypto');
        report.artifacts=Object.fromEntries(['world-model.json','neural-policy.json'].map(file=>[file,{
            directory:models,sha256:crypto.createHash('sha256').update(fs.readFileSync(path.join(models,file))).digest('hex')} ]));
        if(config.guidedModels)report.artifacts.guided={directory:config.guidedModels,sha256:crypto.createHash('sha256').update(fs.readFileSync(path.join(config.guidedModels,'world-model.json'))).digest('hex')};
        const out=config.out||'world_model/reports/tournament.json';fs.mkdirSync(path.dirname(out),{recursive:true});fs.writeFileSync(out,JSON.stringify(report,null,2));
        console.table(report.rows);console.log(`Saved ${out}`);return;
    }
    console.log('Usage: node world_model/cli.js generate|collect-value|merge-value|train|train-structured|train-value|compose|evaluate|evaluate-value|compare|compare-scores|compare-geography|demo|benchmark [options]');
    if(command)process.exitCode=1;
}
main().catch(error=>{console.error(error.stack||error);process.exitCode=1;});
