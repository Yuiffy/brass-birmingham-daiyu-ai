import { parentPort, workerData } from 'node:worker_threads';
import { playMatch } from '../tournament.mjs';
parentPort.on('message',task=>{
  try { parentPort.postMessage({id:task.id,match:{...playMatch({...task,validate:workerData.validate,searchOptions:workerData.searchOptions}),mode:task.mode,focal:task.focal,pair:task.pair,composition:task.composition,challenger:task.challenger,stage:task.stage}}); }
  catch(e){parentPort.postMessage({id:task.id,error:e.stack});}
});
