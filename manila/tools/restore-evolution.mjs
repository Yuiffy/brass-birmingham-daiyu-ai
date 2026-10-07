// Restore raw stage outputs from a published report without replaying games.
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { gunzipSync } from 'node:zlib';
import { createHash } from 'node:crypto';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { aggregateMatches } from '../tournament.mjs';
import { ranked, pairDetails, challengeDetails, pairedFinals } from '../league.mjs';
const source=path.resolve(process.argv[2]||'reports/2026-10-07-evolution'),destination=path.resolve(process.argv[3]||'output/evolution-restored');
const root=path.dirname(path.dirname(fileURLToPath(import.meta.url))),sha=data=>createHash('sha256').update(data).digest('hex');
const report=JSON.parse(await readFile(path.join(source,'summary.json'),'utf8'));
for(const [file,hash] of Object.entries(report.sourceLF_SHA256))if(sha((await readFile(path.join(root,file),'utf8')).replaceAll('\r\n','\n'))!==hash)throw Error(`Source mismatch: ${file}; use the report's frozen commit`);
for(const stage of report.stages){
  const filename=stage.stage+'.jsonl.gz',compressed=await readFile(path.join(source,filename));
  if(sha(compressed)!==report.artifactSHA256[filename])throw Error(`Archive mismatch: ${filename}`);
  const raw=gunzipSync(compressed).toString('utf8'),matches=raw.trim().split('\n').map(JSON.parse);
  const {config}=stage.manifest,summary=aggregateMatches(matches,'league');
  const run={...stage.manifest,sourceLF_SHA256:report.sourceLF_SHA256,summary};
  const selection={stage:stage.stage,config,completedGames:matches.length,summary:ranked(summary),pairDetails:pairDetails(matches),challenge:config.challengers?challengeDetails(matches):undefined,paired:config.finalists?pairedFinals(matches,config.finalists):undefined};
  const directory=path.join(destination,stage.stage);await mkdir(directory,{recursive:true});
  await writeFile(path.join(directory,'matches.jsonl'),raw);
  await writeFile(path.join(directory,'summary.json'),JSON.stringify(run,null,2));
  await writeFile(path.join(directory,'selection.json'),JSON.stringify(selection,null,2));
}
console.log(`Restored ${report.completedGames} terminal games to ${destination}`);
