// Package the independent JavaScript AI page under /ai-lab/.
const fs=require('node:fs'),path=require('node:path');
const root=path.resolve(__dirname,'..'),out=path.join(root,'dist','ai-lab');
const allowed=new Set(['.html','.css','.js','.json','.md','.gz','.zip']);
let files=0,bytes=0;
function copy(relative){
 const source=path.join(root,relative),target=path.join(out,relative);
 if(fs.statSync(source).isDirectory()){
  for(const item of fs.readdirSync(source,{withFileTypes:true})){
   if(item.name.startsWith('.')||item.name.startsWith('data')||item.name==='__pycache__'||item.isSymbolicLink())continue;
   copy(path.join(relative,item.name));
  }
 }else if(allowed.has(path.extname(source))){
  fs.mkdirSync(path.dirname(target),{recursive:true});fs.copyFileSync(source,target);
  files++;bytes+=fs.statSync(source).size;
 }
}
for(const relative of ['index.html','arena.html','css','js','scripts/autorun.js','world_model','docs'])copy(relative);
console.log(`Packaged ${files} public files (${(bytes/1024/1024).toFixed(1)} MiB) at ${out}`);
