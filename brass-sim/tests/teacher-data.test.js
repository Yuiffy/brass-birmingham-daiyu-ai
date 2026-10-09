const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),os=require('node:os'),zlib=require('node:zlib');
process.env.BRASS_RULES='economy-v2';
const root=path.resolve(__dirname,'..'),{augment}=require('../world_model/augment_double_rails');
test('audited doubles augment training only and keep validation states and successors intact',()=>{
 const source=path.join(root,'world_model/experiments/human-teacher-20261009/reproducibility/teacher-transitions');
 const schema=JSON.parse(fs.readFileSync(path.join(source,'schema.json'))),all=JSON.parse(fs.readFileSync(path.join(source,'games.json')));
 const bytes=zlib.gunzipSync(fs.readFileSync(path.join(source,'transitions.f32.gz'))),rw=schema.rowWidth,sd=schema.stateDim,ad=schema.actionDim;
 const selected=[all.find(g=>g.split==='train'),all.find(g=>g.split==='validation')],chunks=[],games=[];let rows=0;
 for(const g of selected){chunks.push(bytes.subarray(g.start*rw*4,g.end*rw*4));games.push({...g,start:rows,end:rows+g.end-g.start});rows+=g.end-g.start;}
 const temp=fs.mkdtempSync(path.join(os.tmpdir(),'brass-teacher-audit-'));
 try{
  const data=path.join(temp,'data'),out=path.join(temp,'out');fs.mkdirSync(data);
  fs.writeFileSync(path.join(data,'schema.json'),JSON.stringify({...schema,rows}));fs.writeFileSync(path.join(data,'games.json'),JSON.stringify(games));
  fs.writeFileSync(path.join(data,'transitions.f32'),Buffer.concat(chunks));
  augment({data,out,teacherRoot:root,maxPerState:1});
  const result=JSON.parse(fs.readFileSync(path.join(out,'schema.json'))),records=JSON.parse(fs.readFileSync(path.join(out,'games.json')));
  assert.ok(result.provenance.addedRows>0);assert.equal(records[1].end-records[1].start,games[1].end-games[1].start);
  const raw=fs.readFileSync(path.join(out,'transitions.f32'));
  for(let i=0;i<records[1].end-records[1].start;i++){
   const old=chunks[1].subarray(i*rw*4,(i+1)*rw*4),next=raw.subarray((records[1].start+i)*rw*4,(records[1].start+i+1)*rw*4);
   assert.deepEqual(next.subarray(0,sd*4),old.subarray(0,sd*4));assert.deepEqual(next.subarray((sd+ad)*4),old.subarray((sd+ad)*4));
  }
  const corrupt=Buffer.concat(chunks);corrupt.writeFloatLE(99,0);fs.writeFileSync(path.join(data,'transitions.f32'),corrupt);
  assert.throws(()=>augment({data,out:path.join(temp,'bad'),teacherRoot:root,maxPerState:0}),/Replay state mismatch/);
 }finally{fs.rmSync(temp,{recursive:true,force:true});}
});
