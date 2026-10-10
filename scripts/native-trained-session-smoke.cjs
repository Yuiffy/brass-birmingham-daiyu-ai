// Complete trained games through stateless cloud requests, including analysis
// reconstruction on every apply. Each browser session remains independent.
const assert=require('node:assert/strict'),fs=require('node:fs');
const url=process.env.BRASS_DEMO_URL||'http://127.0.0.1:3014';
async function game(players) {
  let session={},state;const steps=[];
  async function call(endpoint,body={}) {
    const response=await fetch(`${url}/api/browser_request`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({endpoint,body,session})});
    assert.ok(response.ok,`${players}P ${endpoint}: HTTP ${response.status}`);const data=await response.json();
    if(data.ok!==true)fs.writeFileSync(`.tmp-native-session-failure-${players}.json`,JSON.stringify({endpoint,body,session,state,error:data.error}));
    assert.equal(data.ok,true,`${players}P ${endpoint}: ${data.error}`);session=data.browser_session;if(data.state)state=data.state;return data;
  }
  await call('new_game',{num_players:players,seed:730001});
  while(!state.game_over) {
    if(state.has_pending_shortfall)await call('resolve_shortfalls');
    if(!state.actions_remaining)await call('start_turn');
    if(state.game_over)break;
    const {analysis}=await call('analyze',{simulations:1,top_n:3});
    assert.equal(analysis.model_id,'human-teacher-20261009-epoch105-native-v1');
    const action_key=analysis.recommendations[0].action_key;
    await call('apply_analyzed_action',{revision:analysis.revision,action_key});steps.push(action_key);
    assert.ok(steps.length<=400);if(steps.length%40===0)console.log(`${players}P: ${steps.length} legal saved actions`);
  }
  // Buildings are serialized from a HashMap; compare their semantic slot order.
  const canonical=s=>({...s,buildings:[...s.buildings].sort((a,b)=>a.location-b.location)});
  const terminal=canonical(structuredClone(state));await call('load_game',{game_id:1});assert.deepEqual(canonical(state),terminal);
  fs.mkdirSync('output/native-trained-sessions',{recursive:true});fs.writeFileSync(`output/native-trained-sessions/${players}p.json`,JSON.stringify(session));
  return {players,actions:steps.length,terminal:true,restored:true,sessionBytes:Buffer.byteLength(JSON.stringify(session)),scores:state.players.map(p=>p.victory_points)};
}
(async()=>{const results=await Promise.all([2,3,4].map(game));const output={results,errors:[]};fs.writeFileSync('docs/native-trained-ai-20261010/session-verification.json',JSON.stringify(output,null,2)+'\n');console.log(JSON.stringify(output));})().catch(e=>{console.error(e);process.exit(1)});
