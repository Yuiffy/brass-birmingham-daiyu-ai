const {chromium}=require(process.env.BRASS_PLAYWRIGHT_MODULE || 'playwright');
const assert=require('node:assert/strict'),fs=require('node:fs');
const url=process.env.BRASS_DEMO_URL||'http://127.0.0.1:5179';
const output='output/native-trained-browser';fs.mkdirSync(output,{recursive:true});
(async()=>{
  const browser=await chromium.launch({headless:true}),errors=[],results=[];
  for(const n of [2,3,4]) {
    const context=await browser.newContext({viewport:{width:1440,height:1000}}),page=await context.newPage();
    page.on('pageerror',e=>errors.push(e.message));page.on('console',e=>{if(e.type()==='error')errors.push(e.text())});
    await page.goto(url);await page.locator('#np').selectOption(String(n));await page.locator('#seed').fill('730001');await page.getByRole('button',{name:'Start New Game'}).click();
    await page.waitForFunction(()=>typeof window.render_game_to_text==='function'&&JSON.parse(window.render_game_to_text()).mode!=='setup');
    const text=()=>page.evaluate(()=>JSON.parse(window.render_game_to_text()));
    // Pause any opening automatic turn before changing control mode.
    const pause=page.locator('button[aria-label="暂停 AI 对局"]');
    if(await pause.isEnabled())await pause.click();
    await page.getByRole('combobox',{name:'对局模式'}).selectOption('ai-vs-ai');
    await page.getByRole('button',{name:'2×',exact:true}).click();
    await page.waitForFunction(()=>JSON.parse(window.render_game_to_text()).ai_playback.moves>=2,{},{timeout:30000});
    await page.waitForFunction(()=>{const s=JSON.parse(window.render_game_to_text());return s.analysis?.method==='teacher_trained_v2_native_guided'&&s.ai_playback.status==='running'});
    await pause.click();await page.waitForTimeout(300);
    const paused=await text();assert.equal(paused.ai_playback.status,'paused');assert.equal(paused.analysis.model_id,'human-teacher-20261009-epoch105-native-v1');assert.equal(paused.game.players.length,n);
    await page.screenshot({path:`${output}/${n}p-ai.png`,fullPage:true});
    await page.waitForTimeout(1400);assert.equal((await text()).ai_playback.moves,paused.ai_playback.moves,'pause must remain paused after earlier moves');
    await page.getByRole('button',{name:'AI 分析或前进一步'}).click();
    await page.waitForFunction(m=>{const s=JSON.parse(window.render_game_to_text());return s.ai_playback.moves===m+1&&s.analysis?.method==='teacher_trained_v2_native_guided'},paused.ai_playback.moves);
    assert.equal((await text()).ai_playback.status,'paused');
    await page.getByRole('button',{name:'评分含义',exact:true}).click();await page.waitForFunction(()=>document.body.innerText.includes('20% 教师网络'));
    await page.getByRole('button',{name:'继续 AI 对局'}).click();await page.waitForFunction(m=>JSON.parse(window.render_game_to_text()).ai_playback.moves>m,paused.ai_playback.moves+1);await pause.click();
    await page.getByRole('combobox',{name:'对局模式'}).selectOption('manual');await page.getByRole('button',{name:'对局',exact:true}).click();
    // Original human controls still support staged choices, cancellation and undo.
    const start=page.getByRole('button',{name:'Start Turn',exact:true});if(await start.isVisible())await start.click();
    const stateBefore=await text();
    const loan=page.locator('button.action').filter({hasText:'Loan'});await loan.click();
    await page.waitForFunction(()=>JSON.parse(window.render_game_to_text()).mode==='in_session');
    await page.getByRole('button',{name:'Cancel',exact:true}).click();await page.waitForFunction(()=>JSON.parse(window.render_game_to_text()).mode==='choosing_action');
    assert.deepEqual((await text()).game.players,stateBefore.game.players);
    await loan.click();await page.locator('.card-slot.selectable').first().click();
    await page.locator('.btn.choice').first().click();
    await page.getByRole('button',{name:'Undo Previous Action',exact:true}).click();
    await page.waitForFunction(expected=>JSON.stringify(JSON.parse(window.render_game_to_text()).game.players)===expected,JSON.stringify(stateBefore.game.players));
    assert.deepEqual((await text()).game.players,stateBefore.game.players);
    // Save and reload through the actual browser-owned session path.
    const session=await page.evaluate(async()=>{
      return new Promise((resolve,reject)=>{const r=indexedDB.open('brass-original-saves',1);r.onsuccess=()=>{const q=r.result.transaction('sessions').objectStore('sessions').get('current');q.onsuccess=()=>{resolve(q.result);r.result.close()};q.onerror=()=>reject(q.error)};r.onerror=()=>reject(r.error)});
    });assert.ok(session);
    const saved=await text();await page.reload();await page.getByRole('button',{name:'Join Game',exact:true}).click();
    await page.locator('.join-list .join-btn').first().click();
    await page.waitForFunction(()=>JSON.parse(window.render_game_to_text()).mode!=='setup');
    assert.deepEqual((await text()).game.players,saved.game.players);
    results.push({players:n,model:paused.analysis.model_id,pause:true,step:true,resume:true,explanation:true,manualCancel:true,manualUndo:true,reload:true});console.log(`Verified ${n}P browser flow`);await context.close();
  }
  assert.deepEqual(errors,[]);fs.writeFileSync(`${output}/verification.json`,JSON.stringify({results,errors},null,2));console.log(JSON.stringify({results,errors}));await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
