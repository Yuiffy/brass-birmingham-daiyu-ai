// Check large completed saves and migration of the previous localStorage format.
const {chromium}=require(process.env.BRASS_PLAYWRIGHT_MODULE||'playwright');
const fs=require('node:fs'),assert=require('node:assert/strict');
const url=process.env.BRASS_DEMO_URL||'http://127.0.0.1:3014';
(async()=>{
  const browser=await chromium.launch({headless:true}),errors=[],results=[];
  for(const players of [2,3,4]) {
    const context=await browser.newContext(),page=await context.newPage();
    page.on('pageerror',e=>errors.push(e.message));
    page.on('console',e=>{if(e.type()==='error')errors.push(e.text())});
    await page.goto(url);
    const session=JSON.parse(fs.readFileSync(`output/native-trained-sessions/${players}p.json`));
    await page.evaluate(async session=>{
      const db=await new Promise((resolve,reject)=>{const r=indexedDB.open('brass-original-saves',1);r.onupgradeneeded=()=>r.result.createObjectStore('sessions');r.onsuccess=()=>resolve(r.result);r.onerror=()=>reject(r.error)});
      await new Promise((resolve,reject)=>{const t=db.transaction('sessions','readwrite');t.objectStore('sessions').put(session,'current');t.oncomplete=resolve;t.onerror=()=>reject(t.error)});db.close();
    },session);
    await page.reload();await page.getByRole('button',{name:'Join Game',exact:true}).click();await page.locator('.join-list .join-btn').first().click();
    await page.waitForFunction(()=>JSON.parse(window.render_game_to_text()).mode==='game_over');
    const scores=await page.evaluate(()=>JSON.parse(window.render_game_to_text()).game.players.map(p=>p.vp));
    results.push({players,largeSaveReload:true,bytes:Buffer.byteLength(JSON.stringify(session)),scores});
    await context.close();
  }
  const context=await browser.newContext(),page=await context.newPage();await page.goto(url);
  const legacy=JSON.parse(fs.readFileSync('src/web/fixtures/trained-session.json'));
  await page.evaluate(s=>localStorage.setItem('brass-original-saved-games-v1',JSON.stringify(s)),legacy);
  await page.reload();await page.getByRole('button',{name:'Join Game',exact:true}).click();await page.locator('.join-list .join-btn').first().click();
  await page.waitForFunction(()=>JSON.parse(window.render_game_to_text()).mode!=='setup');
  assert.equal(await page.evaluate(()=>localStorage.getItem('brass-original-saved-games-v1')),null);
  await page.reload();await page.getByRole('button',{name:'Join Game',exact:true}).click();await page.locator('.join-list .join-btn').first().click();
  await page.waitForFunction(()=>JSON.parse(window.render_game_to_text()).mode!=='setup');
  assert.equal(await page.evaluate(()=>JSON.parse(window.render_game_to_text()).game.players.length),3);
  assert.deepEqual(errors,[]);const output={results,legacyMigration:true,errors};
  fs.writeFileSync('docs/native-trained-ai-20261010/save-verification.json',JSON.stringify(output,null,2)+'\n');console.log(JSON.stringify(output));await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
