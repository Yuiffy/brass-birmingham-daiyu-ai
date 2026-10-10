// Run against the local demo server; install playwright or set BRASS_PLAYWRIGHT_MODULE.
const {chromium}=require(process.env.BRASS_PLAYWRIGHT_MODULE||'playwright');
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
(async()=>{
 const base=process.env.BRASS_DEMO_URL||'http://127.0.0.1:8097/';
 const directory=path.resolve('output/browser-options');fs.mkdirSync(directory,{recursive:true});
 const browser=await chromium.launch({headless:true});const errors=[],observed=[];
 try{
  const page=await browser.newPage({viewport:{width:1280,height:960}});
  page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});
  for(const rules of ['legacy-v1','economy-v2']){
   await page.goto(`${base}?rules=${rules}`);await page.waitForFunction(()=>BrassAI.world&&BrassBenchmarkScores);
   if(rules==='economy-v2')assert.equal(await page.evaluate(()=>BrassAI.loadTeacher()),true);
   for(const players of [2,3,4]){
    await page.locator(`.count-btn[data-count="${players}"]`).click();
    for(const profile of rules==='economy-v2'?['','human-guide-v1','teacher-trained-v2']:['']){
     if(rules==='economy-v2')await page.locator('#guided-strategy').selectOption(profile);
     const labels=await page.locator('.player-ai-select').first().locator('option').allTextContents();
     assert.equal(labels.length,6);assert.equal(labels.filter(t=>t.includes('平均约')).length,5);
     assert.equal(labels.some(t=>/待测|仅支持四人|评分不可用/.test(t)),false);
     const checked=await page.evaluate(({rules,players,profile})=>{
      const rows=BrassBenchmarkScores.scores.filter(r=>r.rules===rules&&r.players===players&&r.profile===profile);
      return rows.length===5&&rows.every(r=>r.games===32&&BrassAILabels.optionLabel(r.type,{rules,players,profile})===
       document.querySelector(`.player-ai-select option[value="${r.type}"]`).textContent);
     },{rules,players,profile});assert.equal(checked,true);
     observed.push({rules,players,profile,labels});
    }
   }
  }
  for(const players of [2,3,4]){
   await page.goto(`${base}?rules=economy-v2&strategy=teacher-trained-v2&players=${players}`);
   await page.waitForFunction(()=>BrassAI.teacherByPlayers?.[2]&&BrassAI.teacherByPlayers?.[3]&&BrassAI.teacherWorld);
   assert.equal(await page.locator('.count-btn.active').getAttribute('data-count'),String(players));
   for(const select of await page.locator('.player-ai-select').all())await select.selectOption('guided');
   await page.screenshot({path:path.join(directory,`teacher-${players}p-setup.png`),fullPage:true});
   await page.locator('#start-game-btn').click();await page.locator('#ai-pause').click();
   assert.equal(await page.evaluate(()=>gameState.numPlayers),players);
   assert.equal(await page.evaluate(()=>brassController.planOptions('guided',2).world===BrassAI.getGuided(gameState.numPlayers,'teacher-trained-v2')),true);
   const before=await page.evaluate(()=>JSON.stringify(BrassSimulator.snapshot(gameState)));
   await page.locator('#ai-step').click();assert.notEqual(await page.evaluate(()=>JSON.stringify(BrassSimulator.snapshot(gameState))),before);
   assert.equal(await page.evaluate(()=>brassController.lastPlan.strategy),'human-card-v2');
   await page.evaluate(()=>document.getElementById('ai-depth').value='1');
   await page.locator('#ai-inspect').click();
   assert.equal(await page.evaluate(()=>brassController.preview.type==='guided'&&brassController.preview.strategy==='human-card-v2'&&brassController.preview.branches.length>0),true);
   const preview=await page.evaluate(()=>JSON.stringify(BrassSimulator.snapshot(gameState)));
   await page.locator('#ai-rethink').click();assert.equal(await page.evaluate(()=>JSON.stringify(BrassSimulator.snapshot(gameState))),preview);
   await page.screenshot({path:path.join(directory,`teacher-${players}p-inspect.png`)});
   await page.locator('#ai-execute').click();assert.notEqual(await page.evaluate(()=>JSON.stringify(BrassSimulator.snapshot(gameState))),preview);
   await page.locator('#ai-close').click();
  }
  await page.goto(`${base}?rules=economy-v2&strategy=teacher-trained-v2&players=2`);
  await page.waitForFunction(()=>BrassAI.teacherWorld);await page.setViewportSize({width:375,height:812});
  await page.locator('.player-ai-select').nth(1).selectOption('guided');
  await page.locator('#guided-strategy').scrollIntoViewIfNeeded();
  await page.screenshot({path:path.join(directory,'teacher-2p-mobile.png'),fullPage:true});
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
  assert.equal(await page.locator('#ai-enabled').count(),0);assert.deepEqual(errors,[]);
  fs.writeFileSync(path.join(directory,'verification.json'),JSON.stringify({observedCombinations:observed.length*5,observed,teacherPlayCounts:[2,3,4],errors},null,2));
  console.log('Verified all 60 score labels and actual teacher play/inspection/execution for 2P, 3P and 4P; no browser errors.');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
