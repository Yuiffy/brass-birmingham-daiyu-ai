// Run against the demo server. Install playwright or set BRASS_PLAYWRIGHT_MODULE.
const {chromium}=require(process.env.BRASS_PLAYWRIGHT_MODULE||'playwright');
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
(async()=>{
 const directory=path.resolve('output/human-strategy/controls');fs.mkdirSync(directory,{recursive:true});
 const browser=await chromium.launch({headless:true});
 try{
  const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[];
  page.on('pageerror',error=>errors.push(error.message));
  page.on('console',message=>{if(message.type()==='error')errors.push(message.text());});
  await page.goto(process.env.BRASS_DEMO_URL||'http://127.0.0.1:8097/?rules=economy-v2&strategy=human-guide-v1');
  await page.waitForFunction(()=>BrassAI.getGuided(4)?.valueLayers);
  await page.locator('.count-btn[data-count="4"]').click();
  assert.equal(await page.locator('#guided-strategy').inputValue(),'human-guide-v1');
  for(const select of await page.locator('.player-ai-select').all())await select.selectOption('guided');
  await page.locator('#start-game-btn').click();await page.locator('#ai-pause').click();
  const snapshot=()=>page.evaluate(()=>JSON.stringify(BrassSimulator.snapshot(gameState)));
  const paused=await snapshot();await page.waitForTimeout(1200);assert.equal(await snapshot(),paused);
  await page.locator('#ai-step').click();assert.notEqual(await snapshot(),paused);
  assert.equal(await page.evaluate(()=>brassController.lastPlan.strategy),'human-guide-v1');
  await page.locator('#ai-inspect').click();
  assert.equal(await page.evaluate(()=>brassController.preview.type),'guided');
  assert.equal(await page.evaluate(()=>brassController.preview.strategy),'human-guide-v1');
  const preview=await snapshot();await page.locator('#ai-rethink').click();assert.equal(await snapshot(),preview);
  await page.screenshot({path:path.join(directory,'guided-inspect.png')});
  await page.locator('#ai-execute').click();assert.notEqual(await snapshot(),preview);
  assert.equal(await page.locator('#ai-execute').isDisabled(),true);
  await page.locator('#ai-close').click();await page.locator('#ai-pause').click();
  await page.waitForFunction(()=>brassController.running&&brassController.lastPlan);
  await page.waitForTimeout(1400);await page.locator('#ai-pause').click();
  assert.notEqual(await snapshot(),preview);

  // A supplied legal rail position exercises the double-link preview and commit.
  await page.evaluate(()=>{
   const s=BrassSimulator.create(4,203010001);s.endCanalEra();s.currentPlayerIndex=s.turnOrder.indexOf(0);
   s.players[0].money=50;s.players[0].hand=[{type:'location',location:'birmingham'},{type:'location',location:'coventry'}];
   s.boardIndustries.birmingham_0={playerId:0,type:'brewery',tileData:{...INDUSTRY_DATA.brewery[1]},flipped:false,resourceCubes:2};
   Object.assign(gameState,BrassSimulator.snapshot(s));brassController.running=false;uiManager.refresh();
  });
  await page.locator('#ai-inspect').click();
  await page.evaluate(()=>{
   const action=BrassSimulator.candidates(gameState,{doubleRail:true}).find(a=>a.target?.connectionIds?.join()==='birmingham-oxford,birmingham-coventry');
   if(!action)throw Error('Double rail missing');
   brassController.preview.branches=[{action,label:BrassSimulator.label(action)}];brassController.compare(0);
  });
  assert.match(await page.locator('#ai-truth-status').textContent(),/未训练双铁路/);
  await page.screenshot({path:path.join(directory,'double-preview.png')});
  const before=await page.evaluate(()=>({money:gameState.players[0].money,cards:gameState.players[0].hand.length}));
  await page.locator('#ai-execute').click();
  const after=await page.evaluate(()=>({money:gameState.players[0].money,cards:gameState.players[0].hand.length,
   cubes:gameState.boardIndustries.birmingham_0.resourceCubes,links:Object.keys(gameState.boardLinks),actions:gameState.actionsThisTurn}));
  assert.ok(after.money<=before.money-15);assert.equal(after.cards,before.cards-1);assert.equal(after.cubes,1);
  assert.ok(after.links.includes('birmingham-oxford')&&after.links.includes('birmingham-coventry'));assert.equal(after.actions,1);
  await page.locator('#ai-close').click();await page.screenshot({path:path.join(directory,'double-executed.png')});
  assert.equal(JSON.parse(await page.evaluate(()=>render_game_to_text())).boardLinks,after.links.length);
  assert.deepEqual(errors,[]);fs.writeFileSync(path.join(directory,'verification.json'),JSON.stringify({paused:true,step:true,
   actualGuidedInspection:true,execute:true,resume:true,doubleRail:after,errors},null,2));
  console.log('Verified guided controls and legal double rail; '+directory);
 }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
