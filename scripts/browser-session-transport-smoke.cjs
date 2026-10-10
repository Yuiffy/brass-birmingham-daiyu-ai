// Generate full saves with native-trained-session-smoke.cjs first. Exercise
// real browser gzip transport, old saves, replay, and new-game recovery.
const { chromium } = require(process.env.BRASS_PLAYWRIGHT_MODULE || 'playwright');
const assert = require('node:assert/strict'), fs = require('node:fs');
const { gunzipSync } = require('node:zlib');
const url = process.env.BRASS_DEMO_URL || 'http://127.0.0.1:5182';
const output = process.env.BRASS_TRANSPORT_OUTPUT || 'output/browser-session-transport';
const limit = 4.5 * 1024 * 1024;
(async () => {
  fs.mkdirSync(output, { recursive: true });
  const browser = await chromium.launch(), results = [], errors = [];
  for (const players of [2, 3, 4]) {
    const context = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
    const page = await context.newPage(), requests = [];
    page.on('pageerror', e => errors.push(e.message));
    page.on('request', r => {
      if (!r.url().endsWith('/api/browser_request')) return;
      const bytes = r.postDataBuffer();
      assert.equal(r.headers()['content-encoding'], 'gzip');
      assert.ok(bytes.length < limit, `request exceeds gateway limit: ${bytes.length}`);
      const unpacked = gunzipSync(bytes);
      requests.push({ endpoint: JSON.parse(unpacked).endpoint, wireBytes: bytes.length, jsonBytes: unpacked.length });
    });
    await page.goto(url);
    const saved = JSON.parse(fs.readFileSync(`output/native-trained-sessions/${players}p.json`));
    await page.evaluate(async saved => {
      const db = await new Promise((resolve, reject) => {
        const r = indexedDB.open('brass-original-saves', 1);
        r.onupgradeneeded = () => r.result.createObjectStore('sessions');
        r.onsuccess = () => resolve(r.result); r.onerror = () => reject(r.error);
      });
      await new Promise((resolve, reject) => {
        const t = db.transaction('sessions', 'readwrite');
        t.objectStore('sessions').put(saved, 'current');
        t.oncomplete = resolve; t.onerror = () => reject(t.error);
      }); db.close();
    }, saved);
    await page.reload();
    await page.getByRole('button', { name: 'Join Game', exact: true }).click();
    await page.locator('.join-btn').first().waitFor();
    // Replays still carry their full position and analysis history.
    const replay = await context.request.post(url + '/api/browser_request', {
      headers: { 'Content-Type': 'application/json', 'Content-Encoding': 'gzip', 'Accept-Encoding': 'gzip' },
      data: require('node:zlib').gzipSync(JSON.stringify({ session: saved, endpoint: 'replay', body: { game_id: 1 } }))
    });
    assert.equal(replay.status(), 200);
    assert.equal(replay.headers()['content-encoding'], 'gzip');
    const history = await replay.json();
    assert.equal(history.ok, true); assert.ok(history.positions.length > 10);
    assert.equal(history.browser_session.games[0].action_log, saved.games[0].action_log);
    await page.locator('.join-btn').first().click();
    await page.waitForFunction(() => JSON.parse(window.render_game_to_text()).mode === 'game_over');
    const scores = await page.evaluate(() => JSON.parse(window.render_game_to_text()).game.players.map(p => p.vp));
    await page.screenshot({ path: `${output}/${players}p-restored.png` });
    await page.reload();
    await page.getByRole('button', { name: 'Start New Game', exact: true }).click();
    await page.waitForFunction(() => JSON.parse(window.render_game_to_text()).mode !== 'setup');
    assert.ok(!await page.getByRole('alert').count());
    await page.screenshot({ path: `${output}/${players}p-new.png` });
    results.push({ players, savedBytes: Buffer.byteLength(JSON.stringify(saved)), scores,
      replayPositions: history.positions.length, restored: true, newGame: true, requests });
    await context.close(); console.log(`Verified compressed ${players}P saved game`);
  }
  assert.deepEqual(errors, []);
  fs.writeFileSync(`${output}/verification.json`, JSON.stringify({ results, errors }, null, 2));
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
