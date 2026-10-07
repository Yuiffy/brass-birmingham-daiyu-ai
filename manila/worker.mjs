import { analyzeAuction } from './ai.mjs';
self.onmessage = async ({ data }) => {
  try { const result = await analyzeAuction(data.state, data.observer, data.options, progress => self.postMessage({ progress })); self.postMessage({ result }); }
  catch (e) { self.postMessage({ error: e.message }); }
};
