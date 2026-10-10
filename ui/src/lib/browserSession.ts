// Complete replay snapshots outgrow localStorage's small synchronous quota.
// Keep the same session format in IndexedDB and migrate existing browser saves.
export const legacySessionKey = 'brass-original-saved-games-v1';
let database: Promise<IDBDatabase> | undefined;

function openDatabase(): Promise<IDBDatabase> {
	return database ??= new Promise((resolve, reject) => {
		const request = indexedDB.open('brass-original-saves', 1);
		request.onupgradeneeded = () => request.result.createObjectStore('sessions');
		request.onsuccess = () => resolve(request.result);
		request.onerror = () => { database = undefined; reject(request.error); };
	});
}

export async function readBrowserSession(): Promise<unknown> {
	const db = await openDatabase();
	const saved = await new Promise<unknown>((resolve, reject) => {
		const request = db.transaction('sessions').objectStore('sessions').get('current');
		request.onsuccess = () => resolve(request.result);
		request.onerror = () => reject(request.error);
	});
	if (saved !== undefined) return saved;
	const legacy = localStorage.getItem(legacySessionKey);
	return legacy ? JSON.parse(legacy) : {};
}

export async function saveBrowserSession(session: unknown): Promise<void> {
	const db = await openDatabase();
	await new Promise<void>((resolve, reject) => {
		const transaction = db.transaction('sessions', 'readwrite');
		transaction.objectStore('sessions').put(session, 'current');
		transaction.oncomplete = () => resolve();
		transaction.onerror = () => reject(transaction.error);
		transaction.onabort = () => reject(transaction.error ?? new Error('Save aborted'));
	});
	localStorage.removeItem(legacySessionKey);
}
