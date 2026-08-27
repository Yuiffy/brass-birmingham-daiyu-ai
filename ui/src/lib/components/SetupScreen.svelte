<script lang="ts">
	import { onMount, createEventDispatcher } from 'svelte';
	import { listSavedGames, loadGame, newGame } from '$lib/api';
	import type { SavedGameSummary } from '$lib/api';
	import { History, Play } from 'lucide-svelte';

	const dispatch = createEventDispatcher<{ started: void; replay: number }>();
	let activeTab: 'new' | 'join' = 'new';
	let numPlayers = 2;
	let seedInput = '';
	let seedError = '';
	let savedGames: SavedGameSummary[] = [];
	let loadingGames = false;
	let loadingError = '';

	function formatCreatedAt(createdAt: number | string) {
		const fromString = typeof createdAt === 'string' ? Number(createdAt) : createdAt;
		const numeric = Number.isFinite(fromString) ? fromString : NaN;
		const millis = Number.isFinite(numeric)
			? (numeric > 1_000_000_000_000 ? numeric : numeric * 1000)
			: NaN;
		const parsed = Number.isFinite(millis) ? new Date(millis) : new Date(createdAt);
		return Number.isNaN(parsed.getTime()) ? String(createdAt) : parsed.toLocaleString();
	}

	async function refreshSavedGames() {
		loadingGames = true;
		loadingError = '';
		try {
			savedGames = await listSavedGames();
		} catch {
			loadingError = 'Failed to load saved games';
			savedGames = [];
		} finally {
			loadingGames = false;
		}
	}

	function switchTab(tab: 'new' | 'join') {
		activeTab = tab;
		if (tab === 'join') {
			void refreshSavedGames();
		}
	}

	async function startNewGame() {
		seedError = '';
		let seed: number | null = null;
		const raw = seedInput.trim();
		if (raw.length > 0) {
			const n = Number(raw);
			if (!Number.isInteger(n) || n < 0 || n > Number.MAX_SAFE_INTEGER) {
				seedError = 'Seed must be a non-negative integer';
				return;
			}
			seed = n;
		}
		const data = await newGame(numPlayers, seed);
		if (data) dispatch('started');
	}

	async function joinSavedGame(gameId: number) {
		const data = await loadGame(gameId);
		if (data) dispatch('started');
	}

	onMount(() => {
		void refreshSavedGames();
	});
</script>

<div class="setup">
	<h1>Brass Birmingham</h1>

	<div class="tabs">
		<button class:active={activeTab === 'new'} on:click={() => switchTab('new')}>New Game</button>
		<button class:active={activeTab === 'join'} on:click={() => switchTab('join')}>Join Game</button>
	</div>

	{#if activeTab === 'new'}
		<div class="row">
			<label for="np">Players</label>
			<select id="np" bind:value={numPlayers}>
				<option value={2}>2 Players</option>
				<option value={3}>3 Players</option>
				<option value={4}>4 Players</option>
			</select>
		</div>
		<div class="row">
			<label for="seed">Seed (optional)</label>
			<input id="seed" type="text" bind:value={seedInput} placeholder="Random if empty" />
			{#if seedError}
				<div class="seed-error">{seedError}</div>
			{/if}
		</div>
		<button class="btn" on:click={startNewGame}>Start New Game</button>
	{:else}
		<div class="join-list">
			{#if loadingGames}
				<div class="join-note">Loading saved games...</div>
			{:else if loadingError}
				<div class="join-note error">{loadingError}</div>
			{:else if savedGames.length === 0}
				<div class="join-note">No saved games found.</div>
			{:else}
				{#each savedGames as game}
					<div class="game-row">
						<div class="game-meta">
							<div class="game-title">{formatCreatedAt(game.created_at)}</div>
							<div class="game-details">
								Round {game.round_in_phase + 1} | {game.era} | {game.num_players}p | Seed {game.seed}
							</div>
						</div>
						<div class="game-actions">
							<button class="icon-btn" on:click={() => dispatch('replay', game.id)} title="查看棋谱" aria-label={`查看棋谱 ${game.id}`}>
								<History size={15} aria-hidden="true" />
							</button>
							<button class="btn join-btn" on:click={() => joinSavedGame(game.id)}><Play size={13} fill="currentColor" aria-hidden="true" /> Join</button>
						</div>
					</div>
				{/each}
			{/if}
		</div>
	{/if}
</div>

<style>
	.setup {
		max-width: 380px;
		margin: max(48px, 10svh) auto;
		text-align: center;
		background: #f5f6f4;
		padding: 40px;
		border-radius: 8px;
		border: 1px solid #b9bfbb;
		box-shadow: 0 18px 50px rgba(30, 35, 31, .14);
	}
	h1 { font-size: 28px; color: #202321; margin-bottom: 24px; letter-spacing: 0; }
	.row { margin-bottom: 16px; }
	.tabs {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 8px;
		margin-bottom: 18px;
	}
	.tabs button {
		padding: 8px 10px;
		border-radius: 6px;
		border: 1px solid #c8cdc9;
		background: #fff;
		color: #59605b;
		cursor: pointer;
		font-weight: 600;
	}
	.tabs button.active {
		background: #202321;
		border-color: #202321;
		color: #fff;
	}
	label { display: block; margin-bottom: 6px; color: #686e6a; font-size: 14px; }
	select {
		padding: 8px 20px; border-radius: 6px; border: 1px solid #b9bfbb;
		background: #fff; color: #202321; font-size: 14px;
	}
	input {
		padding: 8px 12px; border-radius: 6px; border: 1px solid #b9bfbb;
		background: #fff; color: #202321; font-size: 14px; width: 100%;
		box-sizing: border-box;
	}
	.seed-error { color: #b54031; font-size: 12px; margin-top: 6px; }
	.btn {
		padding: 10px 32px; border: none; border-radius: 8px; cursor: pointer;
		font-size: 15px; font-weight: 700; color: #fff; background: #087f5b;
		transition: background 0.15s;
	}
	.btn:hover { background: #066b4c; }
	.join-list {
		display: grid;
		gap: 10px;
		text-align: left;
	}
	.join-note {
		color: #5d635f;
		font-size: 14px;
		padding: 10px 12px;
		background: #fff;
		border-radius: 6px;
		border: 1px solid #c8cdc9;
	}
	.join-note.error { color: #b54031; }
	.game-row {
		display: flex;
		justify-content: space-between;
		align-items: center;
		gap: 12px;
		padding: 10px 12px;
		border-radius: 6px;
		border: 1px solid #c8cdc9;
		background: #fff;
	}
	.game-actions { display: flex; align-items: center; gap: 6px; flex-shrink: 0; }
	.icon-btn {
		width: 31px;
		height: 31px;
		display: grid;
		place-items: center;
		border: 1px solid #c8cdc9;
		border-radius: 6px;
		background: #fff;
		color: #4e554f;
		cursor: pointer;
	}
	.icon-btn:hover { border-color: #087f5b; color: #087f5b; }
	.game-meta { min-width: 0; }
	.game-title {
		color: #202321;
		font-size: 14px;
		font-weight: 700;
	}
	.game-details {
		color: #6c726e;
		font-size: 12px;
		margin-top: 2px;
	}
	.join-btn {
		padding: 8px 14px;
		font-size: 13px;
		white-space: nowrap;
		display: inline-flex;
		align-items: center;
		gap: 5px;
	}
</style>
