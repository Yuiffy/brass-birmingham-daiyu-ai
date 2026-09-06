<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import { ChevronLeft, ChevronRight, X } from 'lucide-svelte';
	import { gameState } from '$lib/store';
	import { modalDialog } from '$lib/dialog';
	import { cardImage, isLocationCard, locationCardTown, townCardColor } from '$lib/coords';
	import type { Card } from '$lib/types';

	const dispatch = createEventDispatcher();
	export let open = false;
	export let playerIndex: number | null = null;

	let currentIdx = 0;
	let previousPlayer: number | null = null;
	let wasOpen = false;
	let wheelDelta = 0;
	let lastWheel = 0;

	$: gs = $gameState;
	$: allEntries = gs?.discard_history ?? [];
	$: playerEntries = playerIndex == null ? allEntries : allEntries.filter(entry => entry.player_index === playerIndex);
	$: player = gs?.players.find(player => player.index === playerIndex) ?? null;
	$: if (open !== wasOpen || playerIndex !== previousPlayer) {
		currentIdx = 0;
		wheelDelta = 0;
		wasOpen = open;
		previousPlayer = playerIndex;
	}
	$: currentIdx = Math.min(currentIdx, Math.max(0, playerEntries.length - 1));
	$: entry = playerEntries[currentIdx] ?? null;
	$: card = entry ? { index: entry.order, label: entry.card_label, card_type: entry.card_type } satisfies Card : null;
	$: title = card ? locationCardTown(card) ?? card.label : '';
	$: category = card?.card_type === 'WildLocation' ? 'Wild location'
		: card?.card_type === 'WildIndustry' ? 'Wild industry'
		: card && isLocationCard(card.card_type) ? 'Location' : 'Industry';

	function close() {
		open = false;
		dispatch('close');
	}

	function go(delta: number) {
		currentIdx = Math.max(0, Math.min(currentIdx + delta, playerEntries.length - 1));
	}

	function onKeydown(event: KeyboardEvent) {
		if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
			event.preventDefault();
			go(event.key === 'ArrowRight' ? 1 : -1);
		}
	}

	function onWheel(event: WheelEvent) {
		event.preventDefault();
		const now = performance.now();
		if (now - lastWheel < 180) return;
		wheelDelta += event.deltaY;
		if (Math.abs(wheelDelta) < 40) return;
		go(wheelDelta > 0 ? 1 : -1);
		wheelDelta = 0;
		lastWheel = now;
	}
</script>

{#if open}
	<dialog class="overlay" use:modalDialog={{ onClose: close, onKeydown }} aria-label="Discard Pile">
		<div class="modal">
			<header class="header">
				<h2>Discard Pile {#if player}<span class="for-player" style:color={player.color}>{player.name}</span>{/if}</h2>
				<button class="close-btn" on:click={close} title="Close Discard Pile" aria-label="Close Discard Pile"><X size={20} /></button>
			</header>
			{#if !entry || !card}
				<div class="empty">No discarded cards yet.</div>
			{:else}
				<div class="viewer" on:wheel|nonpassive={onWheel}>
					<button class="nav prev" on:click={() => go(-1)} disabled={currentIdx === 0} title="Previous card" aria-label="Previous card"><ChevronLeft size={20} /></button>
					<figure class="discard-entry">
						<div class="center-card" class:location={isLocationCard(card.card_type)} style:--town-color={townCardColor(card.label)}>
							{#if isLocationCard(card.card_type)}
								<span class="town-name">{title}</span>
							{:else}
								<img src={cardImage(card.label, card.card_type)} alt={title} />
							{/if}
						</div>
						<figcaption>
							<strong>{title}</strong>
							<span class="card-type">{category}</span>
							<div class="meta"><span>Round {entry.round_in_phase + 1}</span><span>Turn {entry.turn_count + 1}</span></div>
						</figcaption>
					</figure>
					<button class="nav next" on:click={() => go(1)} disabled={currentIdx === playerEntries.length - 1} title="Next card" aria-label="Next card"><ChevronRight size={20} /></button>
				</div>
				<div class="counter" role="status" aria-label="Card position">{currentIdx + 1} / {playerEntries.length}</div>
			{/if}
		</div>
	</dialog>
{/if}

<style>
	.overlay {
		position: fixed;
		inset: 0;
		width: 100%;
		height: 100%;
		max-width: none;
		max-height: none;
		margin: 0;
		padding: 12px;
		border: 0;
		background: transparent;
		display: flex;
		align-items: center;
		justify-content: center;
	}
	.overlay::backdrop { background: rgba(0, 0, 0, 0.7); }
	.modal {
		background: #202522;
		color: #e6eee8;
		border: 1px solid #46524a;
		border-radius: 8px;
		width: min(480px, 100%);
		max-height: 100%;
		overflow-y: auto;
		overscroll-behavior: contain;
		padding: 18px;
	}
	.header { display: flex; align-items: center; gap: 12px; margin-bottom: 18px; }
	.header h2 { flex: 1; min-width: 0; font-size: 18px; line-height: 1.4; }
	.for-player { display: block; font-size: 13px; font-weight: 500; }
	.close-btn, .nav {
		width: 32px;
		height: 32px;
		flex: 0 0 32px;
		display: grid;
		place-items: center;
		border: 1px solid #546259;
		border-radius: 50%;
		background: transparent;
		color: #dce6df;
		cursor: pointer;
	}
	.close-btn { border: 0; }
	button:disabled { opacity: .3; cursor: default; }
	button:hover:not(:disabled) { background: #39463d; }
	button:focus-visible { outline: 2px solid #82e7f3; outline-offset: 3px; }
	.empty { text-align: center; color: #a5b5a9; padding: 32px 0; font-size: 14px; }
	.viewer { display: grid; grid-template-columns: 32px minmax(0, 1fr) 32px; gap: 10px; align-items: center; }
	.discard-entry { display: flex; flex-direction: column; align-items: center; min-width: 0; }
	.center-card {
		width: min(160px, 100%);
		aspect-ratio: 5 / 7;
		flex: 0 0 auto;
		overflow: hidden;
		border: 2px solid #7d8d81;
		border-radius: 6px;
		background: #303c33;
		box-shadow: 0 4px 14px rgba(0,0,0,.3);
	}
	.center-card.location { background: var(--town-color); display: grid; place-items: center; }
	.center-card img { width: 100%; height: 100%; object-fit: cover; display: block; }
	.town-name { padding: 10px; color: white; font-size: 16px; font-weight: 700; text-align: center; overflow-wrap: anywhere; text-shadow: 0 1px 3px #222; }
	figcaption { width: 100%; margin-top: 12px; text-align: center; }
	figcaption strong { display: block; font-size: 15px; overflow-wrap: anywhere; }
	.card-type { display: block; margin-top: 4px; color: #a5b5a9; font-size: 12px; }
	.meta { display: flex; justify-content: center; gap: 12px; flex-wrap: wrap; margin-top: 10px; font-size: 11px; color: #a5b5a9; }
	.counter { margin-top: 16px; text-align: center; font-size: 12px; color: #e6eee8; font-variant-numeric: tabular-nums; }
</style>
