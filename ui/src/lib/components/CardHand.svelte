<script lang="ts">
	import { onDestroy } from 'svelte';
	import { cardPreviewTown, choiceSet, gameState } from '$lib/store';
	import { applyChoice } from '$lib/api';
	import { cardImage, isLocationCard, locationCardTown, townCardColor } from '$lib/coords';
	import type { Card } from '$lib/types';

	export let playerIndex: number | null = null;
	export let ownerLabel = '';
	export let interactionLocked = false;

	let hoveredCard: Card | null = null;
	let focusedCard: Card | null = null;

	$: cp = $gameState?.players.find(player => player.index === (playerIndex ?? $gameState?.current_player)) ?? null;
	$: hand = cp?.hand ?? [];
	// Card objects change when a hand is replaced, even if its indices are reused.
	$: if (hoveredCard && !hand.includes(hoveredCard)) hoveredCard = null;
	$: if (focusedCard && !hand.includes(focusedCard)) focusedCard = null;
	$: previewCard = hoveredCard ?? focusedCard;
	$: previewTown = previewCard ? locationCardTown(previewCard) : null;
	$: cardPreviewTown.set(previewTown);
	$: hasCardChoice = $choiceSet?.kind === 'card';
	$: isCardChoice = hasCardChoice && !interactionLocked;
	$: cardChoiceValues = isCardChoice
		? new Set(($choiceSet?.options ?? []).map(o => o.value as number))
		: new Set<number>();

	onDestroy(() => cardPreviewTown.set(null));

	function clearPreview() {
		hoveredCard = null;
		focusedCard = null;
	}

	function selectCard(card: Card) {
		if (interactionLocked || !isCardChoice || !cardChoiceValues.has(card.index)) return;
		applyChoice('card', card.index);
	}
</script>

<svelte:window on:blur={clearPreview} />

<div class="hand-container">
	{#if ownerLabel}<div class="hand-owner">{ownerLabel}</div>{/if}
	{#if hand.length === 0 && cp}
		<div class="hand-hidden">{cp.hand_size} cards (hidden)</div>
	{/if}
	<div class="hand-fan">
		{#each hand as card, i (card.index)}
			{@const selectable = isCardChoice && cardChoiceValues.has(card.index)}
			{@const isLoc = isLocationCard(card.card_type)}
			{@const town = locationCardTown(card)}
			<button
				type="button"
				class="card-slot"
				class:selectable
				class:previewing={previewCard === card && !!town}
				class:dimmed={isCardChoice && !selectable}
				aria-label={town ?? card.label}
				on:pointerenter={(event) => { if (event.pointerType !== 'touch') hoveredCard = card; }}
				on:pointerleave={() => hoveredCard = null}
				on:pointercancel={clearPreview}
				on:focus={(event) => { if (event.currentTarget.matches(':focus-visible')) focusedCard = card; }}
				on:blur={() => focusedCard = null}
				on:click={() => selectCard(card)}
				style="--fan-offset: {(i - hand.length/2) * 8}deg; --z: {i}"
			>
				<div class="card-face">
					{#if isLoc}
						<div class="card-town" style="background: {townCardColor(card.label)}">
							<span class="town-name">{town ?? card.label}</span>
						</div>
					{:else}
						<img
							src={cardImage(card.label, card.card_type)}
							alt={card.label}
							class="card-img"
						/>
					{/if}
					<span class="card-label">{town ?? card.label}</span>
				</div>
			</button>
		{/each}
	</div>
</div>

<style>
	.hand-container {
		position: relative;
		padding: 8px 0;
		min-height: 130px;
	}
	.hand-owner { position: absolute; top: 1px; left: 0; z-index: 120; color: #aeb6b1; font-size: 10px; font-weight: 700; }
	.hand-hidden {
		color: #64748b;
		text-align: center;
		padding: 40px 0;
		font-size: 14px;
	}
	.hand-fan {
		display: flex;
		justify-content: center;
		align-items: flex-end;
		gap: 0;
		perspective: 800px;
		padding: 10px 0 0;
	}
	.card-slot {
		--preview-color: #82e7f3;
		position: relative;
		width: 80px;
		margin: 0 -8px;
		cursor: default;
		background: none;
		border: none;
		padding: 0;
		transform-origin: bottom center;
		transform: rotate(var(--fan-offset));
		transition: filter 0.2s;
		z-index: var(--z);
		filter: brightness(0.85);
	}
	.card-face {
		transition: transform 0.2s cubic-bezier(.34,1.56,.64,1);
	}
	.card-slot:hover,
	.card-slot:focus-visible {
		z-index: 100;
		filter: brightness(1);
	}
	.card-slot:hover .card-face,
	.card-slot:focus-visible .card-face {
		transform: translateY(-20px);
	}
	.card-slot:focus-visible {
		outline: 2px solid var(--preview-color);
		outline-offset: 4px;
		border-radius: 6px;
	}
	.card-slot.previewing .card-town {
		border-color: var(--preview-color);
		box-shadow: 0 0 0 1px #152a2e, 0 0 14px rgba(130,231,243,0.5);
	}
	.card-slot.selectable {
		cursor: pointer;
		filter: brightness(1);
	}
	.card-slot.selectable:hover {
		filter: brightness(1.1) drop-shadow(0 4px 12px rgba(34,197,94,0.5));
	}
	.card-slot.selectable:hover .card-face,
	.card-slot.selectable:focus-visible .card-face {
		transform: translateY(-28px) scale(1.08);
	}
	.card-slot.dimmed {
		filter: brightness(0.45) saturate(0.3);
	}
	.card-slot.dimmed:hover,
	.card-slot.dimmed:focus-visible {
		filter: brightness(0.85);
	}
	.card-img {
		width: 80px;
		height: 112px;
		object-fit: cover;
		border-radius: 6px;
		box-shadow: 0 2px 8px rgba(0,0,0,0.4);
	}
	.card-town {
		width: 80px;
		height: 112px;
		border-radius: 6px;
		box-shadow: 0 2px 8px rgba(0,0,0,0.4);
		display: flex;
		align-items: center;
		justify-content: center;
		border: 2px solid rgba(255,255,255,0.25);
	}
	.town-name {
		color: #fff;
		font-size: 11px;
		font-weight: 700;
		text-align: center;
		text-shadow: 0 1px 4px rgba(0,0,0,0.7);
		padding: 4px;
		line-height: 1.2;
		word-break: break-word;
	}
	.card-label {
		display: block;
		text-align: center;
		font-size: 10px;
		color: #cbd5e1;
		margin-top: 2px;
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
		max-width: 80px;
	}
	@media (max-width: 760px) {
		.hand-container {
			overflow-x: auto;
			overscroll-behavior-x: contain;
			scrollbar-width: none;
		}
		.hand-container::-webkit-scrollbar { display: none; }
		.hand-fan {
			justify-content: flex-start;
			width: max-content;
			min-width: 100%;
			box-sizing: border-box;
			gap: 8px;
			padding-inline: 8px;
			scroll-snap-type: x proximity;
		}
		.card-slot {
			flex: 0 0 68px;
			width: 68px;
			margin-inline: 0;
			transform: translateY(0);
			scroll-snap-align: start;
		}
		.card-slot:hover .card-face,
		.card-slot:focus-visible .card-face {
			transform: translateY(-4px);
		}
		.card-slot.selectable:hover .card-face,
		.card-slot.selectable:focus-visible .card-face {
			transform: translateY(-4px);
		}
		.card-img,
		.card-town {
			width: 68px;
			height: 95px;
		}
		.card-label {
			max-width: 68px;
		}
		.town-name {
			font-size: 10px;
		}
	}
	@media (prefers-reduced-motion: reduce) {
		.card-slot,
		.card-face { transition: none; }
	}
</style>
