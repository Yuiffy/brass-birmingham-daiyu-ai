<script lang="ts">
	import { choiceSet, currentPlayer } from '$lib/store';
	import { applyChoice } from '$lib/api';
	import { cardImage, isLocationCard, townCardColor } from '$lib/coords';
	import type { Card } from '$lib/types';

	$: cp = $currentPlayer;
	$: hand = cp?.hand ?? [];
	$: isCardChoice = $choiceSet?.kind === 'card';
	$: cardChoiceValues = isCardChoice
		? new Set(($choiceSet?.options ?? []).map(o => o.value as number))
		: new Set<number>();

	function selectCard(card: Card) {
		if (!isCardChoice || !cardChoiceValues.has(card.index)) return;
		applyChoice('card', card.index);
	}
</script>

<div class="hand-container">
	{#if hand.length === 0 && cp}
		<div class="hand-hidden">{cp.hand_size} cards (hidden)</div>
	{/if}
	<div class="hand-fan">
		{#each hand as card, i (card.index)}
			{@const selectable = isCardChoice && cardChoiceValues.has(card.index)}
			{@const isLoc = isLocationCard(card.card_type)}
			<button
				class="card-slot"
				class:selectable
				class:dimmed={isCardChoice && !selectable}
				on:click={() => selectCard(card)}
				style="--fan-offset: {(i - hand.length/2) * 8}deg; --z: {i}"
			>
				{#if isLoc}
					<div class="card-town" style="background: {townCardColor(card.label)}">
						<span class="town-name">{card.label}</span>
					</div>
				{:else}
					<img
						src={cardImage(card.label, card.card_type)}
						alt={card.label}
						class="card-img"
					/>
				{/if}
				<span class="card-label">{card.label}</span>
			</button>
		{/each}
	</div>
</div>

<style>
	.hand-container {
		padding: 8px 0;
		min-height: 130px;
	}
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
		position: relative;
		width: 80px;
		margin: 0 -8px;
		cursor: default;
		background: none;
		border: none;
		padding: 0;
		transform-origin: bottom center;
		transform: rotate(var(--fan-offset)) translateY(0);
		transition: transform 0.2s cubic-bezier(.34,1.56,.64,1), filter 0.2s;
		z-index: var(--z);
		filter: brightness(0.85);
	}
	.card-slot:hover {
		transform: rotate(var(--fan-offset)) translateY(-20px);
		z-index: 100;
		filter: brightness(1);
	}
	.card-slot.selectable {
		cursor: pointer;
		filter: brightness(1);
	}
	.card-slot.selectable:hover {
		transform: rotate(var(--fan-offset)) translateY(-28px) scale(1.08);
		filter: brightness(1.1) drop-shadow(0 4px 12px rgba(34,197,94,0.5));
	}
	.card-slot.dimmed {
		filter: brightness(0.45) saturate(0.3);
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
		.card-slot:hover {
			transform: translateY(-12px);
		}
		.card-slot.selectable:hover {
			transform: translateY(-18px) scale(1.03);
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
</style>
