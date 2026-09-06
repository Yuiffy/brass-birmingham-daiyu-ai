<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import { MapPin, Route, Store, X } from 'lucide-svelte';
	import type { BoardDetail } from '$lib/board-interaction';

	export let detail: BoardDetail;
	export let pinned = false;
	export let left = 8;
	export let top = 8;
	export let height = 120;
	const dispatch = createEventDispatcher<{ close: void }>();
</script>

<div class="board-details" class:pinned style:left="{left}px" style:top="{top}px" bind:clientHeight={height} role={pinned ? 'region' : 'tooltip'} aria-label="Board details">
	<header>
		{#if detail.image}<img src={detail.image} alt="" />
		{:else if detail.kind === 'road'}<Route size={24} aria-hidden="true" />
		{:else if detail.kind === 'building'}<MapPin size={24} aria-hidden="true" />
		{:else}<Store size={24} aria-hidden="true" />{/if}
		<div class="title"><strong>{detail.title}</strong><span>{detail.subtitle}</span></div>
		{#if pinned}<button on:click={() => dispatch('close')} title="Close details" aria-label="Close details"><X size={16} /></button>{/if}
	</header>
	<div class="ownership">
		{#if detail.owner}<span class="owner"><i style:background={detail.owner.color}></i>{detail.owner.name}</span>{/if}
		<span class="status">{detail.status}</span>
	</div>
	{#if detail.stats.length}
		<dl>{#each detail.stats as stat}<div><dt>{stat.label}</dt><dd>{stat.value}</dd></div>{/each}</dl>
	{/if}
</div>

<style>
	.board-details { position: absolute; z-index: 3; width: 252px; max-width: calc(100% - 16px); max-height: calc(100% - 56px); overflow-y: auto; padding: 8px; border: 1px solid #83958a; border-radius: 6px; background: #f5f8f5; color: #25362d; box-shadow: 0 3px 12px #0003; pointer-events: none; }
	.board-details.pinned { pointer-events: auto; }
	header { display: flex; align-items: center; gap: 8px; }
	header img { width: 30px; height: 30px; object-fit: contain; flex: 0 0 30px; }
	header :global(svg) { flex-shrink: 0; }
	.title { flex: 1; min-width: 0; }
	.title strong { display: block; font-size: 13px; line-height: 1.3; }
	.title span { display: block; font-size: 11px; line-height: 1.35; overflow-wrap: anywhere; color: #5f7266; }
	button { width: 26px; height: 26px; flex: 0 0 26px; display: grid; place-items: center; border: 0; background: transparent; color: #52675a; cursor: pointer; }
	button:focus-visible { outline: 2px solid #087f5b; border-radius: 4px; }
	.ownership { display: flex; justify-content: space-between; gap: 8px; margin-top: 5px; font-size: 11px; line-height: 16px; }
	.owner { display: inline-flex; align-items: center; gap: 5px; }
	.owner i { width: 8px; height: 8px; border-radius: 50%; }
	.status { color: #5f7266; margin-left: auto; }
	dl { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 2px 10px; margin-top: 5px; padding-top: 5px; border-top: 1px solid #d4dfd7; }
	dl div { display: flex; align-items: baseline; justify-content: space-between; gap: 4px; min-width: 0; font-size: 11px; line-height: 16px; }
	dt { color: #5f7266; }
	dd { font-weight: 700; text-align: right; }
</style>
