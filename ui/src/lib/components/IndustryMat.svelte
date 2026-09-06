<script lang="ts">
	import { createEventDispatcher, tick } from 'svelte';
	import { ArrowLeft, Expand, LoaderCircle, X } from 'lucide-svelte';
	import { modalDialog } from '$lib/dialog';
	import { choiceSet, currentPlayer, allIndustryData, gameState, currentAction, pendingDevelopments } from '$lib/store';
	import { applyChoice, cancelAction } from '$lib/api';
	import { INDUSTRY_COLORS } from '$lib/coords';
	import type { IndustryTile, IndustryLevelData, PendingDevelopment } from '$lib/types';

	const dispatch = createEventDispatcher();

	export let open = false;
	export let playerIndex: number | null = null;

	$: gs = $gameState;
	$: cs = $choiceSet;
	$: isIndustryChoice = cs?.kind === 'industry';
	$: isFreeDev = cs?.kind === 'free_development';
	$: isSecondIndustry = cs?.kind === 'second_industry';
	$: isSelecting = isIndustryChoice || isFreeDev || isSecondIndustry;
	$: cp = isSelecting
		? $currentPlayer
		: (
			playerIndex != null && gs
				? (gs.players.find(p => p.index === playerIndex) ?? $currentPlayer)
				: $currentPlayer
		  );
	$: actionType = $currentAction;
	const INDUSTRIES = ['Coal', 'Iron', 'Beer', 'Goods', 'Pottery', 'Cotton'];

	function optionToIndustryName(opt: { value: unknown; label: string }): string | null {
		if (typeof opt.value === 'string' && INDUSTRIES.includes(opt.value)) return opt.value;
		if (typeof opt.label === 'string' && INDUSTRIES.includes(opt.label)) return opt.label;
		// Defensive fallback if backend emits enum indices.
		if (typeof opt.value === 'number' && opt.value >= 0 && opt.value < INDUSTRIES.length) {
			return INDUSTRIES[opt.value];
		}
		return null;
	}

	$: availableIndustries = isSelecting
		? new Set(
			(cs?.options ?? [])
				.map(optionToIndustryName)
				.filter((v): v is string => v !== null)
		  )
		: new Set<string>();

	$: industryMat = cp?.industry_mat ?? [];
	$: allData = $allIndustryData;
	$: pendingDevs = $pendingDevelopments;
	$: projectedTiles = INDUSTRIES.map(name => projectedTileForMain(name, industryMat, pendingDevs, allData));

	let suppressAutoOpen = false;
	let lastChoiceKind: string | null = null;
	let expandedIndustry: string | null = null;
	$: expandedLevels = allData?.[expandedIndustry ?? ''] ?? [];
	$: expandedTile = industryMat.find(tile => tile.industry === expandedIndustry) ?? null;
	let selectingIndustry: string | null = null;
	let selectionVersion = 0;
	let selectionError = '';
	let dialog: HTMLDialogElement;
	let expandedBack: HTMLButtonElement;

	$: if (isSelecting && !open && !suppressAutoOpen) {
		open = true;
	}
	$: {
		const currentKind = cs?.kind ?? null;
		if (currentKind !== lastChoiceKind) {
			suppressAutoOpen = false;
			lastChoiceKind = currentKind;
		}
		if (!isSelecting) {
			suppressAutoOpen = false;
		}
	}

	async function close() {
		selectionVersion++;
		suppressAutoOpen = true;
		selectingIndustry = null;
		selectionError = '';
		open = false;
		expandedIndustry = null;
		dispatch('close');

		// If this was an in-action selection modal, closing means cancel action.
		if (isSelecting) {
			await cancelAction();
		}
	}

	async function selectIndustry(name: string) {
		if (!isSelecting || !cs || !availableIndustries.has(name) || selectingIndustry) return;
		const version = ++selectionVersion;
		selectingIndustry = name;
		selectionError = '';
		const data = await applyChoice(cs.kind, name);
		// A closed or replaced dialog must not change a later action's UI.
		if (version !== selectionVersion) return;
		selectingIndustry = null;
		if (!data) {
			selectionError = 'Industry selection failed.';
			return;
		}
		const nextKind = data.state?.choice_set?.kind;
		if (nextKind === 'industry' || nextKind === 'free_development' || nextKind === 'second_industry') {
			suppressAutoOpen = false;
			open = true;
		} else {
			suppressAutoOpen = true;
			open = false;
			expandedIndustry = null;
			dispatch('close');
		}
	}

	async function expand(name: string) {
		if (isSelecting) return;
		expandedIndustry = name;
		await tick();
		expandedBack?.focus();
	}

	async function collapseExpanded() {
		const industry = expandedIndustry;
		expandedIndustry = null;
		await tick();
		dialog?.querySelector<HTMLButtonElement>(`[data-industry="${industry}"]`)?.focus();
	}

	function dismiss() {
		if (expandedIndustry) void collapseExpanded();
		else void close();
	}

	function levelToRoman(level: number): string {
		const numerals = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII'];
		return numerals[level - 1] ?? String(level);
	}

	function iconPath(industry: string): string {
		return `/assets/buildings/icons/${industry.toLowerCase()}.svg`;
	}

	function projectedTileForMain(
		name: string,
		tiles: IndustryTile[],
		developments: PendingDevelopment[],
		data: Record<string, IndustryLevelData[]> | null
	): IndustryTile | null {
		const tile = tiles.find(tile => tile.industry === name);
		if (!tile) return null;
		const pending = developments.filter(development => development.industry === name).length;
		let remaining = tile.tiles_remaining - pending;
		if (remaining > 0) {
			return { ...tile, tiles_remaining: remaining };
		}
		let nextLevel = tile.level + 1;
		let next = data?.[name]?.find(level => level.level === nextLevel);
		while (next && remaining + next.num_tiles <= 0) {
			remaining += next.num_tiles;
			nextLevel++;
			next = data?.[name]?.find(level => level.level === nextLevel);
		}
		if (!next) return { ...tile, tiles_remaining: 0, exhausted: true };
		return {
			industry: name,
			level: next.level,
			tiles_remaining: remaining + next.num_tiles,
			money_cost: next.money_cost,
			coal_cost: next.coal_cost,
			iron_cost: next.iron_cost,
			beer_needed: next.beer_needed,
			vp_on_flip: next.vp_on_flip,
			road_vp: next.road_vp,
			resource_amt: next.resource_amt,
			income: next.income,
			removed_after_phase1: next.removed_after_phase1,
			can_develop: next.can_develop,
			exhausted: false,
		};
	}

	function tilesAtLevel(tile: IndustryTile | null, level: IndustryLevelData): number {
		if (!tile || tile.exhausted) return 0;
		if (level.level < tile.level) return 0;
		return level.level === tile.level ? tile.tiles_remaining : level.num_tiles;
	}

	function resourceColor(type: 'coal' | 'iron'): string {
		return type === 'coal' ? '#1a1a1a' : '#f97316';
	}

	function industryResourceType(industry: string): 'coal' | 'iron' | 'beer' | null {
		if (industry === 'Coal') return 'coal';
		if (industry === 'Iron') return 'iron';
		if (industry === 'Beer') return 'beer';
		return null;
	}
</script>

{#if open}
<dialog class="overlay" bind:this={dialog} use:modalDialog={{ onClose: dismiss }} aria-label="Industry Mat">

{#if expandedIndustry}
	<!-- Expanded single-industry view -->
	<div class="expanded-view">
		<div class="expanded-header">
			<button class="back-btn" bind:this={expandedBack} on:click={collapseExpanded} title="Back to Industry Mat" aria-label="Back to Industry Mat"><ArrowLeft size={20} /></button>
			<h2 style="color: {INDUSTRY_COLORS[expandedIndustry] ?? '#ccc'}">{expandedIndustry} — All Levels</h2>
		</div>
		<div class="expanded-row">
			{#each expandedLevels as lvl}
				{@const available = isSelecting && availableIndustries.has(expandedIndustry)}
				{@const disabled = isSelecting && !available}
				{@const currentTiles = tilesAtLevel(expandedTile, lvl)}
				{@const maxTiles = lvl.num_tiles}
				<div
					class="card-wrapper expanded-card"
					class:disabled={disabled || currentTiles === 0}
				>
					<div class="card" class:depleted-level={currentTiles === 0} style="--player-color: {cp?.color ?? '#555'}; --ind-color: {INDUSTRY_COLORS[expandedIndustry] ?? '#666'}">
						<!-- Price -->
						<div class="price-circle">£{lvl.money_cost}</div>

						<!-- Resource costs -->
						<div class="resource-costs">
							{#each Array(lvl.coal_cost) as _}
								<div class="res-cube coal-cube" title="Coal"></div>
							{/each}
							{#each Array(lvl.iron_cost) as _}
								<div class="res-cube iron-cube" title="Iron"></div>
							{/each}
						</div>

						<!-- Canal-only indicator -->
						{#if lvl.removed_after_phase1}
							<div class="canal-indicator" title="This tile can only be built in Canal phase, will be removed during Railroad phase.">
								<svg viewBox="0 0 16 16" width="20" height="20"><path d="M8 1 L15 5 V11 L8 15 L1 11 V5 Z" fill="#3b82f6" stroke="#60a5fa" stroke-width="0.5"/><text x="8" y="10" text-anchor="middle" fill="white" font-size="7" font-weight="bold">C</text></svg>
							</div>
						{/if}

						<!-- Colored inner box with icon -->
						<div class="inner-box">
							<!-- Beer needed indicators (top-right of inner box) -->
							{#if lvl.beer_needed > 0}
								<div class="beer-indicators">
									{#each Array(lvl.beer_needed) as _}
										<img src="/assets/indicators/barrel.svg" alt="Beer" class="barrel-icon" />
									{/each}
								</div>
							{/if}

							<!-- Can't develop indicator -->
							{#if !lvl.can_develop}
								<div class="no-develop-indicator" class:with-beer={lvl.beer_needed > 0}>
									<img src="/assets/indicators/crossed_bulb.svg" alt="Cannot develop" class="crossed-bulb-icon" />
								</div>
							{/if}

							<img src={iconPath(expandedIndustry)} alt={expandedIndustry} class="industry-icon" />

							<!-- Resource production (bottom-left of inner box) -->
							{#if lvl.resource_amt > 0 && industryResourceType(expandedIndustry ?? '') === 'beer'}
								<div class="production-indicators">
									{#each Array(lvl.resource_amt) as _}
										<img src="/assets/indicators/barrel.svg" alt="Beer" class="prod-barrel" />
									{/each}
								</div>
							{:else if lvl.resource_amt > 0 && industryResourceType(expandedIndustry ?? '') === 'coal'}
								<div class="production-indicators">
									{#each Array(lvl.resource_amt) as _}
										<div class="prod-cube coal-cube" title="Coal"></div>
									{/each}
								</div>
							{:else if lvl.resource_amt > 0 && industryResourceType(expandedIndustry ?? '') === 'iron'}
								<div class="production-indicators">
									{#each Array(lvl.resource_amt) as _}
										<div class="prod-cube iron-cube" title="Iron"></div>
									{/each}
								</div>
							{/if}
						</div>

						<!-- Right column: VP, Income, Road VP -->
						<div class="right-stats">
							<div class="vp-hex" title="{lvl.vp_on_flip} victory points on flip">
								<svg viewBox="0 0 28 32" width="32" height="36"><polygon points="14,0 28,8 28,24 14,32 0,24 0,8" fill="#111" stroke="#d4a020" stroke-width="2"/><text x="14" y="20" text-anchor="middle" fill="#d4a020" font-size="13" font-weight="bold">{lvl.vp_on_flip}</text></svg>
							</div>

							{#if lvl.income > 0}
								<div class="income-arrow" title="+{lvl.income} income level increase">
									<svg viewBox="0 0 24 30" width="28" height="34">
										<polygon points="12,0 24,10 18,10 18,30 6,30 6,10 0,10" fill="#e8dbb0" stroke="#c0a860" stroke-width="1"/>
										<text x="12" y="23" text-anchor="middle" fill="#333" font-size="12" font-weight="bold">{lvl.income}</text>
									</svg>
								</div>
							{:else if lvl.income < 0}
								<div class="income-arrow neg" title="{lvl.income} income level">
									<svg viewBox="0 0 24 30" width="28" height="34">
										<polygon points="12,30 24,20 18,20 18,0 6,0 6,20 0,20" fill="#e8b0b0" stroke="#c06060" stroke-width="1"/>
										<text x="12" y="15" text-anchor="middle" fill="#333" font-size="12" font-weight="bold">{Math.abs(lvl.income)}</text>
									</svg>
								</div>
							{/if}

							{#if lvl.road_vp > 0}
								<div class="road-vp" title="{lvl.road_vp} victory points will be attributed to any player who builds a link to the location where this tile is present">
									<div class="road-vp-hexes">
										{#each Array(lvl.road_vp) as _}
											<svg viewBox="0 0 20 24" width="22" height="26"><polygon points="10,0 20,6 20,18 10,24 0,18 0,6" fill="#111" stroke="#d4a020" stroke-width="1.5"/><line x1="4" y1="12" x2="16" y2="12" stroke="#d4a020" stroke-width="2"/></svg>
										{/each}
									</div>
								</div>
							{/if}
						</div>

						<!-- Bottom label -->
						<div class="card-label">
							<span class="ind-name">{expandedIndustry}</span>
							<span class="level-numeral">{levelToRoman(lvl.level)}</span>
						</div>

						<!-- Tile count badge -->
						<div class="tile-count-badge">{currentTiles}/{maxTiles}</div>
					</div>
				</div>
			{/each}
		</div>
	</div>

{:else}
	<!-- Main 3x2 grid view -->
	<div class="modal" role="dialog" aria-modal="true">
		<div class="modal-header">
			<h2>
				Industry Mat
				{#if cp}<span class="mat-player" style="color:{cp.color}"> — {cp.name}</span>{/if}
			</h2>
			{#if isSelecting}
				<div class="pick-prompt">
					{#if isSecondIndustry}
						Select second industry to develop
					{:else if isFreeDev}
						Pick an industry to develop for free
					{:else if actionType === 'Develop' || actionType === 'DevelopDouble'}
						Select an industry to develop
					{:else}
						Select an industry to build
					{/if}
				</div>
			{/if}
			<button class="close-btn" on:click={close} title={isSelecting ? 'Cancel action' : 'Close Industry Mat'} aria-label={isSelecting ? 'Cancel action' : 'Close Industry Mat'}><X size={20} /></button>
		</div>
		{#if selectingIndustry}<div class="selection-status" role="status"><LoaderCircle size={16} class="spin" /> {selectingIndustry}</div>{/if}
		{#if selectionError}<div class="selection-error" role="alert">{selectionError}</div>{/if}
		<div class="industry-grid">
			{#each INDUSTRIES as name, i}
				{@const tile = projectedTiles[i]}
				{@const available = isSelecting && availableIndustries.has(name) && !selectingIndustry}
				{@const disabled = isSelecting && !available}
				{@const exhausted = tile?.exhausted ?? true}
				{@const tilesLeft = tile?.tiles_remaining ?? 0}
				<button
					class="card-wrapper"
					data-industry={name}
					aria-label={isSelecting ? `Select ${name}` : `View ${name} industry levels`}
					disabled={isSelecting && (disabled || exhausted)}
					class:disabled={disabled || exhausted}
					class:selectable={available && !exhausted}
					class:exhausted
					on:click={() => {
						if (available && !exhausted) selectIndustry(name);
						else if (!isSelecting) expand(name);
					}}
				>
					{#if tile && !exhausted}
						<div
							class="card"
							class:hoverable={!isSelecting}
							style="--player-color: {cp?.color ?? '#555'}; --ind-color: {INDUSTRY_COLORS[name] ?? '#666'}; --stack-count: {tilesLeft}"
						>
							<!-- Price circle (top-left, outside color box) -->
							<div class="price-circle">£{tile.money_cost}</div>

							<!-- Resource costs (under price) -->
							<div class="resource-costs">
								{#each Array(tile.coal_cost) as _}
									<div class="res-cube coal-cube" title="Coal"></div>
								{/each}
								{#each Array(tile.iron_cost) as _}
									<div class="res-cube iron-cube" title="Iron"></div>
								{/each}
							</div>

							<!-- Canal-only indicator (bottom-left, outside color box) -->
							{#if tile.removed_after_phase1}
								<div class="canal-indicator" title="This tile can only be built in Canal phase, will be removed during Railroad phase.">
									<svg viewBox="0 0 16 16" width="20" height="20"><path d="M8 1 L15 5 V11 L8 15 L1 11 V5 Z" fill="#3b82f6" stroke="#60a5fa" stroke-width="0.5"/><text x="8" y="10" text-anchor="middle" fill="white" font-size="7" font-weight="bold">C</text></svg>
								</div>
							{/if}

							<!-- Colored inner box with icon -->
							<div class="inner-box">
								<!-- Beer needed (top-right inside inner box) -->
								{#if tile.beer_needed > 0}
									<div class="beer-indicators">
										{#each Array(tile.beer_needed) as _}
											<img src="/assets/indicators/barrel.svg" alt="Beer" class="barrel-icon" />
										{/each}
									</div>
								{/if}

								<!-- Can't develop indicator -->
								{#if !tile.can_develop}
									<div class="no-develop-indicator" class:with-beer={tile.beer_needed > 0}>
										<img src="/assets/indicators/crossed_bulb.svg" alt="Cannot develop" class="crossed-bulb-icon" />
									</div>
								{/if}

								<img src={iconPath(name)} alt={name} class="industry-icon" />

								<!-- Resource production (bottom-left inside inner box) -->
								{#if tile.resource_amt > 0 && industryResourceType(name) === 'beer'}
									<div class="production-indicators">
										{#each Array(tile.resource_amt) as _}
											<img src="/assets/indicators/barrel.svg" alt="Beer" class="prod-barrel" />
										{/each}
									</div>
								{:else if tile.resource_amt > 0 && industryResourceType(name) === 'coal'}
									<div class="production-indicators">
										{#each Array(tile.resource_amt) as _}
											<div class="prod-cube coal-cube" title="Coal"></div>
										{/each}
									</div>
								{:else if tile.resource_amt > 0 && industryResourceType(name) === 'iron'}
									<div class="production-indicators">
										{#each Array(tile.resource_amt) as _}
											<div class="prod-cube iron-cube" title="Iron"></div>
										{/each}
									</div>
								{/if}
							</div>

							<!-- Right stats column (outside color box) -->
							<div class="right-stats">
								<div class="vp-hex" title="{tile.vp_on_flip} victory points on flip">
									<svg viewBox="0 0 28 32" width="32" height="36"><polygon points="14,0 28,8 28,24 14,32 0,24 0,8" fill="#111" stroke="#d4a020" stroke-width="2"/><text x="14" y="20" text-anchor="middle" fill="#d4a020" font-size="13" font-weight="bold">{tile.vp_on_flip}</text></svg>
								</div>

								{#if tile.income > 0}
									<div class="income-arrow" title="+{tile.income} income level increase">
										<svg viewBox="0 0 24 30" width="28" height="34">
											<polygon points="12,0 24,10 18,10 18,30 6,30 6,10 0,10" fill="#e8dbb0" stroke="#c0a860" stroke-width="1"/>
											<text x="12" y="23" text-anchor="middle" fill="#333" font-size="12" font-weight="bold">{tile.income}</text>
										</svg>
									</div>
								{:else if tile.income < 0}
									<div class="income-arrow neg" title="{tile.income} income level">
										<svg viewBox="0 0 24 30" width="28" height="34">
											<polygon points="12,30 24,20 18,20 18,0 6,0 6,20 0,20" fill="#e8b0b0" stroke="#c06060" stroke-width="1"/>
											<text x="12" y="15" text-anchor="middle" fill="#333" font-size="12" font-weight="bold">{Math.abs(tile.income)}</text>
										</svg>
									</div>
								{/if}

								{#if tile.road_vp > 0}
									<div class="road-vp" title="{tile.road_vp} victory points will be attributed to any player who builds a link to the location where this tile is present">
										<div class="road-vp-hexes">
											{#each Array(tile.road_vp) as _}
												<svg viewBox="0 0 20 24" width="22" height="26"><polygon points="10,0 20,6 20,18 10,24 0,18 0,6" fill="#111" stroke="#d4a020" stroke-width="1.5"/><line x1="4" y1="12" x2="16" y2="12" stroke="#d4a020" stroke-width="2"/></svg>
											{/each}
										</div>
									</div>
								{/if}
							</div>

							<!-- Bottom label -->
							<div class="card-label">
								<span class="ind-name">{name}</span>
								<span class="level-numeral">{levelToRoman(tile.level)}</span>
							</div>

							<!-- Stack / tile count (shown on hover via animation) -->
							{#if tilesLeft > 1}
								<div class="stack-shadows">
									{#each Array(Math.min(tilesLeft - 1, 3)) as _, i}
										<div class="stack-shadow" style="--i: {i + 1}"></div>
									{/each}
								</div>
							{/if}

							<div class="tile-count-badge">&times;{tilesLeft}</div>
						</div>

						{#if !isSelecting}
							<span class="expand-btn" aria-hidden="true"><Expand size={14} /></span>
						{/if}
					{:else}
						<div class="card exhausted-card">
							<div class="inner-box exhausted-inner">
								<img src={iconPath(name)} alt={name} class="industry-icon faded" />
							</div>
							<div class="card-label">
								<span class="ind-name">{name}</span>
								<span class="level-numeral">—</span>
							</div>
						</div>
					{/if}
				</button>
			{/each}
		</div>
	</div>
	{/if}
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
		padding: 16px;
		border: 0;
		background: transparent;
		z-index: 1000;
		display: flex;
		align-items: center;
		justify-content: center;
	}
	.overlay::backdrop { background: rgba(0,0,0,0.7); backdrop-filter: blur(2px); }

	.modal {
		position: relative;
		background: #202522;
		border-radius: 8px;
		padding: 28px;
		max-width: 1000px;
		width: 100%;
		max-height: 100%;
		overflow-y: auto;
		overscroll-behavior: contain;
		box-shadow: 0 20px 60px rgba(0,0,0,0.5);
		border: 1px solid #333;
	}

	.modal-header {
		position: sticky;
		top: -28px;
		background: #202522;
		z-index: 5;
		display: flex;
		align-items: center;
		justify-content: space-between;
		margin-bottom: 16px;
		flex-wrap: wrap;
		gap: 8px;
	}

	.modal-header h2 {
		margin: 0;
		font-size: 20px;
		color: #eee;
	}

	.mat-player { font-weight: 600; }

	.pick-prompt {
		background: #303730;
		color: #fbbf24;
		padding: 6px 14px;
		border-radius: 8px;
		font-size: 0.85rem;
		font-weight: 600;
	}

	.close-btn,
	.back-btn {
		width: 32px;
		height: 32px;
		flex: 0 0 32px;
		display: grid;
		place-items: center;
		background: none;
		border: none;
		color: #888;
		font-size: 1.8rem;
		cursor: pointer;
		padding: 0 4px;
		line-height: 1;
	}
	.close-btn:hover { color: #fff; }
	.close-btn:focus-visible,
	.back-btn:focus-visible,
	.card-wrapper:focus-visible { outline: 2px solid #82e7f3; outline-offset: 3px; }
	.selection-status { display: flex; align-items: center; gap: 8px; color: #dce7de; font-size: 13px; margin-bottom: 12px; }
	.selection-error { color: #f5aaa1; font-size: 13px; margin-bottom: 12px; }
	.selection-status :global(.spin) { animation: spin 1s linear infinite; }
	@keyframes spin { to { transform: rotate(360deg); } }

	/* === Industry Grid === */
	.industry-grid {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(min(100%, 240px), 1fr));
		gap: 24px;
	}

	.card-wrapper {
		min-width: 0;
		background: none;
		border: none;
		padding: 0;
		cursor: pointer;
		position: relative;
		transition: transform 0.15s ease;
	}
	.card-wrapper:not(.disabled):not(.exhausted):hover {
		transform: translateY(-6px);
	}
	.card-wrapper.disabled {
		cursor: not-allowed;
		opacity: 0.35;
		filter: grayscale(0.8);
	}

	.card-wrapper.exhausted {
		cursor: default;
		opacity: 0.3;
		filter: grayscale(1);
	}
	.card.depleted-level {
		filter: grayscale(1);
		opacity: 0.45;
	}

	/* === Card === */
	.card {
		position: relative;
		background: #303631;
		border-radius: 8px;
		padding: 8px;
		display: grid;
		grid-template-columns: 42px minmax(80px, 1fr) 42px;
		grid-template-rows: auto 1fr auto;
		gap: 3px;
		min-height: 220px;
		border: 2px solid #444;
		transition: border-color 0.2s, box-shadow 0.2s;
	}
	.card-wrapper.selectable .card {
		border-color: #fbbf24;
		box-shadow: 0 0 12px rgba(251,191,36,0.4);
	}
	.card-wrapper:not(.disabled):not(.exhausted):hover .card {
		border-color: #888;
	}

	.exhausted-card {
		grid-template-columns: 1fr;
		min-height: 170px;
		justify-items: center;
	}

	/* === Price (top-left) === */
	.price-circle {
		grid-column: 1;
		grid-row: 1;
		width: 38px;
		height: 38px;
		border-radius: 50%;
		background: #e8dbb0;
		color: #333;
		font-size: 0.85rem;
		font-weight: 700;
		display: flex;
		align-items: center;
		justify-content: center;
		border: 1.5px solid #c0a860;
		z-index: 2;
	}

	/* === Resource costs (under price) === */
	.resource-costs {
		grid-column: 1;
		grid-row: 2;
		display: flex;
		flex-direction: column;
		gap: 3px;
		align-items: center;
		padding-top: 4px;
	}

	.res-cube {
		width: 16px;
		height: 16px;
		border-radius: 3px;
	}
	.coal-cube { background: #1a1a1a; border: 1px solid #555; }
	.iron-cube { background: #f97316; border: 1px solid #c05a00; }

	/* === Canal indicator (bottom-left) === */
	.canal-indicator {
		grid-column: 1;
		grid-row: 3;
		display: flex;
		align-items: flex-end;
		justify-content: center;
		padding-bottom: 2px;
	}

	/* === Inner colored box (center) === */
	.inner-box {
		grid-column: 2;
		grid-row: 1 / 3;
		background: var(--player-color, #555);
		border-radius: 6px;
		display: flex;
		align-items: center;
		justify-content: center;
		position: relative;
		min-height: 130px;
		overflow: hidden;
	}

	.exhausted-inner {
		background: #333;
	}

	.industry-icon {
		width: 80px;
		height: 80px;
		object-fit: contain;
		filter: drop-shadow(0 2px 4px rgba(0,0,0,0.4));
	}
	.industry-icon.faded { opacity: 0.3; }

	/* Beer needed (top-right of inner box) */
	.beer-indicators {
		position: absolute;
		top: 6px;
		right: 6px;
		display: flex;
		gap: 3px;
	}
	.barrel-icon {
		width: 26px;
		height: 26px;
		filter: drop-shadow(0 1px 1px rgba(0,0,0,0.5));
	}

	/* Can't develop */
	.no-develop-indicator {
		position: absolute;
		top: 6px;
		right: 6px;
	}
	.no-develop-indicator.with-beer {
		top: 36px;
	}
	.crossed-bulb-icon {
		width: 26px;
		height: 26px;
		filter: drop-shadow(0 1px 1px rgba(0,0,0,0.5));
	}

	/* Production indicators (bottom-left of inner box) */
	.production-indicators {
		position: absolute;
		bottom: 6px;
		left: 6px;
		display: flex;
		gap: 3px;
		flex-wrap: wrap;
		max-width: 85px;
	}
	.prod-cube {
		width: 16px;
		height: 16px;
		border-radius: 3px;
	}
	.prod-barrel {
		width: 22px;
		height: 22px;
		filter: drop-shadow(0 1px 1px rgba(0,0,0,0.5));
	}

	/* === Right stats column === */
	.right-stats {
		grid-column: 3;
		grid-row: 1 / 3;
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 4px;
		padding-top: 2px;
	}

	.vp-hex, .income-arrow, .road-vp {
		display: flex;
		align-items: center;
		justify-content: center;
	}

	.road-vp-hexes {
		display: flex;
		gap: 1px;
	}

	/* === Bottom label === */
	.card-label {
		grid-column: 1 / -1;
		grid-row: 3;
		text-align: center;
		padding: 2px 0;
		display: flex;
		justify-content: center;
		gap: 6px;
		align-items: baseline;
	}

	.ind-name {
		color: #ccc;
		font-size: 0.9rem;
		font-weight: 600;
	}
	.level-numeral {
		color: #fbbf24;
		font-size: 0.95rem;
		font-weight: 700;
	}

	/* === Stack shadows (hover animation) === */
	.stack-shadows {
		position: absolute;
		inset: 0;
		pointer-events: none;
		z-index: -1;
	}
	.stack-shadow {
		position: absolute;
		inset: 0;
		background: #303631;
		border-radius: 8px;
		border: 2px solid #444;
		transform: translateY(calc(var(--i) * 4px));
		opacity: 0;
		transition: opacity 0.2s ease, transform 0.2s ease;
	}
	.card-wrapper:hover .stack-shadow {
		opacity: calc(0.6 - var(--i) * 0.15);
		transform: translateY(calc(var(--i) * 6px));
	}

	/* === Tile count badge === */
	.tile-count-badge {
		position: absolute;
		top: -6px;
		right: -6px;
		background: #e74c3c;
		color: white;
		font-size: 0.65rem;
		font-weight: 700;
		padding: 2px 5px;
		border-radius: 10px;
		z-index: 3;
		border: 1px solid #c0392b;
	}

	/* === Expand button === */
	.expand-btn {
		position: absolute;
		bottom: -4px;
		right: -4px;
		width: 22px;
		height: 22px;
		border-radius: 50%;
		background: #444;
		border: 1px solid #666;
		color: #ccc;
		font-size: 0.8rem;
		display: flex;
		align-items: center;
		justify-content: center;
		cursor: pointer;
		z-index: 3;
		transition: background 0.15s;
	}
	.expand-btn:hover {
		background: #666;
		color: #fff;
	}

	/* === Expanded view === */
	.expanded-view {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 20px;
		width: min(100%, 1100px);
		max-height: 100%;
		overflow-y: auto;
		overscroll-behavior: contain;
		background: #202522;
		border-radius: 8px;
		padding: 20px;
	}

	.expanded-header {
		position: sticky;
		top: -20px;
		z-index: 5;
		background: #202522;
		width: 100%;
		display: flex;
		align-items: center;
		gap: 10px;
	}
	.expanded-header h2 {
		margin: 0 0 4px 0;
		font-size: 18px;
	}

	.expanded-row {
		width: 100%;
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(min(100%, 220px), 1fr));
		gap: 16px;
	}
	.expanded-card {
		cursor: default;
	}
	.expanded-card .card {
		min-height: 260px;
	}
	@media (max-width: 600px) {
		.overlay { padding: 12px; }
		.modal { padding: 16px; }
		.modal-header { top: -16px; }
		.modal-header h2 { font-size: 18px; flex: 1; }
		.pick-prompt { order: 3; width: 100%; }
		.industry-grid { gap: 18px; }
		.expanded-view { padding: 16px; }
		.expanded-header { top: -16px; }
	}
	@media (prefers-reduced-motion: reduce) {
		.card-wrapper, .card, .stack-shadow { transition: none; }
		.selection-status :global(.spin) { animation: none; }
	}
</style>
