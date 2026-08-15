<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import { Dices } from 'lucide-svelte';
	import { gameState, turnPhase, actionsAvailable, choiceSet, logs, moneyAtTurnStart, pendingDevelopments, actionBudgetAtTurnStart } from '$lib/store';
	import { INDUSTRY_COLORS } from '$lib/coords';
	import { startTurn, selectAction, applyChoice, confirmAction, cancelAction, endTurn, undoLastAction } from '$lib/api';

	const dispatch = createEventDispatcher();

	$: gs = $gameState;
	$: cs = $choiceSet;
	$: phase = $turnPhase;
	$: actions = $actionsAvailable;
	$: moneySnap = $moneyAtTurnStart;
	$: pendingDevs = $pendingDevelopments;
	$: actionBudget = $actionBudgetAtTurnStart;
	let roundToast: string | null = null;
	let toastTimer: ReturnType<typeof setTimeout> | null = null;
	let prevRound = -1;
	let prevEra = '';
let seedCopied = false;
let seedCopyTimer: ReturnType<typeof setTimeout> | null = null;

	$: orderedPlayers = gs ? gs.turn_order.map(idx => gs!.players.find(p => p.index === idx)!).filter(Boolean) : [];

	const LEVEL_ROMAN = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII'];
	function levelToRoman(level: number): string {
		return LEVEL_ROMAN[level - 1] ?? String(level);
	}
	function devIconPath(industry: string): string {
		return `/assets/buildings/icons/${industry.toLowerCase()}.svg`;
	}

	function roundsPerEra(numPlayers: number): number {
		if (numPlayers === 2) return 10;
		if (numPlayers === 3) return 9;
		return 8;
	}

	function currentActionNumber(): number {
		if (!gs) return 1;
		const n = actionBudget - gs.actions_remaining + 1;
		return Math.min(Math.max(1, n), actionBudget);
	}

	function moneyDelta(pIdx: number, currentMoney: number): string {
		if (moneySnap[pIdx] == null) return '';
		const d = moneySnap[pIdx] - currentMoney;
		if (d > 0) return ` (-£${d})`;
		if (d < 0) return ` (+£${-d})`;
		return '';
	}

	const ACTION_LABELS: Record<string, string> = {
		BuildBuilding: 'Build', BuildRailroad: 'Network', BuildDoubleRailroad: 'Double Rail',
		Develop: 'Develop', DevelopDouble: 'Develop x2', Sell: 'Sell',
		Loan: 'Loan', Scout: 'Scout', Pass: 'Pass'
	};
	const ACTION_MARK: Record<string, string> = {
		BuildBuilding: 'B', BuildRailroad: 'N', BuildDoubleRailroad: 'N2',
		Develop: 'D', DevelopDouble: 'D2', Sell: 'S',
		Loan: 'L', Scout: 'SC', Pass: 'P'
	};
	const CHOICE_TITLES: Record<string, string> = {
		industry: 'Choose Industry', card: 'Choose Card', build_location: 'Choose Location',
		road: 'Choose Road', second_road: 'Choose 2nd Road', coal_source: 'Coal Source',
		iron_source: 'Iron Source', beer_source: 'Beer Source', action_beer_source: 'Beer',
		sell_target: 'Sell Target', free_development: 'Free Develop', second_industry: 'Second Industry',
		network_mode: 'Network Mode', confirm: 'Confirm Action'
	};

	const MAP_KINDS = new Set(['build_location', 'road', 'second_road', 'sell_target', 'beer_source', 'action_beer_source']);
	const MODAL_KINDS = new Set(['industry', 'card', 'free_development', 'second_industry']);

	function onChoice(opt: any) {
		if (!cs) return;
		if (cs.kind === 'confirm') confirmAction();
		else applyChoice(cs.kind, opt.value);
	}

	async function copySeed() {
		if (!gs) return;
		try {
			await navigator.clipboard.writeText(String(gs.seed));
			seedCopied = true;
			if (seedCopyTimer) clearTimeout(seedCopyTimer);
			seedCopyTimer = setTimeout(() => {
				seedCopied = false;
				seedCopyTimer = null;
			}, 1300);
		} catch {
			// Non-critical if clipboard API unavailable.
		}
	}

	$: if (gs) {
		if (prevRound === -1) {
			prevRound = gs.round_in_phase;
			prevEra = gs.era;
		} else {
			const roundChanged = gs.round_in_phase !== prevRound;
			const eraChanged = gs.era !== prevEra;
			if (roundChanged || eraChanged) {
				if (toastTimer) clearTimeout(toastTimer);
				if (eraChanged && gs.era === 'Railroad') {
					roundToast = 'Railroad Era begins';
				} else if (roundChanged) {
					roundToast = `New Round ${gs.round_in_phase + 1}/${roundsPerEra(gs.players.length)}`;
				}
				toastTimer = setTimeout(() => {
					roundToast = null;
					toastTimer = null;
				}, 1800);
			}
			prevRound = gs.round_in_phase;
			prevEra = gs.era;
		}
	}
</script>

{#if gs}
<div class="sidebar">
	{#if roundToast}
		<div class="round-toast">{roundToast}</div>
	{/if}
	<section class="panel phase-panel">
		<div class="phase-era">{gs.era} Era</div>
		<div class="phase-sub">
			Turn {gs.turn_count + 1} &middot; Round {gs.round_in_phase + 1}/{roundsPerEra(gs.players.length)}
			&middot; Action {currentActionNumber()}/{actionBudget}
		</div>
		<button class="seed-chip" on:click={copySeed} title="Copy seed to clipboard">
			<Dices size={13} aria-hidden="true" /> Seed {gs.seed}
		</button>
		{#if seedCopied}
			<div class="seed-copied">Copied seed</div>
		{/if}
	</section>

	<section class="panel">
		<h3>Players (Turn Order)</h3>
		{#each orderedPlayers as p, pos}
			<div class="player-row" class:active={p.index === gs.current_player}>
				<span class="turn-num">{pos + 1}</span>
				<span class="dot" style="background:{p.color}"></span>
				<span class="pname">{p.name}</span>
				<span class="pstats">
					£{p.money}<span class="money-delta">{moneyDelta(p.index, p.money)}</span>
					&middot; VP:{p.victory_points} &middot; Inc:{p.income_amount}
				</span>
				<button class="mini-btn" on:click={() => dispatch('openMat', { playerIndex: p.index })}>Mat</button>
				<button class="mini-btn" on:click={() => dispatch('openDiscard', { playerIndex: p.index })}>Discard</button>
			</div>
		{/each}
	</section>

	<section class="panel">
		<h3>Market</h3>
		<div class="market-row">
			<span class="cube coal"></span> Coal: {gs.coal_market}/14
			<span class="cube iron"></span> Iron: {gs.iron_market}/10
		</div>
	</section>

	<section class="panel controls">
		{#if phase === 'awaiting_start' && !gs.game_over}
			<button class="btn primary" on:click={startTurn}>Start Turn</button>
		{/if}
		{#if phase === 'turn_done' && !gs.game_over}
			<button class="btn secondary" on:click={endTurn}>End Turn</button>
		{/if}
		{#if gs.game_over}
			<div class="game-over">Game Over!</div>
		{/if}
		<button class="btn mat-btn" on:click={() => dispatch('openMat', { playerIndex: gs.current_player })}>My Industry Mat</button>
		<button class="btn mat-btn" on:click={() => dispatch('openDiscard', { playerIndex: gs.current_player })}>My Discard</button>
		{#if (phase === 'in_session' || phase === 'choosing_action' || phase === 'turn_done') && (gs.turn_action_history?.length ?? 0) > 0}
			<button class="btn undo-btn" on:click={undoLastAction}>Undo Previous Action</button>
		{/if}
	</section>

	{#if (gs.turn_action_history?.length ?? 0) > 0 || gs.current_action_selections}
	<section class="panel history-panel">
		<h3>Turn Progress</h3>
		{#each gs.turn_action_history ?? [] as act, idx}
			<div class="history-item">
				<div class="history-title">
					<span class="action-mark">{ACTION_MARK[act.action_type] || '?'}</span>
					<span>{idx + 1}. {ACTION_LABELS[act.action_type] || act.action_type}</span>
				</div>
				{#if act.selections.length > 0}
					<div class="history-tags">
						{#each act.selections as s}
							<span class="history-tag">{s}</span>
						{/each}
					</div>
				{/if}
			</div>
		{/each}
		{#if gs.current_action_selections}
			<div class="history-item current">
				<div class="history-title">
					<span class="action-mark">{ACTION_MARK[gs.current_action_selections.action_type] || '?'}</span>
					<span>Current: {ACTION_LABELS[gs.current_action_selections.action_type] || gs.current_action_selections.action_type}</span>
				</div>
				{#if gs.current_action_selections.selections.length > 0}
					<div class="history-tags">
						{#each gs.current_action_selections.selections as s}
							<span class="history-tag">{s}</span>
						{/each}
					</div>
				{:else}
					<div class="history-tags"><span class="history-tag">No selections yet</span></div>
				{/if}
			</div>
		{/if}
	</section>
	{/if}

	{#if phase === 'choosing_action' && actions}
	<section class="panel">
		<h3>Actions</h3>
		<div class="btn-grid">
			{#each actions as a}
				<button class="btn action" on:click={() => selectAction(a)}>
					{ACTION_LABELS[a] || a}
				</button>
			{/each}
		</div>
	</section>
	{/if}

	{#if phase === 'in_session' && cs && !MAP_KINDS.has(cs.kind) && !MODAL_KINDS.has(cs.kind)}
	<section class="panel choice-panel">
		<h3>{CHOICE_TITLES[cs.kind] || 'Choose'}</h3>
		<div class="btn-grid">
			{#each cs.options as opt}
				<button class="btn choice" on:click={() => onChoice(opt)}>
					{opt.label}
				</button>
			{/each}
		</div>
		<button class="btn cancel" on:click={cancelAction}>Cancel</button>
	</section>
	{/if}

	{#if phase === 'in_session' && cs && MAP_KINDS.has(cs.kind)}
	<section class="panel hint-panel">
		<h3>{CHOICE_TITLES[cs.kind] || 'Choose on map'}</h3>
		<p class="hint-text">Click a highlighted location on the board</p>
		<button class="btn cancel" on:click={cancelAction}>Cancel</button>
	</section>
	{/if}

	{#if phase === 'in_session' && cs && (cs.kind === 'industry' || cs.kind === 'free_development' || cs.kind === 'second_industry')}
	<section class="panel hint-panel develop-hint">
		<h3>{CHOICE_TITLES[cs.kind] || 'Choose'}</h3>
		{#if pendingDevs.length > 0}
			<div class="dev-selections">
				<span class="dev-label">Selected:</span>
				{#each pendingDevs as dev}
					<div class="dev-tile" style="--ind-color: {INDUSTRY_COLORS[dev.industry] ?? '#666'}">
						<img src={devIconPath(dev.industry)} alt={dev.industry} class="dev-tile-icon" />
						<span class="dev-tile-label">{dev.industry} {levelToRoman(dev.level)}</span>
					</div>
				{/each}
			</div>
		{/if}
		<p class="hint-text">Select from the Industry Mat</p>
		<button class="btn cancel" on:click={cancelAction}>Cancel</button>
	</section>
	{/if}

	{#if phase === 'in_session' && cs && cs.kind === 'card'}
	<section class="panel hint-panel">
		<h3>Choose Card</h3>
		<p class="hint-text">Click a card from your hand</p>
		<button class="btn cancel" on:click={cancelAction}>Cancel</button>
	</section>
	{/if}

	<section class="panel log-panel">
		<h3>Log</h3>
		<div class="log-scroll">
			{#each $logs as msg}
				<div class="log-entry">{msg}</div>
			{/each}
		</div>
	</section>
</div>
{/if}

<style>
	.sidebar {
		display: flex;
		flex-direction: column;
		padding: 0 18px 24px;
		overflow-y: auto;
		height: 100%;
		width: 100%;
		position: relative;
		background: #f5f6f4;
		color: #202321;
	}
	.round-toast {
		position: sticky;
		top: 0;
		z-index: 5;
		text-align: center;
		background: #087f5b;
		color: #fff;
		border: 0;
		border-radius: 0 0 6px 6px;
		padding: 8px 10px;
		font-size: 12px;
		font-weight: 700;
		animation: toast-in 180ms ease-out;
	}
	@keyframes toast-in {
		from { opacity: 0; transform: translateY(-6px); }
		to { opacity: 1; transform: translateY(0); }
	}
	.panel {
		background: transparent;
		border-radius: 0;
		padding: 14px 0;
		border: 0;
		border-bottom: 1px solid #d4d8d5;
	}
	.panel h3 {
		font-size: 10px;
		text-transform: uppercase;
		letter-spacing: 0;
		color: #747a76;
		margin: 0 0 6px;
	}
	.phase-panel { text-align: left; padding-top: 18px; }
	.phase-era { font-size: 18px; font-weight: 750; color: #202321; }
	.phase-sub { font-size: 11px; color: #747a76; margin-top: 3px; }
	.seed-chip {
		margin-top: 8px;
		display: inline-flex;
		align-items: center;
		gap: 6px;
		background: #fff;
		border: 1px solid #cbd0cc;
		color: #555b57;
		border-radius: 5px;
		padding: 4px 10px;
		font-size: 11px;
		font-weight: 700;
		cursor: pointer;
	}
	.seed-chip:hover { border-color: #087f5b; color: #087f5b; }
	.seed-copied { margin-top: 4px; font-size: 10px; color: #087f5b; }

	.player-row {
		display: flex; align-items: center; padding: 4px 6px; border-radius: 5px;
		margin-bottom: 2px; font-size: 12px;
	}
	.player-row.active { background: #e4f3ed; box-shadow: inset 2px 0 #087f5b; }
	.turn-num {
		font-size: 10px; font-weight: 700; color: #7c827e;
		width: 16px; text-align: center; flex-shrink: 0;
	}
	.dot { width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; flex-shrink: 0; }
	.pname { font-weight: 600; min-width: 60px; }
	.pstats { color: #666c68; font-size: 11px; margin-left: auto; white-space: nowrap; }
	.money-delta { color: #b54031; font-size: 10px; }
	.mini-btn {
		margin-left: 4px;
		padding: 2px 6px;
		font-size: 10px;
		border-radius: 6px;
		border: 1px solid #c6cbc7;
		background: #fff;
		color: #545a56;
		cursor: pointer;
	}
	.mini-btn:hover {
		border-color: #087f5b;
		color: #087f5b;
	}

	.market-row { display: flex; align-items: center; gap: 10px; font-size: 13px; }
	.cube { width: 12px; height: 12px; border-radius: 3px; display: inline-block; }
	.cube.coal { background: #1a1a1a; border: 1px solid #555; }
	.cube.iron { background: #f97316; }

	.controls { display: flex; justify-content: center; gap: 8px; flex-wrap: wrap; }
	.game-over { text-align: center; font-size: 18px; font-weight: 700; color: #b54031; }

	.btn {
		padding: 7px 14px; border: none; border-radius: 6px; cursor: pointer;
		font-size: 12px; font-weight: 600; color: #fff; transition: background 0.15s;
	}
	.btn.primary { background: #087f5b; }
	.btn.primary:hover { background: #066b4c; }
	.btn.secondary { background: #343936; }
	.btn.secondary:hover { background: #242826; }
	.btn.action { background: #087f5b; }
	.btn.action:hover { background: #066b4c; }
	.btn.choice { background: #006994; min-width: 90px; }
	.btn.choice:hover { background: #00577b; }
	.btn.cancel { background: #b54031; margin-top: 6px; }
	.btn.cancel:hover { background: #943327; }
	.btn.mat-btn { background: #fff; color: #343936; border: 1px solid #c6cbc7; font-size: 11px; padding: 5px 10px; }
	.btn.mat-btn:hover { border-color: #087f5b; color: #087f5b; }
	.btn.undo-btn { background: #b54031; font-size: 11px; padding: 5px 10px; }
	.btn.undo-btn:hover { background: #943327; }

	.btn-grid { display: flex; flex-wrap: wrap; gap: 4px; }

	.hint-panel { text-align: center; }
	.hint-text { color: #087f5b; font-size: 13px; margin: 8px 0; }
	.history-panel { display: flex; flex-direction: column; gap: 6px; }
	.history-item { background: #fff; border: 1px solid #d6dad7; border-radius: 6px; padding: 7px; }
	.history-item.current { border-color: #087f5b; }
	.history-title { display: flex; align-items: center; gap: 6px; font-size: 12px; color: #252925; font-weight: 700; }
	.action-mark { display: grid; place-items: center; width: 22px; height: 22px; border-radius: 50%; background: #e5e8e5; color: #404541; font-size: 9px; }
	.history-tags { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 6px; }
	.history-tag { font-size: 10px; color: #575d59; background: #eceeec; border-radius: 4px; padding: 2px 6px; }
	.develop-hint .dev-selections {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 8px;
		margin: 8px 0;
		justify-content: center;
	}
	.dev-label { font-size: 11px; color: #747a76; margin-right: 4px; }
	.dev-tile {
		display: flex;
		align-items: center;
		gap: 6px;
		padding: 6px 10px;
		background: #fff;
		border-radius: 6px;
		border: 1px solid var(--ind-color, #555);
	}
	.dev-tile-icon { width: 24px; height: 24px; object-fit: contain; }
	.dev-tile-label { font-size: 12px; font-weight: 600; color: #292d2a; }

	.log-panel { flex: 1; min-height: 80px; }
	.log-scroll { max-height: 200px; overflow-y: auto; font-size: 10px; color: #747a76; line-height: 1.6; }
	.log-entry { border-bottom: 1px solid #e0e3e0; padding: 2px 0; }
</style>
