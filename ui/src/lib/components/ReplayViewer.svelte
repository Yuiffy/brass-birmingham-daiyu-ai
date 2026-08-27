<script lang="ts">
	import { createEventDispatcher, onMount } from 'svelte';
	import { ArrowLeft, ChevronLeft, ChevronRight, History, LoaderCircle } from 'lucide-svelte';
	import { loadReplay } from '$lib/api';
	import {
		analysisReport,
		gameState,
		invalidateAnalysis,
		resetAiPlayback,
		selectedAnalysisCandidate,
		selectedAnalysisKey,
		turnPhase
	} from '$lib/store';
	import type { AnalysisCandidate, ReplayData, ReplayPosition } from '$lib/types';
	import Board from '$lib/components/Board.svelte';

	export let gameId: number;

	const dispatch = createEventDispatcher<{ close: void }>();
	let replay: ReplayData | null = null;
	let loading = true;
	let error = '';
	let selectedIndex = 0;

	$: position = replay?.positions[selectedIndex] ?? null;
	$: selectedCandidate = $selectedAnalysisCandidate;

	$: if (position) {
		gameState.set(position.state);
		turnPhase.set('awaiting_start');
		analysisReport.set(position.analysis);
		selectedAnalysisKey.set(position.analysis?.recommendations[0]?.action_key ?? null);
	}

	onMount(async () => {
		resetAiPlayback();
		invalidateAnalysis();
		try {
			replay = await loadReplay(gameId);
			if (!replay) error = '棋谱加载失败';
		} catch (cause) {
			error = cause instanceof Error ? cause.message : String(cause);
		} finally {
			loading = false;
		}
	});

	function close() {
		invalidateAnalysis();
		dispatch('close');
	}

	function selectPosition(index: number) {
		if (!replay?.positions.length) return;
		selectedIndex = Math.max(0, Math.min(index, replay.positions.length - 1));
	}

	function onSlider(event: Event) {
		selectPosition(Number((event.currentTarget as HTMLInputElement).value));
	}

	function candidateFor(position: ReplayPosition, actionKey: string | null): AnalysisCandidate | null {
		return position.analysis?.recommendations.find(candidate => candidate.action_key === actionKey) ?? null;
	}

	function moveLabel(item: ReplayPosition): string {
		if (!item.move) return '仅保存了分析';
		const candidate = candidateFor(item, item.move.action_key);
		return candidate?.action_label ?? item.move.action_type;
	}

	function playerName(item: ReplayPosition, playerIndex: number): string {
		return item.state.players.find(player => player.index === playerIndex)?.name ?? `P${playerIndex + 1}`;
	}

	function percent(value: number | null | undefined): string {
		return value == null ? '—' : `${(value * 100).toFixed(1)}%`;
	}
</script>

<div class="replay-shell" role="dialog" aria-modal="true" aria-label="棋谱回放">
	<header class="replay-header">
		<div class="replay-title">
			<span class="title-icon"><History size={18} aria-hidden="true" /></span>
			<div>
				<strong>棋谱回放</strong>
				<span>对局 #{gameId} · {replay?.positions.length ?? 0} 个记录局面</span>
			</div>
		</div>
		<button class="back-button" on:click={close}>
			<ArrowLeft size={16} aria-hidden="true" />
			<span>返回存档</span>
		</button>
	</header>

	{#if loading}
		<div class="replay-loading"><span class="spin"><LoaderCircle size={20} aria-hidden="true" /></span><span>读取棋谱</span></div>
	{:else if error}
		<div class="replay-empty"><strong>{error}</strong><span>这个存档暂时无法读取。</span></div>
	{:else if !replay?.positions.length}
		<div class="replay-empty">
			<strong>这个存档还没有棋谱快照</strong>
			<span>新进行的对局会自动保存每手局面和 AI 选择。</span>
		</div>
	{:else if position}
		<main class="replay-main">
			<section class="board-pane">
				<div class="pane-heading">
					<div>
						<span class="eyebrow">POSITION {position.index + 1}</span>
						<h1>{position.move ? `${playerName(position, position.move.player_idx)} · ${moveLabel(position)}` : '分析局面'}</h1>
					</div>
					<span class="era-label">{position.state.era} Era · Turn {position.state.turn_count + 1}</span>
				</div>
				<div class="replay-board"><Board /></div>
				<div class="stepbar">
					<button on:click={() => selectPosition(selectedIndex - 1)} disabled={selectedIndex === 0} title="上一手" aria-label="上一手"><ChevronLeft size={18} /></button>
					<input type="range" min="0" max={(replay?.positions.length ?? 1) - 1} value={selectedIndex} on:input={onSlider} aria-label="选择棋谱局面" />
					<button on:click={() => selectPosition(selectedIndex + 1)} disabled={selectedIndex >= (replay?.positions.length ?? 1) - 1} title="下一手" aria-label="下一手"><ChevronRight size={18} /></button>
				</div>
			</section>

			<aside class="replay-inspector">
				<section class="timeline-section">
					<div class="section-heading"><span>棋谱</span><strong>{selectedIndex + 1}/{replay?.positions.length}</strong></div>
					<div class="timeline-list">
						{#each replay?.positions ?? [] as item}
							<button class:active={item.index === selectedIndex} class="timeline-row" on:click={() => selectPosition(item.index)}>
								<span class="timeline-index">{item.index + 1}</span>
								<span class="timeline-copy">
									<strong>{item.move ? playerName(item, item.move.player_idx) : '分析'}</strong>
									<span>{moveLabel(item)}</span>
								</span>
								{#if item.analysis}<span class="has-analysis" title="已保存一选、二选、三选">AI</span>{/if}
							</button>
						{/each}
					</div>
				</section>

				<section class="move-section">
					<span class="section-label">当前记录</span>
					{#if position.move}
						<h2>{playerName(position, position.move.player_idx)} · {moveLabel(position)}</h2>
						{#if position.move.selections.length}
							<div class="selection-list">
								{#each position.move.selections as selection}<span>{selection}</span>{/each}
							</div>
						{/if}
					{:else}
						<h2>分析但未落子</h2>
					{/if}
				</section>

				{#if position.analysis}
					<section class="candidate-section">
						<div class="section-heading"><span>当时的 AI 候选</span><span class="muted">一选 / 二选 / 三选</span></div>
						{#each position.analysis.recommendations as candidate}
							<button class:active={selectedCandidate?.action_key === candidate.action_key} class="candidate-row" on:click={() => selectedAnalysisKey.set(candidate.action_key)}>
								<span class="rank" class:first={candidate.rank === 1}>{candidate.rank}</span>
								<span class="candidate-copy"><strong>{candidate.action_label}</strong><span>{candidate.summary}</span></span>
								<span class="candidate-score"><strong>{percent(candidate.estimated_shared_win_rate)}</strong><span>{percent(candidate.visit_share)} 访问</span></span>
							</button>
					{/each}
						{#if selectedCandidate}
							<div class="candidate-detail">
								<strong>{selectedCandidate.action_label}</strong>
								<span>{selectedCandidate.summary}</span>
								<div class="detail-stats"><span>访问 {percent(selectedCandidate.visit_share)}</span><span>共享胜分 {percent(selectedCandidate.estimated_shared_win_rate)}</span><span>策略先验 {percent(selectedCandidate.policy_probability)}</span></div>
							</div>
						{/if}
					</section>
				{:else}
					<div class="no-analysis">这手没有保存 AI 分析，只保留了动作和局面。</div>
				{/if}
			</aside>
		</main>
	{/if}
</div>

<style>
	.replay-shell {
		position: fixed;
		inset: 0;
		z-index: 30;
		display: flex;
		flex-direction: column;
		background: #d7dbd8;
		color: #202321;
	}
	.replay-header {
		min-height: 56px;
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 14px;
		padding: 8px 18px;
		border-bottom: 1px solid #bfc5c1;
		background: #f5f6f4;
	}
	.replay-title { display: flex; align-items: center; gap: 9px; min-width: 0; }
	.title-icon { width: 30px; height: 30px; display: grid; place-items: center; border: 1px solid #c4c9c5; border-radius: 50%; color: #087f5b; background: #fff; }
	.replay-title div { display: flex; flex-direction: column; min-width: 0; }
	.replay-title strong { font-size: 14px; }
	.replay-title span { margin-top: 2px; color: #717873; font-size: 10px; }
	.back-button { display: inline-flex; align-items: center; gap: 6px; min-height: 32px; padding: 0 10px; border: 1px solid #c5cac6; border-radius: 5px; background: #fff; color: #3d443f; cursor: pointer; font-size: 11px; }
	.back-button:hover { border-color: #087f5b; color: #087f5b; }
	.replay-main { flex: 1; min-height: 0; display: grid; grid-template-columns: minmax(0, 1fr) 390px; }
	.board-pane { min-width: 0; min-height: 0; display: flex; flex-direction: column; padding: 14px 18px 16px; }
	.pane-heading { display: flex; align-items: end; justify-content: space-between; gap: 12px; margin-bottom: 8px; }
	.eyebrow, .section-label { color: #747a76; font-size: 10px; font-weight: 700; letter-spacing: .04em; }
	.pane-heading h1 { margin: 3px 0 0; font-size: 18px; letter-spacing: 0; }
	.era-label { color: #68706b; font-size: 11px; white-space: nowrap; }
	.replay-board { flex: 1; min-height: 0; display: flex; align-items: center; justify-content: center; overflow: hidden; background: #c9cecb; border: 1px solid #b8bfba; }
	.replay-board :global(canvas) { pointer-events: none; }
	.stepbar { display: grid; grid-template-columns: 34px minmax(0, 1fr) 34px; gap: 8px; align-items: center; margin-top: 10px; }
	.stepbar button { width: 34px; height: 30px; display: grid; place-items: center; border: 1px solid #c4cac5; border-radius: 5px; background: #fff; color: #343a36; cursor: pointer; }
	.stepbar button:hover:not(:disabled) { border-color: #087f5b; color: #087f5b; }
	.stepbar input { width: 100%; accent-color: #087f5b; }
	.replay-inspector { min-width: 0; min-height: 0; overflow-y: auto; padding: 14px 16px 24px; border-left: 1px solid #bfc5c1; background: #f5f6f4; }
	.timeline-section { padding-bottom: 14px; border-bottom: 1px solid #d1d6d2; }
	.section-heading { display: flex; align-items: center; justify-content: space-between; gap: 8px; color: #343a36; font-size: 11px; font-weight: 700; }
	.section-heading strong { color: #087f5b; font-variant-numeric: tabular-nums; }
	.muted { color: #7a817c; font-weight: 500; font-size: 10px; }
	.timeline-list { display: grid; gap: 5px; margin-top: 9px; }
	.timeline-row { width: 100%; display: grid; grid-template-columns: 26px minmax(0, 1fr) auto; gap: 8px; align-items: center; padding: 7px; border: 1px solid #d0d5d1; border-radius: 5px; background: #fff; color: #282d2a; text-align: left; cursor: pointer; }
	.timeline-row:hover { border-color: #7e8781; }
	.timeline-row.active { border-color: #087f5b; background: #eef8f4; }
	.timeline-index { width: 22px; height: 22px; display: grid; place-items: center; border-radius: 50%; background: #e2e6e2; color: #5e655f; font-size: 10px; font-weight: 700; }
	.timeline-copy { display: flex; flex-direction: column; min-width: 0; }
	.timeline-copy strong { overflow: hidden; font-size: 11px; text-overflow: ellipsis; white-space: nowrap; }
	.timeline-copy span { margin-top: 2px; overflow: hidden; color: #737a75; font-size: 10px; text-overflow: ellipsis; white-space: nowrap; }
	.has-analysis { padding: 3px 5px; border: 1px solid #9bcdb9; border-radius: 4px; color: #087f5b; font-size: 9px; font-weight: 800; }
	.move-section, .candidate-section { margin-top: 16px; padding-top: 13px; border-top: 1px solid #d1d6d2; }
	.move-section h2 { margin: 4px 0 0; font-size: 17px; letter-spacing: 0; }
	.selection-list { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 9px; }
	.selection-list span { padding: 4px 6px; border: 1px solid #d0d5d1; border-radius: 4px; background: #fff; color: #5d655f; font-size: 10px; }
	.candidate-section { display: grid; gap: 7px; }
	.candidate-row { width: 100%; display: grid; grid-template-columns: 24px minmax(0, 1fr) auto; gap: 8px; align-items: center; padding: 8px; border: 1px solid #d0d5d1; border-radius: 5px; background: #fff; color: #282d2a; text-align: left; cursor: pointer; }
	.candidate-row:hover { border-color: #7e8781; }
	.candidate-row.active { border-color: #087f5b; background: #eef8f4; }
	.rank { width: 22px; height: 22px; display: grid; place-items: center; border-radius: 50%; background: #e2e6e2; font-size: 11px; font-weight: 800; }
	.rank.first { background: #202321; color: #fff; }
	.candidate-copy { display: flex; flex-direction: column; min-width: 0; }
	.candidate-copy strong { overflow: hidden; font-size: 11px; text-overflow: ellipsis; white-space: nowrap; }
	.candidate-copy span { margin-top: 2px; overflow: hidden; color: #737a75; font-size: 9px; text-overflow: ellipsis; white-space: nowrap; }
	.candidate-score { display: flex; flex-direction: column; align-items: end; white-space: nowrap; }
	.candidate-score strong { color: #087f5b; font-size: 13px; }
	.candidate-score span { margin-top: 2px; color: #737a75; font-size: 9px; }
	.candidate-detail { display: grid; gap: 4px; padding: 9px; border-left: 2px solid #087f5b; background: #edf5f1; font-size: 11px; }
	.candidate-detail span { color: #5f6861; font-size: 10px; }
	.detail-stats { display: flex; flex-wrap: wrap; gap: 5px 10px; color: #4f5952; font-size: 9px; }
	.no-analysis, .replay-empty, .replay-loading { display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 7px; min-height: 220px; color: #747b76; font-size: 12px; text-align: center; }
	.replay-empty strong { color: #3f4641; font-size: 15px; }
	.spin { animation: spin .8s linear infinite; }
	@keyframes spin { to { transform: rotate(360deg); } }
	@media (max-width: 820px) {
		.replay-header { padding-inline: 11px; }
		.replay-main { grid-template-columns: 1fr; overflow-y: auto; }
		.board-pane { min-height: 58svh; padding: 10px 10px 12px; }
		.replay-inspector { overflow: visible; border-top: 1px solid #bfc5c1; border-left: 0; }
		.pane-heading h1 { font-size: 15px; }
		.era-label { font-size: 10px; }
	}
	@media (prefers-reduced-motion: reduce) { .spin { animation: none; } }
</style>
