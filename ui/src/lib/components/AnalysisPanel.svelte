<script lang="ts">
	import { createEventDispatcher, onDestroy } from 'svelte';
	import { get } from 'svelte/store';
	import {
		AlertTriangle,
		BarChart3,
		Bot,
		BrainCircuit,
		ChevronDown,
		LoaderCircle,
		MessageCircle,
		Pause,
		Play,
		RefreshCw,
		Send,
		SkipForward,
		Square
	} from 'lucide-svelte';
	import {
		aiPlayback,
		analysisAutoEnabled,
		analysisError,
		analysisLoading,
		analysisProgress,
		analysisReport,
		aiControlsPlayer,
		gameState,
		playerName,
		selectedAnalysisCandidate,
		selectedAnalysisKey,
		turnPhase,
		type AiPlaybackStatus
	} from '$lib/store';
	import {
		analyzePosition,
		analyzePositionProgressive,
		applyAnalyzedAction,
		endTurn,
		explainAnalyzedAction,
		resolveShortfalls,
		startTurn
	} from '$lib/api';
	import type { AnalysisCandidate, AnalysisExplanation, AnalysisReport, GameControlMode, GameState } from '$lib/types';

	const dispatch = createEventDispatcher();
	export let controlMode: GameControlMode = 'human-vs-ai';
	export let humanPlayerIndex = 0;
	export let controlSyncing = false;
	const budgets = [
		{ value: 800, label: '快速' },
		{ value: 3000, label: '标准' },
		{ value: 15000, label: '深入' }
	];
	const playbackDelays = [
		{ value: 4000, label: '0.5×' },
		{ value: 2000, label: '1×' },
		{ value: 600, label: '2×' }
	];

	type Exchange = { question: string; response: AnalysisExplanation };

	let simulations = 3000;
	let question = '';
	let asking = false;
	let applying = false;
	let continuationOpen = false;
	let exchanges: Exchange[] = [];
	let previousSelection: string | null = null;
	let playbackDelay = 2000;
	let loopGeneration = 0;
	let delayTimer: ReturnType<typeof setTimeout> | null = null;
	let finishDelay: (() => void) | null = null;
	let controlIdentity = '';
	let lastAutoStartKey: string | null = null;
	let lastAutoAnalysisKey: string | null = null;

	$: report = $analysisReport;
	$: usesSuccessorValues = report?.value_source === 'batched_successor_model';
	$: usesNeuralTree = report?.value_source === 'batched_neural_tree_search';
	$: selected = $selectedAnalysisCandidate;
	$: phase = $turnPhase;
	$: gs = $gameState;
	$: aiCanMove = gs
		? aiControlsPlayer(controlMode, humanPlayerIndex, gs.current_player)
		: false;
	$: humanName = gs ? playerName(gs, humanPlayerIndex) : '你';
	$: canAnalyze = phase === 'choosing_action' && !gs?.game_over && !gs?.has_pending_shortfall;
	$: canRunAi = !controlSyncing && aiCanMove && !gs?.game_over && phase !== 'in_session';
	$: analysisPositionKey = gs
		? JSON.stringify([controlMode, humanPlayerIndex, gs])
		: null;
	$: canAutoAnalyze = canAnalyze && !aiCanMove && !gs?.game_over;
	$: aiPositionKey = gs
		? [
			controlMode,
			humanPlayerIndex,
			gs.seed,
			gs.era,
			gs.round_in_phase,
			gs.turn_count,
			gs.current_player,
			gs.actions_remaining,
			phase,
			gs.choice_set?.kind ?? '',
			gs.has_pending_shortfall ? 'shortfall' : 'ready'
		].join(':')
		: null;
	$: if (controlIdentity !== `${controlMode}:${humanPlayerIndex}`) {
		controlIdentity = `${controlMode}:${humanPlayerIndex}`;
		lastAutoStartKey = null;
		lastAutoAnalysisKey = null;
	}
	$: effects = selected ? immediateEffects(selected) : [];
	$: if (selected?.action_key !== previousSelection) {
		previousSelection = selected?.action_key ?? null;
		exchanges = [];
		continuationOpen = false;
	}
	$: if (!aiCanMove && $aiPlayback.status === 'running') pauseAutoplay();
	$: if (!canAutoAnalyze) lastAutoAnalysisKey = null;
	$: if (
		$analysisAutoEnabled &&
		canAutoAnalyze &&
		!$analysisLoading &&
		!applying &&
		!report &&
		analysisPositionKey &&
		analysisPositionKey !== lastAutoAnalysisKey
	) {
		lastAutoAnalysisKey = analysisPositionKey;
		exchanges = [];
		void analyzePositionProgressive(simulations);
	}
	$: if (
		canRunAi &&
		!$analysisLoading &&
		!applying &&
		$aiPlayback.status !== 'running' &&
		aiPositionKey &&
		aiPositionKey !== lastAutoStartKey
	) {
		lastAutoStartKey = aiPositionKey;
		void startAutoplay();
	}

	function stateAllowsAi(state: GameState): boolean {
		return aiControlsPlayer(controlMode, humanPlayerIndex, state.current_player);
	}

	async function runAnalysis() {
		if (!canAnalyze || $analysisLoading) return;
		pauseAutoplay();
		exchanges = [];
		await analyzePositionProgressive(simulations);
	}

	function toggleAutoAnalysis(event: Event) {
		lastAutoAnalysisKey = null;
		analysisAutoEnabled.set((event.currentTarget as HTMLInputElement).checked);
	}

	function selectCandidate(candidate: AnalysisCandidate) {
		pauseAutoplay();
		selectedAnalysisKey.set(candidate.action_key);
	}

	async function ask(text = question) {
		const prompt = text.trim();
		if (!report || !selected || !prompt || asking) return;
		pauseAutoplay();
		asking = true;
		question = '';
		try {
			const response = await explainAnalyzedAction(report, selected.action_key, prompt);
			if (response) exchanges = [...exchanges, { question: prompt, response }];
		} finally {
			asking = false;
		}
	}

	async function applySelected() {
		if (!report || !selected || applying) return;
		pauseAutoplay();
		applying = true;
		try {
			const result = await applyAnalyzedAction(report, selected.action_key);
			if (result) dispatch('applied');
		} finally {
			applying = false;
		}
	}

	async function prepareAiDecision(): Promise<boolean> {
		let state = get(gameState);
		if (!state || state.game_over || !stateAllowsAi(state)) return false;
		if (state.has_pending_shortfall) {
			const resolved = await resolveShortfalls();
			if (!resolved) throw new Error('欠款处理失败');
			state = get(gameState);
			if (!state || state.game_over || !stateAllowsAi(state)) return false;
		}

		const currentPhase = get(turnPhase);
		if (currentPhase === 'in_session') throw new Error('请先完成或取消当前手动动作');
		if (currentPhase === 'turn_done') {
			if (!await endTurn()) throw new Error('结束回合失败');
			state = get(gameState);
			if (!state || state.game_over || !stateAllowsAi(state)) return false;
		}
		if (get(turnPhase) === 'awaiting_start') {
			state = get(gameState);
			if (!state || !stateAllowsAi(state)) return false;
			if (!await startTurn()) throw new Error('开始回合失败');
		}
		return get(turnPhase) === 'choosing_action';
	}

	async function analyzeForPlayback(): Promise<AnalysisReport | null> {
		if (!await prepareAiDecision()) return null;
		exchanges = [];
		return analyzePosition(simulations);
	}

	function waitForPlaybackDelay(): Promise<void> {
		cancelPlaybackDelay();
		return new Promise(resolve => {
			finishDelay = () => {
				if (delayTimer) clearTimeout(delayTimer);
				delayTimer = null;
				finishDelay = null;
				resolve();
			};
			delayTimer = setTimeout(() => finishDelay?.(), playbackDelay);
		});
	}

	function cancelPlaybackDelay() {
		finishDelay?.();
	}

	async function startAutoplay() {
		if (!canRunAi || $analysisLoading || applying || $aiPlayback.status === 'running') return;
		const previous = get(aiPlayback);
		const resetMoves = previous.status === 'idle' || previous.status === 'complete' || previous.status === 'error';
		aiPlayback.set({ status: 'running', moves: resetMoves ? 0 : previous.moves, error: null });
		const generation = ++loopGeneration;
		await autoplayLoop(generation);
	}

	async function autoplayLoop(generation: number) {
		try {
			while (generation === loopGeneration && get(aiPlayback).status === 'running') {
				const stateAtLoopStart = get(gameState);
				if (!stateAtLoopStart || !stateAllowsAi(stateAtLoopStart)) {
					aiPlayback.update(state => ({ ...state, status: 'paused', error: null }));
					return;
				}
				let currentReport = get(analysisReport);
				if (!currentReport) currentReport = await analyzeForPlayback();
				if (!currentReport) {
					if (get(gameState)?.game_over) {
						aiPlayback.update(state => ({ ...state, status: 'complete' }));
						return;
					}
					const currentState = get(gameState);
					if (currentState && !stateAllowsAi(currentState)) {
						aiPlayback.update(state => ({ ...state, status: 'paused', error: null }));
						return;
					}
					throw new Error('当前局面无法分析');
				}

				await waitForPlaybackDelay();
				if (generation !== loopGeneration || get(aiPlayback).status !== 'running') return;
				currentReport = get(analysisReport);
				const best = currentReport?.recommendations[0];
				if (!currentReport || !best) throw new Error('当前分析没有可执行的一选');
				applying = true;
				const result = await applyAnalyzedAction(currentReport, best.action_key);
				applying = false;
				if (!result) throw new Error('AI 落子失败');
				aiPlayback.update(state => ({ ...state, moves: state.moves + 1 }));
				if (result.state?.game_over) {
					aiPlayback.update(state => ({ ...state, status: 'complete' }));
					return;
				}
				if (!stateAllowsAi(result.state as GameState)) {
					aiPlayback.update(state => ({ ...state, status: 'paused', error: null }));
					return;
				}
			}
		} catch (error) {
			applying = false;
			if (generation !== loopGeneration) return;
			aiPlayback.update(state => ({
				...state,
				status: 'error',
				error: error instanceof Error ? error.message : String(error)
			}));
		}
	}

	function pauseAutoplay() {
		if (get(aiPlayback).status !== 'running') return;
		loopGeneration += 1;
		cancelPlaybackDelay();
		aiPlayback.update(state => ({ ...state, status: 'paused' }));
	}

	function stopAutoplay(resetMoves = true) {
		loopGeneration += 1;
		cancelPlaybackDelay();
		const moves = resetMoves ? 0 : get(aiPlayback).moves;
		aiPlayback.set({ status: 'idle', moves, error: null });
	}

	async function stepOnce() {
		if (!canRunAi || $analysisLoading || applying) return;
		pauseAutoplay();
		try {
			let currentReport = get(analysisReport);
			if (!currentReport) {
				currentReport = await analyzeForPlayback();
				if (currentReport) aiPlayback.update(state => ({ ...state, status: 'paused', error: null }));
				return;
			}
			const best = currentReport.recommendations[0];
			if (!best) throw new Error('当前分析没有可执行的一选');
			applying = true;
			const result = await applyAnalyzedAction(currentReport, best.action_key);
			applying = false;
			if (!result) throw new Error('AI 落子失败');
			aiPlayback.update(state => ({ ...state, moves: state.moves + 1, status: result.state?.game_over ? 'complete' : 'paused', error: null }));
			if (!result.state?.game_over && stateAllowsAi(result.state as GameState)) await analyzeForPlayback();
		} catch (error) {
			applying = false;
			aiPlayback.update(state => ({ ...state, status: 'error', error: error instanceof Error ? error.message : String(error) }));
		}
	}

	function playbackStatus(
		status: AiPlaybackStatus,
		moves: number,
		canMove: boolean,
		mode: GameControlMode,
		playerName: string
	): string {
		if (!canMove) {
			if (mode === 'manual') return '全部手动';
			return `${playerName} 回合`;
		}
		switch (status) {
			case 'running': return `自动运行中 · ${moves} 手`;
			case 'paused': return `已暂停 · ${moves} 手`;
			case 'complete': return `已结束 · ${moves} 手`;
			case 'error': return '已停止';
			default: return '待机';
		}
	}

	onDestroy(() => stopAutoplay(false));

	function percent(value: number | null): string {
		return value == null ? '—' : `${(value * 100).toFixed(1)}%`;
	}

	function signed(value: number): string {
		return value > 0 ? `+${value}` : String(value);
	}

	function immediateEffects(candidate: AnalysisCandidate): { label: string; value: string }[] {
		const effect = candidate.immediate_effect;
		const items: { label: string; value: string }[] = [];
		if (effect.money_delta) items.push({ label: '现金', value: `£${signed(effect.money_delta)}` });
		if (effect.income_level_delta) items.push({ label: '收入轨', value: signed(effect.income_level_delta) });
		if (effect.potential_era_victory_points_delta) {
			items.push({ label: '时代潜在 VP', value: signed(effect.potential_era_victory_points_delta) });
		}
		if (effect.roads_on_board_delta) items.push({ label: '线路', value: signed(effect.roads_on_board_delta) });
		if (effect.buildings_on_board_delta) items.push({ label: '建筑', value: signed(effect.buildings_on_board_delta) });
		if (effect.flipped_buildings_delta) items.push({ label: '翻面', value: signed(effect.flipped_buildings_delta) });
		if (effect.wild_location_pool_delta || effect.wild_industry_pool_delta) {
			items.push({ label: '万能牌', value: '+2' });
		}
		if (items.length === 0) items.push({ label: '立即变化', value: '无公共分数变化' });
		return items;
	}

	function valueSampleLabel(candidate: AnalysisCandidate): string {
		if (candidate.value_source === 'batched_neural_tree_search') return '次树回传';
		if (candidate.value_source === 'batched_successor_model') return '个后继样本';
		return '个续弈样本';
	}

	function valueMetricLabel(candidate: AnalysisCandidate): string {
		if (candidate.value_source === 'batched_neural_tree_search') return '树搜索胜分';
		if (candidate.value_source === 'batched_successor_model') return '一步模型胜分';
		return '样本胜分率';
	}

	function marginMetricLabel(candidate: AnalysisCandidate): string {
		if (candidate.value_source === 'batched_neural_tree_search') return '树搜索分差';
		if (candidate.value_source === 'batched_successor_model') return '模型分差';
		return '平均分差';
	}
</script>

<div class="analysis-panel">
	<header class="analysis-header">
		<div>
			<div class="eyebrow">POSITION REVIEW</div>
			<h2>AI 分析</h2>
		</div>
		<BarChart3 size={20} strokeWidth={1.8} aria-hidden="true" />
	</header>

	<div class="analysis-controls">
		<div class="budget-control" aria-label="分析深度">
			{#each budgets as budget}
				<button
					class:active={simulations === budget.value}
					on:click={() => simulations = budget.value}
					disabled={$analysisLoading}
				>{budget.label}</button>
			{/each}
		</div>
		<button class="analyze-button" on:click={runAnalysis} disabled={!canAnalyze || $analysisLoading}>
			{#if $analysisLoading}
				<span class="spin"><LoaderCircle size={17} aria-hidden="true" /></span>
				<span>搜索中{#if $analysisProgress} · {$analysisProgress.completed.toLocaleString()}/{$analysisProgress.target.toLocaleString()}{/if}</span>
			{:else if report}
				<RefreshCw size={17} aria-hidden="true" />
				<span>重新分析</span>
			{:else}
				<BrainCircuit size={17} aria-hidden="true" />
				<span>分析局面</span>
			{/if}
		</button>
	</div>
	<label class="auto-analysis-toggle">
		<input type="checkbox" checked={$analysisAutoEnabled} on:change={toggleAutoAnalysis} />
		<span>自动分析玩家回合</span>
	</label>

	<section class="playback-control" aria-label="AI 对局控制">
		<div class="playback-main">
			<div class="playback-title">
				<span class="playback-icon" class:live={$aiPlayback.status === 'running'}><Bot size={17} aria-hidden="true" /></span>
				<div><strong>AI 对局</strong><span>{playbackStatus($aiPlayback.status, $aiPlayback.moves, aiCanMove, controlMode, humanName)}</span></div>
			</div>
			<div class="playback-buttons">
				<button on:click={startAutoplay} disabled={!canRunAi || $analysisLoading || applying || $aiPlayback.status === 'running'} title={$aiPlayback.status === 'paused' ? '继续' : '播放'} aria-label={$aiPlayback.status === 'paused' ? '继续 AI 对局' : '播放 AI 对局'}>
					<Play size={16} fill="currentColor" aria-hidden="true" />
				</button>
				<button on:click={pauseAutoplay} disabled={$aiPlayback.status !== 'running'} title="暂停" aria-label="暂停 AI 对局">
					<Pause size={16} fill="currentColor" aria-hidden="true" />
				</button>
				<button on:click={stepOnce} disabled={!canRunAi || $analysisLoading || applying || $aiPlayback.status === 'running'} title="分析或前进一步" aria-label="AI 分析或前进一步">
					<SkipForward size={16} fill="currentColor" aria-hidden="true" />
				</button>
				<button on:click={() => stopAutoplay(true)} disabled={$aiPlayback.status === 'idle'} title="停止" aria-label="停止 AI 对局">
					<Square size={14} fill="currentColor" aria-hidden="true" />
				</button>
			</div>
		</div>
		<div class="playback-speed" aria-label="播放速度">
			{#each playbackDelays as speed}
				<button class:active={playbackDelay === speed.value} on:click={() => playbackDelay = speed.value}>{speed.label}</button>
			{/each}
		</div>
		{#if $aiPlayback.error}<div class="playback-error">{$aiPlayback.error}</div>{/if}
	</section>

	{#if !canAnalyze && !report}
		<div class="availability-note">
			<AlertTriangle size={16} aria-hidden="true" />
			<span>{phase === 'in_session' ? '先完成或取消当前动作' : '开始回合后可分析'}</span>
		</div>
	{/if}

	{#if $analysisError}
		<div class="error-note">{$analysisError}</div>
	{/if}

	{#if $analysisLoading && $analysisProgress}
		<div class="analysis-progress" aria-live="polite">
			<div class="progress-copy">
				<span>分阶段搜索</span>
				<strong>{$analysisProgress.completed.toLocaleString()} / {$analysisProgress.target.toLocaleString()} · 第 {$analysisProgress.stage}/{$analysisProgress.totalStages} 阶段</strong>
			</div>
			<div class="progress-track"><span style={`width: ${Math.min(100, ($analysisProgress.completed / Math.max(1, $analysisProgress.target)) * 100)}%`}></span></div>
		</div>
	{/if}

	{#if $analysisLoading && !report}
		<div class="analysis-skeleton" aria-label="正在分析">
			<div class="skeleton-line wide"></div>
			<div class="skeleton-row"></div>
			<div class="skeleton-row"></div>
			<div class="skeleton-row"></div>
		</div>
	{:else if report}
		<div class="coverage-line">
			<span>{report.completed_simulations.toLocaleString()} 次模拟</span>
			<span>{report.evaluated_action_count}/{report.root_action_count} 估值 · {report.visited_action_count} 访问</span>
			<span>{(report.elapsed_ms / 1000).toFixed(2)}s</span>
		</div>
		<div class="method-line">
			<span class:model={report.model_id != null} title={report.model_id ?? report.method}>
				{usesNeuralTree ? '深层 PUCT' : usesSuccessorValues ? '一步价值 PUCT' : report.model_id ? '策略 PUCT' : '随机 UCB'}
			</span>
			<strong>{report.method_label}</strong>
			{#if report.root_model_shared_win_rate != null}
				<span>根价值 {percent(report.root_model_shared_win_rate)}</span>
			{/if}
		</div>
		{#if usesNeuralTree}
			<div class="search-stats">
				<span>最大深度 <strong>{report.max_search_depth ?? '—'}</strong></span>
				<span>神经叶 <strong>{report.neural_leaf_evaluations?.toLocaleString() ?? '—'}</strong></span>
				<span>推理批次 <strong>{report.inference_batches?.toLocaleString() ?? '—'}</strong></span>
			</div>
		{/if}
		{#if !report.all_root_actions_evaluated}
			<div class="coverage-warning">
				{usesNeuralTree
					? `PUCT 本轮访问 ${report.visited_action_count}/${report.root_action_count} 个根动作；其余动作有策略先验但尚无树搜索价值回传。`
					: `仅覆盖 ${percent(report.coverage)}，排名不代表全部合法动作。`}
			</div>
		{/if}

		<section class="candidate-section" aria-label="推荐动作">
			{#each report.recommendations as candidate (candidate.action_key)}
				<button
					class="candidate-row"
					class:selected={selected?.action_key === candidate.action_key}
					on:click={() => selectCandidate(candidate)}
				>
					<span class="rank" class:first={candidate.rank === 1}>{candidate.rank}</span>
					<span class="candidate-copy">
						<strong>{candidate.summary}</strong>
						<span>{candidate.visits} 次访问 · {candidate.value_sample_count} {valueSampleLabel(candidate)} · SE {percent(candidate.shared_win_rate_standard_error)}</span>
					</span>
					<span class="candidate-score">
						<strong>{percent(candidate.estimated_shared_win_rate)}</strong>
						<span>{percent(candidate.visit_share)} 访问</span>
					</span>
				</button>
			{/each}
		</section>

		{#if selected}
			<section class="selected-detail">
				<div class="detail-heading">
					<div>
						<span class="detail-rank">第 {selected.rank} 选择</span>
						<h3>{selected.action_label}</h3>
					</div>
					<button class="apply-button" on:click={applySelected} disabled={applying}>
						{#if applying}<span class="spin"><LoaderCircle size={16} /></span>{:else}<Play size={16} fill="currentColor" />{/if}
						<span>采用此步</span>
					</button>
				</div>

				{#if selected.selections.length > 0}
					<div class="selection-list">
						{#each selected.selections as selection}
							<div>{selection}</div>
						{/each}
					</div>
				{/if}

				<div class="metric-grid">
					<div><span>访问占比</span><strong>{percent(selected.visit_share)}</strong></div>
					<div><span>{valueMetricLabel(selected)}</span><strong>{percent(selected.estimated_shared_win_rate)}</strong></div>
					<div><span>策略概率</span><strong>{percent(selected.policy_probability)}</strong></div>
					<div><span>校准胜率</span><strong>{percent(selected.calibrated_win_rate)}</strong></div>
				</div>
				<div class="secondary-metrics">
					{#if selected.estimated_outright_win_rate != null}<span>独赢 {percent(selected.estimated_outright_win_rate)}</span>{/if}
					{#if selected.estimated_tied_first_rate != null}<span>并列第一 {percent(selected.estimated_tied_first_rate)}</span>{/if}
					{#if selected.average_final_victory_points != null}<span>平均 VP {selected.average_final_victory_points.toFixed(1)}</span>{/if}
					<span>{marginMetricLabel(selected)} {signed(Number(selected.average_victory_point_margin.toFixed(1)))}</span>
				</div>

				<div class="effect-list">
					<div class="section-label">立即变化</div>
					{#each effects as effect}
						<div class="effect-row"><span>{effect.label}</span><strong>{effect.value}</strong></div>
					{/each}
				</div>

				<button class="continuation-toggle" on:click={() => continuationOpen = !continuationOpen}>
					<span>{selected.sample_random_continuation.label}</span>
					<ChevronDown size={16} class={continuationOpen ? 'open' : ''} />
				</button>
				{#if continuationOpen}
					<div class="continuation">
						{#each selected.sample_random_continuation.steps as step}
							<div><span>{step.action_number}</span> P{step.player_idx + 1} · {step.action_label}</div>
						{/each}
						<div class="final-score">终局 VP {selected.sample_random_continuation.final_victory_points.join(' / ')}</div>
					</div>
				{/if}
			</section>

			<section class="why-section">
				<div class="why-heading"><MessageCircle size={17} /><h3>为什么这样走？</h3></div>
				<div class="quick-questions">
					<button on:click={() => ask('为什么这是当前选择？')}>选择依据</button>
					<button on:click={() => ask('这个胜率可靠吗？')}>胜率可信度</button>
					<button on:click={() => ask('这步的风险在哪里？')}>主要风险</button>
					{#if selected.rank > 1}<button on:click={() => ask('和一选相比差在哪里？')}>对比一选</button>{/if}
				</div>

				{#each exchanges as exchange}
					<div class="exchange">
						<div class="question">{exchange.question}</div>
						<div class="answer">{exchange.response.answer}</div>
						<details>
							<summary>搜索证据</summary>
							{#each exchange.response.evidence as evidence}<div>{evidence}</div>{/each}
							<p>{exchange.response.caveat}</p>
						</details>
					</div>
				{/each}

				<form class="question-form" on:submit|preventDefault={() => ask()}>
					<input bind:value={question} maxlength="500" placeholder="继续追问当前走法" aria-label="追问当前走法" />
					<button type="submit" disabled={!question.trim() || asking} title="发送问题" aria-label="发送问题">
						{#if asking}<span class="spin"><LoaderCircle size={17} /></span>{:else}<Send size={17} />{/if}
					</button>
				</form>
			</section>
		{/if}
	{:else}
		<div class="empty-analysis">
			<BrainCircuit size={28} strokeWidth={1.5} aria-hidden="true" />
			<strong>等待局面分析</strong>
			<span>{simulations.toLocaleString()} 次模拟</span>
		</div>
	{/if}
</div>

<style>
	.analysis-panel {
		display: flex;
		flex-direction: column;
		min-height: 100%;
		padding: 18px 18px 28px;
		color: #202321;
		background: #f5f6f4;
	}
	.analysis-header { display: flex; align-items: center; justify-content: space-between; }
	.eyebrow { font-size: 10px; font-weight: 700; color: #737975; }
	h2 { margin: 2px 0 0; font-size: 22px; line-height: 1.1; letter-spacing: 0; }
	.analysis-controls { display: grid; grid-template-columns: 1fr auto; gap: 10px; margin-top: 16px; }
	.budget-control { display: grid; grid-template-columns: repeat(3, 1fr); border: 1px solid #c9ceca; border-radius: 6px; overflow: hidden; }
	.budget-control button { border: 0; border-right: 1px solid #c9ceca; background: #fff; color: #606662; font-size: 12px; cursor: pointer; min-height: 34px; }
	.budget-control button:last-child { border-right: 0; }
	.budget-control button.active { background: #242826; color: #fff; }
	.analyze-button, .apply-button { border: 0; border-radius: 6px; display: inline-flex; align-items: center; justify-content: center; gap: 7px; min-height: 36px; padding: 0 13px; font-weight: 700; cursor: pointer; }
	.analyze-button { color: #fff; background: #087f5b; }
	.apply-button { color: #fff; background: #b54031; white-space: nowrap; }
	.auto-analysis-toggle { display: inline-flex; align-items: center; gap: 7px; margin-top: 10px; color: #4d554f; font-size: 11px; cursor: pointer; }
	.auto-analysis-toggle input { width: 15px; height: 15px; margin: 0; accent-color: #087f5b; cursor: pointer; }
	.playback-control { margin-top: 12px; border-top: 1px solid #cdd2ce; border-bottom: 1px solid #cdd2ce; background: #eceeec; }
	.playback-main { min-height: 48px; display: flex; align-items: center; justify-content: space-between; gap: 10px; padding: 7px 8px; }
	.playback-title { display: flex; align-items: center; gap: 8px; min-width: 0; }
	.playback-title > div { display: flex; flex-direction: column; min-width: 0; }
	.playback-title strong { font-size: 11px; }
	.playback-title span { margin-top: 2px; color: #737975; font-size: 10px; white-space: nowrap; }
	.playback-icon { width: 28px; height: 28px; display: grid; place-items: center; flex: 0 0 auto; border: 1px solid #c3c8c4; border-radius: 50%; background: #fff; color: #555c57; }
	.playback-icon.live { border-color: #087f5b; color: #087f5b; box-shadow: 0 0 0 3px rgba(8,127,91,.1); }
	.playback-buttons { display: flex; gap: 4px; }
	.playback-buttons button { width: 30px; height: 30px; display: grid; place-items: center; border: 1px solid #c4c9c5; border-radius: 5px; background: #fff; color: #303532; cursor: pointer; }
	.playback-buttons button:hover:not(:disabled) { border-color: #087f5b; color: #087f5b; }
	.playback-speed { display: grid; grid-template-columns: repeat(3, 1fr); border-top: 1px solid #d2d6d2; }
	.playback-speed button { min-height: 26px; border: 0; border-right: 1px solid #d2d6d2; background: transparent; color: #6c726e; font-size: 10px; cursor: pointer; }
	.playback-speed button:last-child { border-right: 0; }
	.playback-speed button.active { background: #252a27; color: #fff; }
	.playback-error { padding: 7px 8px; border-top: 1px solid #d2d6d2; color: #9f3429; font-size: 10px; }
	button:disabled { opacity: .45; cursor: not-allowed; }
	.availability-note, .error-note, .coverage-warning { margin-top: 12px; padding: 9px 0; border-bottom: 1px solid #d7dad7; color: #747975; font-size: 12px; }
	.availability-note { display: flex; align-items: center; gap: 8px; }
	.error-note, .coverage-warning { color: #9f3429; }
	.analysis-progress { margin-top: 13px; padding: 9px 0; border-top: 1px solid #d7dad7; border-bottom: 1px solid #d7dad7; }
	.progress-copy { display: flex; justify-content: space-between; gap: 8px; color: #737975; font-size: 10px; }
	.progress-copy strong { color: #303632; font-weight: 650; font-variant-numeric: tabular-nums; text-align: right; }
	.progress-track { height: 4px; margin-top: 7px; overflow: hidden; background: #d9dedb; }
	.progress-track span { display: block; height: 100%; background: #087f5b; transition: width .2s ease; }
	.analysis-skeleton { display: grid; gap: 10px; margin-top: 24px; }
	.skeleton-line, .skeleton-row { background: #e3e6e3; animation: breathe 1.2s ease-in-out infinite; }
	.skeleton-line { width: 70%; height: 12px; }
	.skeleton-row { height: 70px; border-radius: 6px; }
	.coverage-line { display: flex; justify-content: space-between; gap: 8px; margin-top: 16px; color: #737975; font-size: 11px; }
	.method-line { display: flex; flex-wrap: wrap; align-items: center; gap: 5px 8px; margin-top: 8px; color: #747a76; font-size: 10px; }
	.method-line > span:first-child { flex: 0 0 auto; padding: 3px 6px; border: 1px solid #c9ceca; border-radius: 4px; background: #fff; color: #59605b; font-weight: 700; }
	.method-line > span:first-child.model { border-color: #6ab89b; background: #eef9f5; color: #087f5b; }
	.method-line strong { flex: 1 1 190px; min-width: 0; color: #414743; font-weight: 600; line-height: 1.35; }
	.method-line > span:last-child:not(:first-child) { white-space: nowrap; font-variant-numeric: tabular-nums; }
	.search-stats { display: flex; flex-wrap: wrap; gap: 6px 14px; margin-top: 8px; padding: 8px 0; border-top: 1px solid #d7dad7; border-bottom: 1px solid #d7dad7; color: #68706b; font-size: 10px; }
	.search-stats strong { margin-left: 3px; color: #202321; font-variant-numeric: tabular-nums; }
	.candidate-section { display: grid; gap: 7px; margin-top: 12px; }
	.candidate-row { width: 100%; display: grid; grid-template-columns: 28px minmax(0, 1fr) auto; gap: 9px; align-items: center; padding: 10px; text-align: left; border: 1px solid #d3d7d3; border-radius: 6px; background: #fff; color: #262a27; cursor: pointer; transition: border-color .14s, transform .14s, background .14s; }
	.candidate-row:hover { border-color: #7d8580; transform: translateY(-1px); }
	.candidate-row.selected { border-color: #087f5b; background: #f0faf6; }
	.rank { width: 24px; height: 24px; display: grid; place-items: center; border-radius: 50%; background: #e3e6e3; font-size: 12px; font-weight: 800; }
	.rank.first { background: #202321; color: #fff; }
	.candidate-copy, .candidate-score { display: flex; flex-direction: column; min-width: 0; }
	.candidate-copy strong { font-size: 12px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
	.candidate-copy span, .candidate-score span { color: #7a807c; font-size: 10px; margin-top: 3px; }
	.candidate-score { text-align: right; }
	.candidate-score strong { color: #087f5b; font-size: 16px; }
	.selected-detail, .why-section { margin-top: 20px; padding-top: 16px; border-top: 1px solid #cfd3cf; }
	.detail-heading { display: flex; justify-content: space-between; align-items: center; gap: 10px; }
	.detail-rank, .section-label { color: #737975; font-size: 10px; font-weight: 700; text-transform: uppercase; }
	.detail-heading h3, .why-heading h3 { margin: 2px 0 0; font-size: 17px; letter-spacing: 0; }
	.selection-list { display: grid; gap: 4px; margin-top: 12px; padding-left: 12px; border-left: 2px solid #b9bfbb; color: #4e544f; font-size: 12px; }
	.metric-grid { display: grid; grid-template-columns: 1fr 1fr; margin-top: 16px; border-top: 1px solid #d5d9d5; border-left: 1px solid #d5d9d5; }
	.metric-grid div { min-height: 58px; padding: 9px; border-right: 1px solid #d5d9d5; border-bottom: 1px solid #d5d9d5; display: flex; flex-direction: column; justify-content: space-between; }
	.metric-grid span { color: #737975; font-size: 10px; }
	.metric-grid strong { font-size: 17px; }
	.secondary-metrics { display: flex; flex-wrap: wrap; gap: 6px 12px; margin-top: 9px; color: #666c68; font-size: 10px; }
	.effect-list { margin-top: 16px; }
	.effect-row { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #e1e4e1; font-size: 12px; }
	.effect-row strong { font-variant-numeric: tabular-nums; }
	.continuation-toggle { width: 100%; display: flex; align-items: center; justify-content: space-between; margin-top: 13px; padding: 9px 0; border: 0; border-bottom: 1px solid #d5d9d5; background: transparent; color: #333734; font-size: 11px; cursor: pointer; }
	.continuation-toggle :global(svg) { transition: transform .16s; }
	.continuation-toggle :global(svg.open) { transform: rotate(180deg); }
	.continuation { padding: 8px 0 0; color: #5b615d; font-size: 11px; line-height: 1.8; }
	.continuation div span { display: inline-grid; place-items: center; width: 18px; height: 18px; margin-right: 5px; background: #e2e5e2; border-radius: 50%; font-size: 9px; }
	.continuation .final-score { margin-top: 6px; color: #202321; font-weight: 700; }
	.why-heading { display: flex; align-items: center; gap: 8px; }
	.quick-questions { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 11px; }
	.quick-questions button { border: 1px solid #c8cdc9; border-radius: 5px; background: #fff; color: #4e544f; padding: 6px 8px; font-size: 10px; cursor: pointer; }
	.quick-questions button:hover { border-color: #087f5b; color: #087f5b; }
	.exchange { margin-top: 14px; padding-left: 12px; border-left: 2px solid #087f5b; }
	.question { color: #656b67; font-size: 11px; }
	.answer { margin-top: 7px; color: #262a27; font-size: 12px; line-height: 1.65; white-space: pre-line; }
	details { margin-top: 8px; color: #717773; font-size: 10px; }
	details summary { cursor: pointer; color: #087f5b; }
	details div { margin-top: 4px; }
	details p { margin-top: 7px; color: #9f3429; }
	.question-form { display: grid; grid-template-columns: minmax(0, 1fr) 38px; gap: 7px; margin-top: 14px; }
	.question-form input { min-width: 0; height: 38px; border: 1px solid #c8cdc9; border-radius: 6px; background: #fff; color: #202321; padding: 0 10px; font-size: 12px; outline: none; }
	.question-form input:focus { border-color: #087f5b; box-shadow: 0 0 0 2px rgba(8,127,91,.12); }
	.question-form button { display: grid; place-items: center; border: 0; border-radius: 6px; background: #202321; color: #fff; cursor: pointer; }
	.empty-analysis { flex: 1; min-height: 260px; display: flex; flex-direction: column; align-items: center; justify-content: center; color: #828884; }
	.empty-analysis strong { margin-top: 10px; color: #464b47; font-size: 13px; }
	.empty-analysis span { margin-top: 3px; font-size: 11px; }
	.spin { display: inline-flex; animation: spin .8s linear infinite; }
	@keyframes spin { to { transform: rotate(360deg); } }
	@keyframes breathe { 50% { opacity: .55; } }
	@media (prefers-reduced-motion: reduce) {
		.spin, .skeleton-line, .skeleton-row { animation: none; }
		.candidate-row { transition: none; }
	}
</style>
