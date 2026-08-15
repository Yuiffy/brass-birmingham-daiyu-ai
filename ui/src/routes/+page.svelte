<script lang="ts">
	import { get } from 'svelte/store';
	import { onMount } from 'svelte';
	import { BrainCircuit, Gamepad2 } from 'lucide-svelte';
	import {
		analysisReport,
		aiPlayback,
		aiControlsPlayer,
		choiceSet,
		gameState,
		resetAiPlayback,
		selectedAnalysisCandidate,
		turnPhase,
		viewerPlayerIndex
	} from '$lib/store';
	import type { GameControlMode } from '$lib/types';
	import SetupScreen from '$lib/components/SetupScreen.svelte';
	import Board from '$lib/components/Board.svelte';
	import Sidebar from '$lib/components/Sidebar.svelte';
	import AnalysisPanel from '$lib/components/AnalysisPanel.svelte';
	import ControlModeBar from '$lib/components/ControlModeBar.svelte';
	import CardHand from '$lib/components/CardHand.svelte';
	import IndustryMat from '$lib/components/IndustryMat.svelte';
	import DiscardPileViewer from '$lib/components/DiscardPileViewer.svelte';
	import { setObserverPlayer } from '$lib/api';

	let started = false;
	let matOpen = false;
	let matPlayerIndex: number | null = null;
	let discardOpen = false;
	let discardPlayerIndex: number | null = null;
	let inspectorTab: 'game' | 'analysis' = 'game';
	let controlMode: GameControlMode = 'human-vs-ai';
	let humanPlayerIndex = 0;
	let previousControlKey = '';
	let controlSyncing = false;

	function showGameInspector() {
		inspectorTab = 'game';
		aiPlayback.update(playback => playback.status === 'running'
			? { ...playback, status: 'paused' }
			: playback
		);
	}

	async function handleStarted() {
		const state = get(gameState);
		humanPlayerIndex = state?.players[0]?.index ?? 0;
		controlMode = state?.players.length === 2 ? 'human-vs-ai' : 'manual';
		previousControlKey = '';
		inspectorTab = 'game';
		controlSyncing = true;
		try {
			await setObserverPlayer(controlMode === 'human-vs-ai' ? humanPlayerIndex : null);
		} finally {
			controlSyncing = false;
			started = true;
		}
	}

	async function handleControlChange(event: CustomEvent<{ mode: GameControlMode; humanPlayerIndex: number }>) {
		controlMode = event.detail.mode;
		humanPlayerIndex = event.detail.humanPlayerIndex;
		resetAiPlayback();
		previousControlKey = '';
		controlSyncing = true;
		try {
			await setObserverPlayer(controlMode === 'human-vs-ai' ? humanPlayerIndex : null);
		} finally {
			controlSyncing = false;
		}
	}

	function handleAnalysisApplied() {
		const state = get(gameState);
		inspectorTab = state && aiControlsPlayer(controlMode, humanPlayerIndex, state.current_player)
			? 'analysis'
			: 'game';
	}

	$: gs = $gameState;
	$: cs = $choiceSet;
	$: isCardChoice = cs?.kind === 'card';
	$: if (gs && !gs.players.some(player => player.index === humanPlayerIndex)) {
		humanPlayerIndex = gs.players[0]?.index ?? 0;
	}
	$: aiTurn = gs ? aiControlsPlayer(controlMode, humanPlayerIndex, gs.current_player) : false;
	$: handPlayerIndex = gs
		? viewerPlayerIndex(controlMode, humanPlayerIndex, gs.current_player)
		: null;
	$: handPlayer = gs?.players.find(player => player.index === handPlayerIndex) ?? null;
	$: handOwnerLabel = handPlayer
		? `${controlMode === 'human-vs-ai' ? '你的手牌' : '当前手牌'} · ${handPlayer.name}`
		: '';
	$: if (started && gs) {
		const controlKey = `${controlMode}:${humanPlayerIndex}:${gs.current_player}`;
		if (controlKey !== previousControlKey) {
			previousControlKey = controlKey;
			inspectorTab = aiTurn ? 'analysis' : 'game';
		}
	}

	onMount(() => {
		const testWindow = window as typeof window & {
			render_game_to_text?: () => string;
			advanceTime?: (ms: number) => void;
			__brassVisualTime?: number;
		};
		testWindow.render_game_to_text = () => {
			const state = get(gameState);
			const report = get(analysisReport);
			const selected = get(selectedAnalysisCandidate);
			const playback = get(aiPlayback);
			return JSON.stringify({
				mode: !started ? 'setup' : state?.game_over ? 'game_over' : get(turnPhase),
				coordinate_system: 'Board image pixels; origin top-left; x right; y down.',
				control: state ? {
					mode: controlMode,
					human_player_index: controlMode === 'human-vs-ai' ? humanPlayerIndex : null,
					current_controller: aiControlsPlayer(controlMode, humanPlayerIndex, state.current_player) ? 'ai' : 'human',
					hand_player_index: viewerPlayerIndex(controlMode, humanPlayerIndex, state.current_player)
				} : null,
				game: state ? {
					era: state.era,
					round: state.round_in_phase + 1,
					turn: state.turn_count + 1,
					current_player: state.current_player,
					actions_remaining: state.actions_remaining,
					players: state.players.map(player => ({
						index: player.index,
						money: player.money,
						income: player.income_amount,
						vp: player.victory_points,
						hand_size: player.hand_size
					})),
					buildings: state.buildings,
					roads: state.roads,
					choice_set: state.choice_set,
					available_actions: state.available_actions
				} : null,
				analysis: report ? {
					method: report.method,
					value_source: report.value_source,
					model_id: report.model_id,
					root_model_shared_win_rate: report.root_model_shared_win_rate,
					simulations: report.completed_simulations,
					max_search_depth: report.max_search_depth,
					neural_leaf_evaluations: report.neural_leaf_evaluations,
					inference_batches: report.inference_batches,
					coverage: report.coverage,
					evaluated_action_count: report.evaluated_action_count,
					visited_action_count: report.visited_action_count,
					selected_action_key: selected?.action_key ?? null,
					recommendations: report.recommendations.map(candidate => ({
						rank: candidate.rank,
						action_key: candidate.action_key,
						visits: candidate.visits,
						visit_share: candidate.visit_share,
						value_source: candidate.value_source,
						value_sample_count: candidate.value_sample_count,
						policy_probability: candidate.policy_probability,
						estimated_shared_win_rate: candidate.estimated_shared_win_rate
					}))
				} : null,
				ai_playback: playback
			});
		};
		testWindow.advanceTime = (ms: number) => {
			testWindow.__brassVisualTime = (testWindow.__brassVisualTime ?? 0) + Math.max(0, ms);
			window.dispatchEvent(new CustomEvent('brass:advance-time'));
		};

		const toggleFullscreen = (event: KeyboardEvent) => {
			if (event.key.toLowerCase() !== 'f') return;
			const target = event.target as HTMLElement | null;
			if (target?.matches('input, textarea, select, [contenteditable="true"]')) return;
			if (document.fullscreenElement) void document.exitFullscreen();
			else void document.documentElement.requestFullscreen();
		};
		window.addEventListener('keydown', toggleFullscreen);
		return () => {
			window.removeEventListener('keydown', toggleFullscreen);
			delete testWindow.render_game_to_text;
			delete testWindow.advanceTime;
		};
	});
</script>

{#if !started}
	<SetupScreen on:started={handleStarted} />
{:else if gs}
	<div class="game-layout">
		<div class="left-col">
			<div class="board-area">
				<Board />
			</div>
			<div class="bottom-bar" class:card-active={isCardChoice}>
				{#if $turnPhase === 'in_session' && isCardChoice}
					<div class="card-prompt">Pick a card to discard</div>
				{/if}
				<CardHand playerIndex={handPlayerIndex} ownerLabel={handOwnerLabel} interactionLocked={aiTurn} />
			</div>
		</div>
		<div class="right-col">
			<nav class="inspector-tabs" aria-label="侧边栏视图">
				<button class:active={inspectorTab === 'game'} on:click={showGameInspector}>
					<Gamepad2 size={16} aria-hidden="true" /><span>对局</span>
				</button>
				<button class:active={inspectorTab === 'analysis'} on:click={() => inspectorTab = 'analysis'}>
					<BrainCircuit size={16} aria-hidden="true" /><span>AI 分析</span>
				</button>
			</nav>
			<ControlModeBar
				mode={controlMode}
				humanPlayerIndex={humanPlayerIndex}
				players={gs.players}
				currentPlayerIndex={gs.current_player}
				disabled={controlSyncing || $turnPhase === 'in_session' || $aiPlayback.status === 'running' || gs.game_over}
				on:change={handleControlChange}
			/>
			<div class="inspector-content">
				<div class="inspector-view" class:active={inspectorTab === 'game'} aria-hidden={inspectorTab !== 'game'}>
					<Sidebar
						interactionLocked={aiTurn}
						viewerPlayerIndex={handPlayerIndex}
						on:openMat={(e) => {
							matPlayerIndex = e.detail?.playerIndex ?? null;
							matOpen = true;
						}}
						on:openDiscard={(e) => {
							discardPlayerIndex = e.detail?.playerIndex ?? null;
							discardOpen = true;
						}}
					/>
				</div>
				<div class="inspector-view" class:active={inspectorTab === 'analysis'} aria-hidden={inspectorTab !== 'analysis'}>
					<AnalysisPanel {controlMode} {humanPlayerIndex} on:applied={handleAnalysisApplied} />
				</div>
			</div>
		</div>
	</div>
	<IndustryMat bind:open={matOpen} playerIndex={matPlayerIndex} on:close={() => matOpen = false} />
	<DiscardPileViewer bind:open={discardOpen} playerIndex={discardPlayerIndex} on:close={() => discardOpen = false} />
{/if}

<style>
	.game-layout {
		display: flex;
		height: 100svh;
		overflow: hidden;
		background: #d7dbd8;
	}
	.left-col {
		flex: 1;
		display: flex;
		flex-direction: column;
		min-width: 0;
	}
	.board-area {
		flex: 1;
		overflow: hidden;
		position: relative;
		min-height: 0;
		background: #c9cecb;
	}
	.bottom-bar {
		flex-shrink: 0;
		background: #191c1a;
		border-top: 1px solid #343936;
		padding: 4px 16px;
		min-height: 130px;
		transition: background 0.3s;
	}
	.bottom-bar.card-active {
		background: #152a23;
		border-top-color: #2b9b73;
	}
	.card-prompt {
		text-align: center;
		color: #70d7ae;
		font-size: 13px;
		font-weight: 600;
		margin-bottom: 2px;
	}
	.right-col {
		width: clamp(350px, 28vw, 430px);
		flex-shrink: 0;
		display: flex;
		flex-direction: column;
		border-left: 1px solid #afb5b1;
		background: #f5f6f4;
		min-width: 0;
	}
	.inspector-tabs {
		display: grid;
		grid-template-columns: 1fr 1fr;
		min-height: 45px;
		border-bottom: 1px solid #cdd2ce;
		background: #eceeec;
	}
	.inspector-tabs button {
		display: flex;
		align-items: center;
		justify-content: center;
		gap: 7px;
		border: 0;
		border-right: 1px solid #cdd2ce;
		border-bottom: 2px solid transparent;
		background: transparent;
		color: #6a706c;
		font-size: 12px;
		font-weight: 700;
		cursor: pointer;
	}
	.inspector-tabs button:last-child { border-right: 0; }
	.inspector-tabs button.active { color: #202321; border-bottom-color: #087f5b; background: #f5f6f4; }
	.inspector-content {
		flex: 1;
		min-height: 0;
		overflow-y: auto;
	}
	.inspector-view { display: none; min-height: 100%; }
	.inspector-view.active { display: block; }
	@media (max-width: 960px) {
		.right-col { width: 340px; }
	}
	@media (max-width: 760px) {
		.game-layout { height: auto; min-height: 100svh; flex-direction: column; overflow: visible; }
		.left-col { height: 72svh; min-height: 560px; }
		.right-col { width: 100%; min-height: 70svh; border-left: 0; border-top: 1px solid #afb5b1; }
		.inspector-content { overflow: visible; }
	}
</style>
