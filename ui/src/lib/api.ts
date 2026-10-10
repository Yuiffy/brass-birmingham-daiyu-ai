import { get } from 'svelte/store';
import { readBrowserSession, saveBrowserSession } from './browserSession';
import {
	gameState, turnPhase, actionsAvailable, logMessage,
	moneyAtTurnStart, snapshotMoney, currentAction, playerName,
	allIndustryData, actionBudgetAtTurnStart, analysisReport, analysisLoading,
	analysisError, analysisProgress, analysisInvalidationVersion,
	selectedAnalysisKey, invalidateAnalysis, resetAiPlayback
} from './store';
import type { AnalysisExplanation, AnalysisReport, GameState, ReplayData } from './types';

const ACTION_LABELS: Record<string, string> = {
	BuildBuilding: 'Build', BuildRailroad: 'Network', BuildDoubleRailroad: 'Double Rail',
	Develop: 'Develop', DevelopDouble: 'Develop x2', Sell: 'Sell',
	Loan: 'Loan', Scout: 'Scout', Pass: 'Pass'
};

const CHOICE_LABELS: Record<string, string> = {
	industry: 'industry', card: 'card', build_location: 'location',
	road: 'road', second_road: '2nd road', coal_source: 'coal source',
	iron_source: 'iron source', beer_source: 'beer source',
	action_beer_source: 'beer', sell_target: 'sell target',
	free_development: 'free develop', second_industry: 'second industry', network_mode: 'network mode', confirm: 'confirm'
};

export interface SavedGameSummary {
	id: number;
	created_at: number;
	round_in_phase: number;
	era: string;
	num_players: number;
	seed: number;
}

let lastTurnPlayer: number | null = null;
let currentTurnActionBudget = 1;
let analysisRunId = 0;

// Browser clicks, map handlers, and AI playback can all issue mutations from
// separate async callbacks.  Keep them in one FIFO so a stale response can
// never overwrite a newer position (and a double-click cannot advance two
// turns before the first response is applied).
const MUTATION_ENDPOINTS = new Set([
	'new_game', 'load_game', 'set_observer', 'start_turn', 'start_action',
	'apply_choice', 'confirm_action', 'cancel_action', 'undo_last_action',
	'end_turn', 'apply_analyzed_action', 'resolve_shortfalls'
]);
let mutationQueue: Promise<void> = Promise.resolve();
const browserSessions = import.meta.env.VITE_BRASS_BROWSER_SESSIONS === 'true';
let browserSession: unknown;

async function getBrowserSession() {
	if (browserSession === undefined) {
		browserSession = await readBrowserSession();
	}
	return browserSession;
}

function cpName(): string {
	const gs = get(gameState);
	return gs ? playerName(gs, gs.current_player) : '???';
}

export function applyLoadedState(state: GameState) {
	gameState.set(state);
	currentAction.set(null);
	lastTurnPlayer = state.current_player;
	currentTurnActionBudget = Math.max(1, state.actions_remaining);
	actionBudgetAtTurnStart.set(currentTurnActionBudget);
	if (state.choice_set) {
		turnPhase.set('in_session');
		actionsAvailable.set(null);
	} else if (state.actions_remaining <= 0) {
		turnPhase.set('turn_done');
		actionsAvailable.set(null);
	} else {
		turnPhase.set('choosing_action');
		actionsAvailable.set(state.available_actions || []);
	}
	loadIndustryData();
}

export async function api(endpoint: string, body?: unknown): Promise<any> {
	const request = async () => {
		const opts: RequestInit = { method: body !== undefined ? 'POST' : 'GET' };
		if (body !== undefined) {
			opts.headers = { 'Content-Type': 'application/json' };
			opts.body = JSON.stringify(body);
		}
		const postNoBody = ['start_turn', 'confirm_action', 'cancel_action', 'undo_last_action', 'end_turn'];
		if (!opts.method || (opts.method === 'GET' && postNoBody.includes(endpoint))) {
			opts.method = 'POST';
		}
		try {
			const res = browserSessions
				? await fetch('/api/browser_request', {
					method: 'POST',
					headers: { 'Content-Type': 'application/json' },
					body: JSON.stringify({ endpoint, body: body ?? null, session: await getBrowserSession() })
				})
				: await fetch('/api/' + endpoint, opts);
			if (!res.ok) { logMessage(`Server error: ${res.status}`); return null; }
			const text = await res.text();
			if (!text) { logMessage('Empty response'); return null; }
			const data = JSON.parse(text);
			if (!data.ok) { logMessage('Error: ' + (data.error || 'unknown')); return null; }
			if (browserSessions && data.browser_session) {
				await saveBrowserSession(data.browser_session);
				browserSession = data.browser_session;
			}
			if (data.state) gameState.set(data.state);
			return data;
		} catch (e: any) {
			logMessage('Network error: ' + e.message);
			return null;
		}
	};

	if (!browserSessions && !MUTATION_ENDPOINTS.has(endpoint)) return request();
	const queued = mutationQueue.then(request, request);
	// Always release the queue, including failed requests.  `request` already
	// converts transport/application errors to null for its caller.
	mutationQueue = queued.then(() => undefined, () => undefined);
	return queued;
}

export async function loadIndustryData() {
	try {
		const res = await fetch('/api/industry_data');
		const json = await res.json();
		if (json.ok && json.data) allIndustryData.set(json.data);
	} catch { /* non-critical */ }
}

export async function listSavedGames(): Promise<SavedGameSummary[]> {
	const data = await api('games');
	return data?.games ?? [];
}

export async function loadReplay(gameId: number): Promise<ReplayData | null> {
	const data = await api('replay', { game_id: gameId });
	if (!data?.positions) return null;
	return {
		game_id: data.game_id as number,
		positions: data.positions as ReplayData['positions'],
		final_state: (data.final_state as GameState | null | undefined) ?? null
	};
}

export async function loadGame(gameId: number) {
	resetAiPlayback();
	invalidateAnalysis();
	const data = await api('load_game', { game_id: gameId });
	if (data?.state) {
		applyLoadedState(data.state as GameState);
		logMessage(`Loaded game #${gameId}`);
	}
	return data;
}

export async function newGame(numPlayers: number, seed: number | null = null) {
	resetAiPlayback();
	invalidateAnalysis();
	const data = await api('new_game', { num_players: numPlayers, seed });
	if (data?.state) {
		applyLoadedState(data.state as GameState);
		turnPhase.set('awaiting_start');
		actionsAvailable.set(null);
		lastTurnPlayer = null;
		currentTurnActionBudget = 1;
		actionBudgetAtTurnStart.set(1);
		logMessage(`New game started with ${numPlayers} players`);
	}
	return data;
}

export async function setObserverPlayer(playerIndex: number | null) {
	// Observer changes alter which private hand and action context the UI represents.
	invalidateAnalysis();
	const data = await api('set_observer', { player_index: playerIndex });
	if (data?.state && get(turnPhase) === 'choosing_action') {
		actionsAvailable.set(data.state.available_actions ?? []);
	}
	return data;
}

export async function startTurn() {
	invalidateAnalysis();
	const name = cpName();
	snapshotMoney();
	const phaseBeforeStart = get(turnPhase);
	const data = await api('start_turn');
	if (data?.state) {
		const state = data.state;
		const currentPlayer = state.current_player as number;
		const remaining = state.actions_remaining as number;
		const isNewPersonalTurn = lastTurnPlayer !== currentPlayer || phaseBeforeStart === 'awaiting_start';
		if (isNewPersonalTurn) {
			currentTurnActionBudget = Math.max(1, remaining);
			lastTurnPlayer = currentPlayer;
		}
		actionBudgetAtTurnStart.set(currentTurnActionBudget);
		actionsAvailable.set(data.state.available_actions || []);
		turnPhase.set('choosing_action');
		currentAction.set(null);
		logMessage(`${name}'s turn begins (£${data.state.players.find((p: any) => p.index === data.state.current_player)?.money ?? '?'})`);
	}
	return data;
}

export async function selectAction(actionType: string) {
	invalidateAnalysis();
	const name = cpName();
	const label = ACTION_LABELS[actionType] || actionType;
	const data = await api('start_action', { action_type: actionType });
	if (data) {
		actionsAvailable.set(null);
		turnPhase.set('in_session');
		currentAction.set(actionType);
		logMessage(`${name} chose action: ${label}`);
	}
	return data;
}

export async function applyChoice(kind: string, value: unknown) {
	invalidateAnalysis();
	const name = cpName();
	const gs = get(gameState);
	const kindLabel = CHOICE_LABELS[kind] || kind;
	let detail = String(value);

	if (kind === 'card' && gs) {
		const cp = gs.players.find(p => p.index === gs.current_player);
		const card = cp?.hand.find(c => c.index === value);
		if (card) detail = card.label;
	}
	if (kind === 'build_location' || kind === 'sell_target') {
		const cs = gs?.choice_set;
		const opt = cs?.options.find(o => o.value === value);
		if (opt) detail = opt.label;
	}

	const data = await api('apply_choice', { choice_kind: kind, value });
	if (data) {
		logMessage(`${name} picked ${kindLabel}: ${detail}`);
	}
	return data;
}

export async function confirmAction() {
	invalidateAnalysis();
	const name = cpName();
	const action = get(currentAction);
	const actionLabel = action ? (ACTION_LABELS[action] || action) : 'action';
	const moneyBefore = get(moneyAtTurnStart);
	const data = await api('confirm_action');
	if (data) {
		const gs = data.state;
		const cp = gs.players.find((p: any) => p.index === gs.current_player);
		const spent = moneyBefore[gs.current_player] != null
			? moneyBefore[gs.current_player] - (cp?.money ?? 0)
			: 0;
		const spentStr = spent > 0 ? ` (spent £${spent})` : spent < 0 ? ` (gained £${-spent})` : '';
		logMessage(`${name} confirmed ${actionLabel}${spentStr}`);
		currentAction.set(null);
		if (gs.actions_remaining > 0) {
			return startTurn();
		} else {
			turnPhase.set('turn_done');
		}
	}
	return data;
}

export async function cancelAction() {
	invalidateAnalysis();
	const name = cpName();
	const action = get(currentAction);
	const actionLabel = action ? (ACTION_LABELS[action] || action) : 'action';
	const data = await api('cancel_action');
	if (data) {
		logMessage(`${name} cancelled ${actionLabel}`);
		currentAction.set(null);
		return startTurn();
	}
	return data;
}

export async function undoLastAction() {
	invalidateAnalysis();
	const name = cpName();
	const data = await api('undo_last_action');
	if (data) {
		turnPhase.set('choosing_action');
		currentAction.set(null);
		actionsAvailable.set(data.state?.available_actions || []);
		logMessage(`${name} undid previous confirmed action`);
	}
	return data;
}

export async function endTurn() {
	invalidateAnalysis();
	const name = cpName();
	const moneyBefore = get(moneyAtTurnStart);
	const gs = get(gameState);
	let spentStr = '';
	if (gs) {
		const cp = gs.players.find(p => p.index === gs.current_player);
		if (cp && moneyBefore[cp.index] != null) {
			const delta = moneyBefore[cp.index] - cp.money;
			if (delta > 0) spentStr = ` — spent £${delta} total this turn`;
			else if (delta < 0) spentStr = ` — gained £${-delta} total this turn`;
		}
	}
	const data = await api('end_turn');
	if (data) {
		actionsAvailable.set(null);
		turnPhase.set('awaiting_start');
		currentAction.set(null);
		lastTurnPlayer = null;
		logMessage(`${name} ended turn${spentStr}`);
	}
	return data;
}

export type AnalysisMode = 'trained' | 'auto' | 'rule';

function analysisStages(simulations: number, mode: AnalysisMode): number[] {
	if (mode !== 'auto') return [1];
	return [...new Set([100, 400, 800, 3000, simulations])]
		.filter(stage => stage > 0 && stage <= simulations)
		.sort((left, right) => left - right);
}

export async function analyzePosition(
	simulations: number,
	mode: AnalysisMode = 'trained'
): Promise<AnalysisReport | null> {
	return analyzePositionProgressive(simulations, mode);
}

export async function analyzePositionProgressive(
	simulations: number,
	mode: AnalysisMode = 'trained'
): Promise<AnalysisReport | null> {
	const runId = ++analysisRunId;
	const invalidationVersion = get(analysisInvalidationVersion);
	const stages = analysisStages(simulations, mode);
	const progressTarget = mode !== 'auto' ? 1 : simulations;
	analysisLoading.set(true);
	analysisError.set(null);
	analysisProgress.set({
		completed: 0,
		target: progressTarget,
		stage: 0,
		totalStages: stages.length
	});
	let latestReport: AnalysisReport | null = null;
	const isCurrent = () => runId === analysisRunId
		&& get(analysisInvalidationVersion) === invalidationVersion;
	try {
		for (const [index, progressTo] of stages.entries()) {
			if (!isCurrent()) return null;
			analysisProgress.set({
				completed: latestReport?.completed_simulations ?? 0,
				target: progressTarget,
				stage: index + 1,
				totalStages: stages.length
			});
			const data = await api('analyze', {
				simulations: mode !== 'auto' ? 1 : simulations,
				progress_to: progressTo,
				top_n: 3,
				mode
			});
			if (!isCurrent()) return null;
			if (!data?.analysis) {
				analysisError.set('分析失败，请确认当前处于可行动状态');
				return null;
			}
			const report = data.analysis as AnalysisReport;
			latestReport = report;
			analysisReport.set(report);
			const selectedKey = get(selectedAnalysisKey);
			if (!selectedKey || !report.recommendations.some(candidate => candidate.action_key === selectedKey)) {
				selectedAnalysisKey.set(report.recommendations[0]?.action_key ?? null);
			}
			analysisProgress.set({
				completed: report.completed_simulations,
				target: progressTarget,
				stage: index + 1,
				totalStages: stages.length
			});
		}
		if (latestReport) {
			if (mode === 'trained') {
				logMessage(`学习增强 v2 评估了 ${latestReport.evaluated_action_count} 个候选动作，用时 ${latestReport.elapsed_ms}ms`);
			} else if (mode === 'rule') {
				logMessage(`CPU 规则树评估了 ${latestReport.root_action_count} 个合法动作，用时 ${latestReport.elapsed_ms}ms`);
			} else {
				logMessage(`AI analyzed ${latestReport.completed_simulations} continuations in ${latestReport.elapsed_ms}ms`);
			}
		}
		return latestReport;
	} catch (error) {
		if (isCurrent()) {
			analysisError.set(error instanceof Error ? error.message : String(error));
		}
		return null;
	} finally {
		if (isCurrent()) analysisLoading.set(false);
	}
}

export async function explainAnalyzedAction(
	report: AnalysisReport,
	actionKey: string,
	question: string
): Promise<AnalysisExplanation | null> {
	const data = await api('explain', {
		revision: report.revision,
		action_key: actionKey,
		question
	});
	return (data?.explanation as AnalysisExplanation | undefined) ?? null;
}

export async function applyAnalyzedAction(report: AnalysisReport, actionKey: string) {
	const data = await api('apply_analyzed_action', {
		revision: report.revision,
		action_key: actionKey
	});
	if (!data?.state) return null;

	const state = data.state as GameState;
	currentAction.set(null);
	if (state.game_over) {
		turnPhase.set('turn_done');
		actionsAvailable.set(null);
	} else if (state.choice_set) {
		turnPhase.set('in_session');
		actionsAvailable.set(null);
	} else if (state.actions_remaining > 0) {
		turnPhase.set('choosing_action');
		actionsAvailable.set(state.available_actions ?? []);
	} else {
		turnPhase.set('turn_done');
		actionsAvailable.set(null);
	}
	logMessage(`Applied AI recommendation: ${actionKey}`);
	invalidateAnalysis();
	return data;
}

export async function resolveShortfalls() {
	const data = await api('resolve_shortfalls', {});
	if (!data?.state) return null;
	applyLoadedState(data.state as GameState);
	const resolved = Array.isArray(data.resolved) ? data.resolved.length : 0;
	if (resolved > 0) logMessage(`AI resolved ${resolved} income shortfall${resolved === 1 ? '' : 's'}`);
	return data;
}
