import { writable, derived, get } from 'svelte/store';
import type { AnalysisReport, GameControlMode, GameState, IndustryLevelData } from './types';

export const gameState = writable<GameState | null>(null);
export type TurnPhase = 'awaiting_start' | 'choosing_action' | 'in_session' | 'turn_done';
export const turnPhase = writable<TurnPhase>('awaiting_start');
export const actionsAvailable = writable<string[] | null>(null);
export const logs = writable<string[]>([]);
export const allIndustryData = writable<Record<string, IndustryLevelData[]> | null>(null);
export const analysisReport = writable<AnalysisReport | null>(null);
export const analysisLoading = writable(false);
export const analysisError = writable<string | null>(null);
export const selectedAnalysisKey = writable<string | null>(null);
export interface AnalysisProgressState {
	completed: number;
	target: number;
	stage: number;
	totalStages: number;
}
export const analysisProgress = writable<AnalysisProgressState | null>(null);
export const analysisAutoEnabled = writable(false);
export const analysisInvalidationVersion = writable(0);

export type AiPlaybackStatus = 'idle' | 'running' | 'paused' | 'complete' | 'error';
export interface AiPlaybackState {
	status: AiPlaybackStatus;
	moves: number;
	error: string | null;
}
export const aiPlayback = writable<AiPlaybackState>({ status: 'idle', moves: 0, error: null });

/** Money each player had at the start of the current player's turn */
export const moneyAtTurnStart = writable<Record<number, number>>({});
/** Currently selected action for this turn (for logging) */
export const currentAction = writable<string | null>(null);
export const actionBudgetAtTurnStart = writable<number>(1);

export function logMessage(msg: string) {
	logs.update(l => [msg, ...l].slice(0, 100));
}

export function snapshotMoney() {
	const gs = get(gameState);
	if (!gs) return;
	const snap: Record<number, number> = {};
	for (const p of gs.players) snap[p.index] = p.money;
	moneyAtTurnStart.set(snap);
}

export const currentPlayer = derived(gameState, $gs => {
	if (!$gs) return null;
	return $gs.players.find(p => p.index === $gs.current_player) ?? null;
});

export const choiceSet = derived(gameState, $gs => $gs?.choice_set ?? null);

export const selectedAnalysisCandidate = derived(
	[analysisReport, selectedAnalysisKey],
	([$report, $key]) => $report?.recommendations.find(candidate => candidate.action_key === $key)
		?? $report?.recommendations[0]
		?? null
);

export const analysisHighlights = derived(
	selectedAnalysisCandidate,
	$candidate => $candidate?.highlights ?? null
);

/** Pending development choices (industry + level) during Develop/Develop x2 action */
export const pendingDevelopments = derived(
	gameState,
	$gs => $gs?.pending_developments ?? []
);

export function playerName(gs: GameState, idx: number): string {
	return gs.players.find(p => p.index === idx)?.name ?? `P${idx}`;
}

export function aiControlsPlayer(
	mode: GameControlMode,
	humanPlayerIndex: number,
	playerIndex: number
): boolean {
	if (mode === 'ai-vs-ai') return true;
	return mode === 'human-vs-ai' && playerIndex !== humanPlayerIndex;
}

export function viewerPlayerIndex(
	mode: GameControlMode,
	humanPlayerIndex: number,
	currentPlayerIndex: number
): number {
	return mode === 'human-vs-ai' ? humanPlayerIndex : currentPlayerIndex;
}

export function invalidateAnalysis() {
	analysisReport.set(null);
	analysisError.set(null);
	analysisLoading.set(false);
	selectedAnalysisKey.set(null);
	analysisProgress.set(null);
	analysisInvalidationVersion.update(version => version + 1);
}

export function resetAiPlayback() {
	aiPlayback.set({ status: 'idle', moves: 0, error: null });
}
