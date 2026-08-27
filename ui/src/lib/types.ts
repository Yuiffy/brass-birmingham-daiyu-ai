export type GameControlMode = 'manual' | 'human-vs-ai' | 'ai-vs-ai';

export interface PendingDevelopment {
	industry: string;
	level: number;
}

export interface TurnActionView {
	action_type: string;
	selections: string[];
}

export interface GameState {
	seed: number;
	current_player: number;
	phase: string;
	era: string;
	turn_count: number;
	round_in_phase: number;
	actions_remaining: number;
	players: Player[];
	buildings: Building[];
	roads: Road[];
	coal_market: number;
	iron_market: number;
	trade_posts: TradePostSlot[];
	available_actions: string[] | null;
	choice_set: ChoiceSet | null;
	pending_developments?: PendingDevelopment[];
	turn_action_history?: TurnActionView[];
	current_action_selections?: TurnActionView | null;
	discard_history?: DiscardEntry[];
	has_pending_shortfall: boolean;
	game_over: boolean;
	turn_order: number[];
}

export interface DiscardEntry {
	order: number;
	player_index: number;
	player_name: string;
	round_in_phase: number;
	turn_count: number;
	card_type: string;
	card_label: string;
}

export interface Player {
	index: number;
	name: string;
	color: string;
	money: number;
	income_level: number;
	income_amount: number;
	victory_points: number;
	hand: Card[];
	hand_size: number;
	industry_mat: IndustryTile[];
}

export interface IndustryTile {
	industry: string;
	level: number;
	tiles_remaining: number;
	money_cost: number;
	coal_cost: number;
	iron_cost: number;
	beer_needed: number;
	vp_on_flip: number;
	road_vp: number;
	resource_amt: number;
	income: number;
	removed_after_phase1: boolean;
	can_develop: boolean;
	exhausted: boolean;
}

export interface IndustryLevelData {
	industry: string;
	level: number;
	money_cost: number;
	coal_cost: number;
	iron_cost: number;
	beer_needed: number;
	vp_on_flip: number;
	road_vp: number;
	resource_amt: number;
	income: number;
	removed_after_phase1: boolean;
	can_develop: boolean;
	num_tiles: number;
}

export interface Card {
	index: number;
	card_type: string;
	label: string;
}

export interface Building {
	location: number;
	town: string;
	industry: string;
	level: number;
	owner: number;
	flipped: boolean;
	resource_amt: number;
	road_vp: number;
	vp_on_flip: number;
}

export interface Road {
	index: number;
	owner: number;
}

export interface TradePostSlot {
	slot_index: number;
	trade_post: string;
	tile_type: string | null;
	has_beer: boolean;
}

export interface ChoiceSet {
	kind: string;
	options: ChoiceOption[];
}

export interface ChoiceOption {
	value: any;
	label: string;
}

export interface AnalysisReport {
	revision: number;
	method: string;
	method_label: string;
	value_source: string;
	model_id: string | null;
	root_model_shared_win_rate: number | null;
	root_model_victory_point_margin: number | null;
	root_player: number;
	requested_simulations: number;
	completed_simulations: number;
	root_action_count: number;
	evaluated_action_count: number;
	visited_action_count: number;
	all_root_actions_evaluated: boolean;
	max_search_depth: number | null;
	neural_leaf_evaluations: number | null;
	inference_batches: number | null;
	coverage: number;
	elapsed_ms: number;
	recommendations: AnalysisCandidate[];
}

export interface AnalysisCandidate {
	rank: number;
	action_key: string;
	action_type: string;
	action_label: string;
	summary: string;
	selections: string[];
	visits: number;
	visit_share: number;
	value_source: string;
	value_sample_count: number;
	estimated_shared_win_rate: number;
	estimated_outright_win_rate: number | null;
	estimated_tied_first_rate: number | null;
	average_final_victory_points: number | null;
	average_victory_point_margin: number;
	shared_win_rate_standard_error: number | null;
	policy_probability: number | null;
	calibrated_win_rate: number | null;
	immediate_effect: AnalysisImmediateEffect;
	highlights: AnalysisHighlights;
	sample_random_continuation: AnalysisContinuation;
}

export interface AnalysisImmediateEffect {
	money_delta: number;
	income_level_delta: number;
	victory_points_delta: number;
	visible_victory_points_delta: number;
	potential_era_victory_points_delta: number;
	hand_size_delta: number;
	round_spend_delta: number;
	roads_on_board_delta: number;
	buildings_on_board_delta: number;
	flipped_buildings_delta: number;
	draw_deck_size_delta: number;
	market_coal_delta: number;
	market_iron_delta: number;
	wild_location_pool_delta: number;
	wild_industry_pool_delta: number;
	turn_advanced: boolean;
	phase_changed: boolean;
}

export interface AnalysisHighlights {
	building_locations: number[];
	source_locations: number[];
	roads: number[];
	merchant_slots: number[];
}

export interface AnalysisContinuation {
	label: string;
	steps: AnalysisContinuationStep[];
	final_victory_points: number[];
	official_winners: number[];
}

export interface AnalysisContinuationStep {
	action_number: number;
	player_idx: number;
	action_type: string;
	action_label: string;
	action_key: string;
}

export interface AnalysisExplanation {
	revision: number;
	action_key: string;
	question: string;
	answer: string;
	evidence: string[];
	caveat: string;
}

export interface ReplayMove {
	player_idx: number;
	action_type: string;
	action_key: string | null;
	selections: string[];
}

export interface ReplayPosition {
	index: number;
	position_key: string;
	player_idx: number;
	state: GameState;
	analysis: AnalysisReport | null;
	move: ReplayMove | null;
}

export interface ReplayData {
	game_id: number;
	positions: ReplayPosition[];
	final_state: GameState | null;
}
