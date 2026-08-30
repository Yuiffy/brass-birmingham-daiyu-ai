use crate::board::connectivity::Connectivity;
use crate::board::Board;
use crate::consts::STARTING_HAND_SIZE;
use crate::core::locations::LocationName;
use crate::core::player::Player;
use crate::core::static_data::{INDUSTRY_MAT, LINK_LOCATIONS};
use crate::core::types::*;
use crate::game::framework::{
    ActionChoice, ActionIntent, ChoiceSet, GameFramework, ShortfallResolutionSession,
};

#[derive(Debug, Clone)]
struct TurnCheckpoint {
    board_state: crate::board::state::BoardState,
    actions_remaining_in_turn: u8,
    discard_history_len: usize,
}

#[derive(Clone)]
pub struct ReplayTurnCheckpoint {
    board_state: crate::board::state::BoardState,
    game_phase: GamePhase,
    turn_count: u32,
    round_in_phase: u32,
    actions_remaining_in_turn: u8,
    personal_turns_taken: Vec<u32>,
    pending_shortfall_sessions: Vec<ShortfallResolutionSession>,
    current_player: usize,
    turn_started: bool,
    discard_history: Vec<DiscardHistoryEntry>,
}

#[derive(Debug, Clone)]
pub struct DiscardHistoryEntry {
    pub order: usize,
    pub player_idx: usize,
    pub round_in_phase: u32,
    pub turn_count: u32,
    pub card: Card,
}

#[derive(Clone)]
pub struct GameRunner {
    pub framework: GameFramework,
    pub game_phase: GamePhase,
    pub turn_count: u32,
    pub round_in_phase: u32,
    pub actions_remaining_in_turn: u8,
    /// Whether the current player's turn has been explicitly started.
    ///
    /// `actions_remaining_in_turn == 0` is ambiguous: it is true both before
    /// a turn starts and after its final action.  Keeping this bit separate
    /// makes an `end_turn` request idempotent instead of allowing a duplicate
    /// request to advance the next player.
    pub turn_started: bool,
    pub personal_turns_taken: Vec<u32>,
    pub pending_shortfall_sessions: Vec<ShortfallResolutionSession>,
    turn_checkpoints: Vec<TurnCheckpoint>,
    turn_action_history: Vec<ActionIntent>,
    discard_history: Vec<DiscardHistoryEntry>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GamePhase {
    Canal,
    Railroad,
    GameEnd,
}

impl GameRunner {
    pub fn new(num_players: usize, seed: Option<u64>) -> Self {
        let board = Board::new(num_players, seed);
        let current_player = board.state.turn_order[0];
        let framework = GameFramework::new(board, current_player);
        Self {
            framework,
            game_phase: GamePhase::Canal,
            turn_count: 0,
            round_in_phase: 0,
            actions_remaining_in_turn: 0,
            turn_started: false,
            personal_turns_taken: vec![0; num_players],
            pending_shortfall_sessions: Vec::new(),
            turn_checkpoints: Vec::new(),
            turn_action_history: Vec::new(),
            discard_history: Vec::new(),
        }
    }

    pub fn start_turn(&mut self) -> Vec<ActionType> {
        let player_idx = self.framework.current_player;
        if !self.turn_started {
            // A few low-level callers restore a position with a non-zero
            // action budget directly.  Treat that as an already-open turn.
            if self.actions_remaining_in_turn == 0 {
                self.actions_remaining_in_turn = if self.personal_turns_taken[player_idx] == 0 {
                    1
                } else {
                    2
                };
                self.turn_checkpoints.clear();
                self.turn_action_history.clear();
            }
            self.turn_started = true;
        }
        self.framework.get_valid_root_actions()
    }

    pub fn start_action(&mut self, action_type: ActionType) -> ChoiceSet {
        let _ = self.framework.start_action_session(action_type);
        self.framework
            .get_next_choice_set()
            .unwrap_or(ChoiceSet::ConfirmOnly)
    }

    /// Strict Web/API entry point for opening an action session.
    pub fn try_start_action(&mut self, action_type: ActionType) -> Result<ChoiceSet, String> {
        if self.is_game_finished() {
            return Err("Game is already finished".to_string());
        }
        if !self.turn_started {
            return Err("Turn has not been started".to_string());
        }
        if self.actions_remaining_in_turn == 0 {
            return Err("No actions remain in this turn".to_string());
        }
        if self.has_pending_shortfall() {
            return Err("Income shortfall must be resolved before an action".to_string());
        }
        if self.framework.current_session().is_some() {
            return Err("An action session is already active".to_string());
        }
        let root_action = if action_type == ActionType::BuildDoubleRailroad {
            ActionType::BuildRailroad
        } else {
            action_type
        };
        if !self
            .framework
            .get_valid_root_actions()
            .contains(&root_action)
        {
            return Err(format!("Action {:?} is not legal", action_type));
        }
        Ok(self.start_action(action_type))
    }

    pub fn apply_choice(&mut self, choice: ActionChoice) -> Option<ChoiceSet> {
        let _ = self.framework.apply_action_choice(choice);
        self.framework.get_next_choice_set()
    }

    /// Strict Web/API entry point that validates the currently advertised
    /// choice before mutating the action session.
    pub fn try_apply_choice(&mut self, choice: ActionChoice) -> Result<Option<ChoiceSet>, String> {
        if self.is_game_finished() {
            return Err("Game is already finished".to_string());
        }
        if !self.turn_started {
            return Err("Turn has not been started".to_string());
        }
        let choice_set = self
            .framework
            .get_next_choice_set()
            .ok_or_else(|| "No active action choice is pending".to_string())?;
        if !crate::game::legal_actions::choice_set_contains(&choice_set, &choice) {
            return Err(format!(
                "Choice {:?} is not legal for the current choice set {:?}",
                choice, choice_set
            ));
        }
        self.framework.apply_action_choice(choice)?;
        Ok(self.framework.get_next_choice_set())
    }

    pub fn confirm_action(&mut self) -> Result<(), String> {
        let session = self
            .framework
            .current_session()
            .ok_or_else(|| "No active action session".to_string())?;
        let discard_len_before = self.framework.board.state.discard_pile.len();
        self.turn_checkpoints.push(TurnCheckpoint {
            board_state: self.framework.board.state.clone(),
            actions_remaining_in_turn: self.actions_remaining_in_turn,
            discard_history_len: self.discard_history.len(),
        });
        if let Err(error) = self.framework.confirm_action_session() {
            let checkpoint = self
                .turn_checkpoints
                .pop()
                .expect("checkpoint was pushed immediately before confirmation");
            self.framework.board.state = checkpoint.board_state;
            self.actions_remaining_in_turn = checkpoint.actions_remaining_in_turn;
            self.discard_history
                .truncate(checkpoint.discard_history_len);
            return Err(error);
        }
        let discard_len_after = self.framework.board.state.discard_pile.len();
        if discard_len_after > discard_len_before {
            for card in self.framework.board.state.discard_pile
                [discard_len_before..discard_len_after]
                .iter()
                .cloned()
            {
                let entry = DiscardHistoryEntry {
                    order: self.discard_history.len(),
                    player_idx: self.framework.current_player,
                    round_in_phase: self.round_in_phase,
                    turn_count: self.turn_count,
                    card,
                };
                self.discard_history.push(entry);
            }
        }
        self.turn_action_history.push(session.intent);
        self.end_action_slot();
        Ok(())
    }

    pub fn undo_last_confirmed_action(&mut self) -> Result<(), String> {
        let Some(checkpoint) = self.turn_checkpoints.pop() else {
            return Err("No previously confirmed action to undo this turn".to_string());
        };
        self.framework.cancel_action_session();
        self.framework.board.state = checkpoint.board_state;
        self.actions_remaining_in_turn = checkpoint.actions_remaining_in_turn;
        self.discard_history
            .truncate(checkpoint.discard_history_len);
        self.turn_action_history.pop();
        Ok(())
    }

    pub fn turn_action_history(&self) -> &Vec<ActionIntent> {
        &self.turn_action_history
    }

    pub fn discard_history(&self) -> &Vec<DiscardHistoryEntry> {
        &self.discard_history
    }

    pub fn end_action_slot(&mut self) {
        if self.actions_remaining_in_turn > 0 {
            self.actions_remaining_in_turn -= 1;
        }
        self.draw_cards_for_player(self.framework.current_player);

        let state = &self.framework.board.state;
        if state.deck.is_empty()
            && state.players[self.framework.current_player]
                .hand
                .cards
                .is_empty()
        {
            self.actions_remaining_in_turn = 0;
        }
    }

    pub fn end_turn(&mut self) {
        // Keep this low-level method safe as well: all normal callers should
        // have consumed the action budget and opened exactly one turn.
        if !self.turn_started
            || self.actions_remaining_in_turn != 0
            || self.framework.current_session().is_some()
            || self.has_pending_shortfall()
        {
            return;
        }
        self.turn_checkpoints.clear();
        self.turn_action_history.clear();
        self.finish_turn_and_advance();
    }

    /// Validate and finish a browser/API turn.  Duplicate calls return an
    /// error rather than advancing the next seat.
    pub fn try_end_turn(&mut self) -> Result<(), String> {
        if self.is_game_finished() {
            return Err("Game is already finished".to_string());
        }
        if !self.turn_started {
            return Err("No active turn to end".to_string());
        }
        if self.actions_remaining_in_turn != 0 {
            return Err(format!(
                "Cannot end turn with {} action(s) remaining",
                self.actions_remaining_in_turn
            ));
        }
        if self.framework.current_session().is_some() {
            return Err("Finish or cancel the active action before ending the turn".to_string());
        }
        if self.has_pending_shortfall() {
            return Err("Resolve income shortfall before ending the turn".to_string());
        }
        self.end_turn();
        if self.turn_started {
            return Err("Turn did not advance".to_string());
        }
        Ok(())
    }

    /// Replay-only turn advancement.  Historical logs may contain malformed
    /// boundary markers; preserve their exact old semantics while the audit
    /// layer reports the anomaly.  Normal callers must use `try_end_turn`.
    pub fn end_turn_for_replay(&mut self) {
        self.turn_started = true;
        self.turn_checkpoints.clear();
        self.turn_action_history.clear();
        self.finish_turn_and_advance();
    }

    pub fn checkpoint_replay_turn(&self) -> ReplayTurnCheckpoint {
        ReplayTurnCheckpoint {
            board_state: self.framework.board.state.clone(),
            game_phase: self.game_phase,
            turn_count: self.turn_count,
            round_in_phase: self.round_in_phase,
            actions_remaining_in_turn: self.actions_remaining_in_turn,
            personal_turns_taken: self.personal_turns_taken.clone(),
            pending_shortfall_sessions: self.pending_shortfall_sessions.clone(),
            current_player: self.framework.current_player,
            turn_started: self.turn_started,
            discard_history: self.discard_history.clone(),
        }
    }

    pub fn restore_replay_turn(&mut self, checkpoint: ReplayTurnCheckpoint) {
        self.framework.board.state = checkpoint.board_state;
        self.game_phase = checkpoint.game_phase;
        self.turn_count = checkpoint.turn_count;
        self.round_in_phase = checkpoint.round_in_phase;
        self.actions_remaining_in_turn = checkpoint.actions_remaining_in_turn;
        self.personal_turns_taken = checkpoint.personal_turns_taken;
        self.pending_shortfall_sessions = checkpoint.pending_shortfall_sessions;
        self.framework.current_player = checkpoint.current_player;
        self.turn_started = checkpoint.turn_started;
        self.discard_history = checkpoint.discard_history;
        self.framework.cancel_action_session();
        self.turn_checkpoints.clear();
        self.turn_action_history.clear();
    }

    pub fn finish_turn_and_advance(&mut self) {
        self.turn_started = false;
        let player_idx = self.framework.current_player;
        self.personal_turns_taken[player_idx] += 1;
        self.turn_count += 1;

        let current_turn_pos = self
            .framework
            .board
            .state
            .turn_order
            .iter()
            .position(|p| *p == player_idx)
            .unwrap_or(0);
        let turn_order_len = self.framework.board.state.turn_order.len();
        let mut next_turn_pos = current_turn_pos + 1;
        if self.framework.board.state.deck.is_empty() {
            while next_turn_pos < turn_order_len {
                let candidate = self.framework.board.state.turn_order[next_turn_pos];
                if !self.framework.board.state.players[candidate]
                    .hand
                    .cards
                    .is_empty()
                {
                    break;
                }
                next_turn_pos += 1;
            }
        }

        if next_turn_pos >= turn_order_len {
            self.end_round();
        } else {
            self.framework.current_player = self.framework.board.state.turn_order[next_turn_pos];
        }
    }

    pub fn end_round(&mut self) {
        self.round_in_phase += 1;

        let era_ending = self.framework.board.state.deck.is_empty()
            && self
                .framework
                .board
                .state
                .players
                .iter()
                .all(|player| player.hand.cards.is_empty());
        let final_round = era_ending && self.game_phase == GamePhase::Railroad;

        if !final_round {
            let num_players = self.framework.board.state.players.len();
            for player_idx in 0..num_players {
                self.resolve_income_shortfall_for_player(player_idx);
            }
        }

        self.framework.board.turn_order_next();
        for player in &mut self.framework.board.state.players {
            player.spent_this_turn = 0;
        }
        self.try_phase_transition();

        if !self.is_game_finished() {
            if let Some(first_player) =
                self.framework
                    .board
                    .state
                    .turn_order
                    .iter()
                    .copied()
                    .find(|player_idx| {
                        !self.framework.board.state.deck.is_empty()
                            || !self.framework.board.state.players[*player_idx]
                                .hand
                                .cards
                                .is_empty()
                    })
            {
                self.framework.current_player = first_player;
            }
        }
    }

    fn draw_cards_for_player(&mut self, player_idx: usize) {
        let hand_limit = STARTING_HAND_SIZE as usize;
        let current_hand_size = self.framework.board.state.players[player_idx]
            .hand
            .cards
            .len();
        if current_hand_size < hand_limit {
            let cards_needed = hand_limit - current_hand_size;
            let drawn = self.framework.board.state.deck.draw_n(cards_needed);
            for card in drawn {
                self.framework.board.state.players[player_idx]
                    .hand
                    .add_card(card);
            }
        }
    }

    fn try_phase_transition(&mut self) {
        let era_ending = self.framework.board.state.deck.is_empty()
            && self
                .framework
                .board
                .state
                .players
                .iter()
                .all(|player| player.hand.cards.is_empty());
        if !era_ending {
            return;
        }

        if self.game_phase == GamePhase::Canal {
            self.game_phase = GamePhase::Railroad;
            self.framework.board.state.era = Era::Railroad;
            self.round_in_phase = 0;
            self.end_era();
            self.prepare_railroad_cards_and_merchants();
        } else if self.game_phase == GamePhase::Railroad {
            self.game_phase = GamePhase::GameEnd;
            self.end_era();
        }
    }

    fn prepare_railroad_cards_and_merchants(&mut self) {
        let state = &mut self.framework.board.state;
        state.trade_post_beer.clear();
        for (slot_idx, merchant) in state.trade_post_slots.iter().enumerate() {
            if merchant.as_ref().is_some_and(|tile| {
                tile.tile_type != crate::market::merchants::MerchantTileType::Blank
            }) {
                state.trade_post_beer.insert(slot_idx);
            }
        }

        let discarded_cards = std::mem::take(&mut state.discard_pile);
        state.deck.reshuffle_with_cards(discarded_cards);
        for player_idx in 0..state.players.len() {
            let cards = state.deck.draw_n(STARTING_HAND_SIZE as usize);
            state.players[player_idx].hand.cards = cards;
        }
    }

    pub fn end_era(&mut self) {
        let state = &mut self.framework.board.state;
        let num_players = state.players.len();

        // Score roads: for each road, the owner gets VPs equal to the sum
        // of road_vp for every building present in the road's connected locations.
        let built_road_indices: Vec<usize> = state.built_roads.ones().collect();
        for road_idx in built_road_indices {
            let mut road_owner = None;
            for p_idx in 0..num_players {
                if state.player_road_mask[p_idx].contains(road_idx) {
                    road_owner = Some(p_idx);
                    break;
                }
            }

            if let Some(owner_idx) = road_owner {
                let mut road_vps: u16 = 0;
                for loc_idx in LINK_LOCATIONS[road_idx].locations.ones() {
                    let bl_set = LocationName::from_usize(loc_idx).to_bl_set();
                    for bl_idx in bl_set.ones() {
                        if let Some(building) = state
                            .bl_to_building
                            .get(&bl_idx)
                            .filter(|building| building.flipped)
                        {
                            let data = &INDUSTRY_MAT[building.industry as usize]
                                [building.level.as_usize()];
                            road_vps += data.road_vp as u16;
                        }
                    }
                }
                state.players[owner_idx].victory_points += road_vps;
                state.visible_vps[owner_idx] += road_vps;
            }
        }

        // Score all flipped buildings: each flipped building awards its
        // vp_on_flip to the owning player.
        for building in state.bl_to_building.values() {
            if building.flipped {
                let data = &INDUSTRY_MAT[building.industry as usize][building.level.as_usize()];
                let owner_idx = building.owner.as_usize();
                state.players[owner_idx].victory_points += data.vp_on_flip as u16;
                state.visible_vps[owner_idx] += data.vp_on_flip as u16;
            }
        }

        // Canal-to-Railroad transition: remove level 1 tiles and all roads,
        // then reset connectivity.
        if self.game_phase == GamePhase::Railroad {
            let to_remove: Vec<usize> = state
                .bl_to_building
                .iter()
                .filter(|(_, b)| {
                    INDUSTRY_MAT[b.industry as usize][b.level.as_usize()].removed_after_phase1
                })
                .map(|(&loc, _)| loc)
                .collect();

            for loc_idx in to_remove {
                state.remove_building_from_board(loc_idx);
            }

            state.built_roads.clear();
            for p_idx in 0..num_players {
                state.player_road_mask[p_idx].clear();
            }

            state.connectivity = Connectivity::new();
            for p_idx in 0..num_players {
                state.player_network_mask[p_idx] = Connectivity::new();
                state.players[p_idx].network = Connectivity::new();
            }
        }
    }

    pub fn resolve_income_shortfall_for_player(&mut self, player_idx: usize) {
        let player = &self.framework.board.state.players[player_idx];
        let income = player.get_income_amount(player.income_level);
        if income >= 0 {
            self.framework.board.state.players[player_idx].gain_money(income as u16);
            return;
        }

        let debt = income.unsigned_abs() as u16;
        if player.money >= debt {
            self.framework.board.state.players[player_idx].money =
                self.framework.board.state.players[player_idx]
                    .money
                    .saturating_sub(debt);
            return;
        }

        let shortfall = debt - player.money;
        self.framework.board.state.players[player_idx].money = 0;
        let session = self
            .framework
            .start_shortfall_resolution_session(player_idx, shortfall);
        self.pending_shortfall_sessions.push(session);
    }

    pub fn has_pending_shortfall(&self) -> bool {
        !self.pending_shortfall_sessions.is_empty()
    }

    pub fn take_shortfall_sessions(&mut self) -> Vec<ShortfallResolutionSession> {
        std::mem::take(&mut self.pending_shortfall_sessions)
    }

    pub fn resolve_shortfall_with_tiles(
        &mut self,
        session: ShortfallResolutionSession,
        chosen_tile_order: Vec<usize>,
    ) {
        self.framework
            .resolve_shortfall_with_tile_choices(session, chosen_tile_order);
    }

    pub fn get_game_state(&self) -> GameState {
        GameState {
            current_player: self.framework.current_player,
            phase: self.game_phase,
            turn_count: self.turn_count,
            players: self.framework.board.players().clone(),
            actions_remaining_in_turn: self.actions_remaining_in_turn,
            round_in_phase: self.round_in_phase,
        }
    }

    pub fn is_game_finished(&self) -> bool {
        self.game_phase == GamePhase::GameEnd
    }
}

#[derive(Debug, Clone)]
pub struct GameState {
    pub current_player: usize,
    pub phase: GamePhase,
    pub turn_count: u32,
    pub players: Vec<Player>,
    pub actions_remaining_in_turn: u8,
    pub round_in_phase: u32,
}
