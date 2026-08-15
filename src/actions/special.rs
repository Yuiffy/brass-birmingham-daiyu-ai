use crate::board::resources::{discard_card, remove_building_from_board};
use crate::core::types::*;

/// Special actions: Loan, Scout, Pass
pub struct SpecialActions;

impl SpecialActions {
    /// Execute loan action
    pub fn execute_loan_action(
        board_state: &mut crate::board::BoardState,
        player_idx: usize,
        card_to_discard_idx: usize,
    ) {
        if !Self::can_take_loan(board_state, player_idx) {
            eprintln!("Player {} cannot take loan.", player_idx);
            return;
        }

        discard_card(board_state, player_idx, card_to_discard_idx);
        board_state.players[player_idx].gain_money(30);
        // Loan costs 3 income levels
        board_state.players[player_idx].decrease_income_level(3);
    }

    /// Execute scout action
    pub fn execute_scout_action(
        board_state: &mut crate::board::BoardState,
        player_idx: usize,
        card_action_idx: usize,
        card_add1_idx: usize,
        card_add2_idx: usize,
    ) {
        if !Self::can_scout(board_state, player_idx) {
            eprintln!("Player {} cannot scout.", player_idx);
            return;
        }

        let mut discards_indices = vec![card_action_idx, card_add1_idx, card_add2_idx];
        discards_indices.sort_unstable_by(|a, b| b.cmp(a)); // Sort descending for safe removal
        discards_indices.dedup(); // Ensure uniqueness

        if discards_indices.len() != 3 {
            eprintln!("Scout action requires 3 unique card indices to discard.");
            return;
        }

        // Discard the 3 cards (descending indices already computed)
        for &idx in &discards_indices {
            if idx >= board_state.players[player_idx].hand.cards.len() {
                eprintln!(
                    "Invalid card index {} for scout discard for player {}. Hand size: {}.",
                    idx,
                    player_idx,
                    board_state.players[player_idx].hand.cards.len()
                );
                return;
            }
            discard_card(board_state, player_idx, idx);
        }

        // Give wild cards
        board_state.wild_location_cards_available =
            board_state.wild_location_cards_available.saturating_sub(1);
        board_state.wild_industry_cards_available =
            board_state.wild_industry_cards_available.saturating_sub(1);
        board_state.players[player_idx]
            .hand
            .cards
            .push(Card::new(CardType::WildLocation));
        board_state.players[player_idx]
            .hand
            .cards
            .push(Card::new(CardType::WildIndustry));
    }

    /// Check if player can take a loan
    pub fn can_take_loan(board_state: &crate::board::BoardState, player_idx: usize) -> bool {
        let income_level = board_state.players[player_idx].income_level;
        let income_val = board_state.players[player_idx].get_income_amount(income_level);
        income_val > -10
    }

    /// Check if player can scout
    pub fn can_scout(board_state: &crate::board::BoardState, player_idx: usize) -> bool {
        if board_state.players[player_idx].hand.cards.len() < 3 {
            return false;
        }
        if board_state.wild_location_cards_available == 0
            || board_state.wild_industry_cards_available == 0
        {
            return false;
        }

        let (mut has_wl, mut has_wi) = (false, false);
        for card in &board_state.players[player_idx].hand.cards {
            if card.card_type == CardType::WildLocation {
                has_wl = true;
            }
            if card.card_type == CardType::WildIndustry {
                has_wi = true;
            }
        }
        !(has_wl && has_wi) // Cannot scout if already holding both types
    }

    /// Liquidate assets for a player who can't pay debt
    pub fn liquidate_for_player(board_state: &mut crate::board::BoardState, player_idx: usize) {
        use crate::core::static_data::INDUSTRY_MAT;

        let mut amount_owed = {
            let player = &board_state.players[player_idx];
            let current_income = player.get_income_amount(player.income_level);
            if current_income >= 0 || player.money >= current_income.abs() as u16 {
                return;
            }
            current_income.abs() as u16 - player.money
        };

        board_state.players[player_idx].money = 0;

        let mut player_buildings_to_consider: Vec<(usize, u16)> = board_state.player_building_mask
            [player_idx]
            .ones()
            .filter_map(|loc| {
                board_state.bl_to_building.get(&loc).map(|building| {
                    let cost = INDUSTRY_MAT[building.industry as usize][building.level.as_usize()]
                        .money_cost;
                    (loc, cost)
                })
            })
            .collect();

        player_buildings_to_consider.sort_by_key(|&(_, cost)| cost);

        for (loc_to_remove, original_cost) in player_buildings_to_consider {
            if amount_owed == 0 {
                break;
            }
            let money_recovered = original_cost / 2;
            board_state.players[player_idx].gain_money(money_recovered);
            amount_owed = amount_owed.saturating_sub(money_recovered);

            remove_building_from_board(board_state, loc_to_remove);
        }

        if amount_owed > 0 {
            let vp_to_lose = amount_owed as u16;
            board_state.players[player_idx].victory_points = board_state.players[player_idx]
                .victory_points
                .saturating_sub(vp_to_lose);
            board_state.visible_vps[player_idx] =
                board_state.visible_vps[player_idx].saturating_sub(vp_to_lose);
        }
    }

    /// Update turn order based on spending
    pub fn update_turn_order(board_state: &mut crate::board::BoardState) {
        let mut player_spend_info: Vec<(usize, u16, usize)> = board_state
            .players
            .iter()
            .enumerate()
            .map(|(idx, p)| {
                (
                    idx,
                    p.spent_this_turn,
                    board_state
                        .turn_order
                        .iter()
                        .position(|&original_pos_idx| original_pos_idx == idx)
                        .unwrap_or(usize::MAX),
                )
            })
            .collect();

        player_spend_info.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.2.cmp(&b.2)));
        board_state.turn_order = player_spend_info.iter().map(|info| info.0).collect();
    }
}
