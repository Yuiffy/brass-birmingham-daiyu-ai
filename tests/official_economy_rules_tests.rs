use fast_brass::board::Board;
use fast_brass::core::static_data::LINK_LOCATIONS;
use fast_brass::core::types::{ActionType, Card, CardType};
use fast_brass::game::runner::{GamePhase, GameRunner};

#[test]
fn loan_loses_exactly_three_income_bands_at_every_legal_space() {
    let mut board = Board::new(2, Some(10));
    for space in 0..=100 {
        board.state.players[0].income_level = space;
        let income = board.state.players[0].get_income_amount(space);
        assert_eq!(board.can_take_loan(0), income >= -7, "space {space}");
        if income >= -7 {
            let p = &mut board.state.players[0];
            p.decrease_income_level(3);
            assert_eq!(
                p.get_income_amount(p.income_level),
                income - 3,
                "space {space}"
            );
            assert!(p.income_level == 100 || p.get_income_amount(p.income_level + 1) > income - 3);
        }
    }
}

#[test]
fn either_wild_card_prevents_scouting() {
    let mut board = Board::new(2, Some(11));
    assert!(board.can_scout(0));
    for wild in [CardType::WildIndustry, CardType::WildLocation] {
        board.state.players[0].hand.cards[0] = Card::new(wild);
        assert!(!board.can_scout(0));
    }
}

#[test]
fn merchant_link_icons_score_without_active_merchants() {
    let mut runner = GameRunner::new(2, Some(12));
    runner.framework.board.state.trade_post_slots.fill(None);
    let road = LINK_LOCATIONS
        .iter()
        .position(|link| link.locations.ones().any(|town| town == 26))
        .unwrap();
    runner.framework.board.state.place_link(0, road);
    assert_eq!(runner.framework.board.state.link_victory_points(road), 2);
    runner.end_era();
    assert_eq!(runner.framework.board.state.players[0].victory_points, 2);
}

#[test]
fn passing_games_have_official_round_and_action_counts() {
    use fast_brass::game::framework::ActionChoice;
    for players in 2..=4 {
        let mut runner = GameRunner::new(players, Some(13));
        let rounds = 12 - players;
        let mut counts = vec![[0; 2]; players];
        while !runner.is_game_finished() {
            runner.start_turn();
            let actor = runner.framework.current_player;
            let era = usize::from(runner.game_phase == GamePhase::Railroad);
            runner.start_action(ActionType::Pass);
            runner.apply_choice(ActionChoice::Card(0));
            runner.confirm_action().unwrap();
            counts[actor][era] += 1;
            if runner.actions_remaining_in_turn == 0 {
                runner.end_turn();
            }
        }
        assert_eq!(runner.round_in_phase as usize, rounds);
        assert!(
            counts.iter().all(|c| *c == [rounds * 2 - 1, rounds * 2]),
            "{players} players: {counts:?}"
        );
    }
}
