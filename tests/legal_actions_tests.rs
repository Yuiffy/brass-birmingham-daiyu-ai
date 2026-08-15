use fast_brass::core::building::BuiltBuilding;
use fast_brass::core::locations::TownName;
use fast_brass::core::player::PlayerId;
use fast_brass::core::types::{
    ActionType, Card, CardType, IndustryLevel, IndustrySet, IndustryType,
};
use fast_brass::game::framework::ActionChoice;
use fast_brass::game::legal_actions::enumerate_legal_actions;
use fast_brass::game::runner::GameRunner;
use fast_brass::market::merchants::{MerchantTile, MerchantTileType};

#[test]
fn legal_action_enumeration_is_deterministic() {
    let runner = GameRunner::new(2, Some(2026));

    let first = enumerate_legal_actions(&runner).expect("first enumeration should succeed");
    let second = enumerate_legal_actions(&runner).expect("second enumeration should succeed");

    assert!(!first.is_empty());
    assert_eq!(first, second);
}

#[test]
fn every_enumerated_initial_action_can_be_applied() {
    let runner = GameRunner::new(2, Some(2027));
    let initial_player = runner.framework.current_player;
    let actions = enumerate_legal_actions(&runner).expect("enumeration should succeed");

    for action in &actions {
        let mut branch = runner.clone();
        action.apply(&mut branch).unwrap_or_else(|error| {
            panic!("enumerated action failed to apply: {action:?}: {error}")
        });
        assert!(branch.framework.action_context.is_none());
        assert_ne!(branch.framework.current_player, initial_player);
        assert_eq!(branch.actions_remaining_in_turn, 1);
    }
}

#[test]
fn scout_actions_always_use_three_distinct_cards() {
    let runner = GameRunner::new(2, Some(2028));
    let actions = enumerate_legal_actions(&runner).expect("enumeration should succeed");
    let scout_actions = actions
        .iter()
        .filter(|action| action.root == ActionType::Scout)
        .collect::<Vec<_>>();

    assert!(!scout_actions.is_empty());
    assert_eq!(
        scout_actions.len(),
        56,
        "8 choose 3 Scout combinations expected"
    );
    for action in scout_actions {
        let card_indices = action
            .choices
            .iter()
            .filter_map(|choice| match choice {
                ActionChoice::Card(index) => Some(*index),
                _ => None,
            })
            .collect::<Vec<_>>();
        let mut unique_indices = card_indices.clone();
        unique_indices.sort_unstable();
        unique_indices.dedup();

        assert_eq!(card_indices.len(), 3, "unexpected scout action: {action:?}");
        assert_eq!(unique_indices.len(), 3, "duplicate scout card: {action:?}");
        assert!(card_indices.windows(2).all(|pair| pair[0] < pair[1]));
        assert_eq!(action.intent.scout_additional_discard_indices.len(), 2);
    }
}

#[test]
fn duplicate_card_copies_do_not_duplicate_semantic_legal_actions() {
    let mut runner = GameRunner::new(2, Some(2_028_001));
    let player_idx = runner.framework.current_player;
    runner.framework.board.state.players[player_idx].hand.cards = vec![
        Card::new(CardType::Location(TownName::Birmingham)),
        Card::new(CardType::Location(TownName::Birmingham)),
        Card::new(CardType::Location(TownName::Dudley)),
    ];

    let actions = enumerate_legal_actions(&runner).expect("enumeration should succeed");
    let loan_actions = actions
        .iter()
        .filter(|action| action.root == ActionType::Loan)
        .collect::<Vec<_>>();

    assert_eq!(loan_actions.len(), 2, "one loan per distinct card type");
    let discarded_indices = loan_actions
        .iter()
        .map(|action| {
            action
                .choices
                .iter()
                .find_map(|choice| match choice {
                    ActionChoice::Card(index) => Some(*index),
                    _ => None,
                })
                .expect("loan action should discard a card")
        })
        .collect::<Vec<_>>();
    assert_eq!(discarded_indices, vec![0, 2]);
    for action in loan_actions {
        let mut branch = runner.clone();
        action
            .apply(&mut branch)
            .expect("canonical duplicate-card action should apply");
    }
}

#[test]
fn sell_actions_can_stop_after_one_target_or_continue() {
    let mut runner = GameRunner::new(2, Some(2029));
    let player_idx = runner.framework.current_player;
    let board = &mut runner.framework.board.state;

    // Connect Birmingham to a single cotton merchant in Oxford.
    board.place_link(player_idx, 31);
    for slot in &mut board.trade_post_slots {
        *slot = None;
    }
    board.trade_post_slots[1] = Some(MerchantTile::from_type(MerchantTileType::Cotton));
    board.trade_post_beer.clear();

    let cotton_locations = [36usize, 37usize];
    let brewery_locations = [17usize, 25usize];
    for location in cotton_locations {
        add_building(
            board,
            player_idx,
            location,
            IndustryType::Cotton,
            IndustryLevel::I,
        );
    }
    for location in brewery_locations {
        add_building(
            board,
            player_idx,
            location,
            IndustryType::Beer,
            IndustryLevel::I,
        );
    }

    board.players[player_idx].hand.cards = vec![Card::new(CardType::Industry(
        IndustrySet::new_from_industry_types(&[IndustryType::Cotton]),
    ))];

    let actions = enumerate_legal_actions(&runner).expect("enumeration should succeed");
    let sell_actions = actions
        .iter()
        .filter(|action| action.root == ActionType::Sell)
        .collect::<Vec<_>>();
    let stop = sell_actions
        .iter()
        .find(|action| action.intent.sell_choices.len() == 1)
        .expect("a one-target sell action should be enumerated");
    let continue_selling = sell_actions
        .iter()
        .find(|action| action.intent.sell_choices.len() == 2)
        .expect("a two-target sell action should be enumerated");

    let mut stop_branch = runner.clone();
    stop.apply(&mut stop_branch)
        .expect("one-target sell should apply");
    assert_eq!(flipped_count(&stop_branch, &cotton_locations), 1);

    let mut continue_branch = runner.clone();
    continue_selling
        .apply(&mut continue_branch)
        .expect("two-target sell should apply");
    assert_eq!(flipped_count(&continue_branch, &cotton_locations), 2);
}

fn add_building(
    board: &mut fast_brass::board::BoardState,
    player_idx: usize,
    location: usize,
    industry: IndustryType,
    level: IndustryLevel,
) {
    board.bl_to_building.insert(
        location,
        BuiltBuilding::build(
            industry,
            level,
            location as u8,
            PlayerId::from_usize(player_idx),
        ),
    );
    board.build_locations_occupied.insert(location);
    board.player_building_mask[player_idx].insert(location);
}

fn flipped_count(runner: &GameRunner, locations: &[usize]) -> usize {
    locations
        .iter()
        .filter(|location| {
            runner
                .framework
                .board
                .state
                .bl_to_building
                .get(location)
                .is_some_and(|building| building.flipped)
        })
        .count()
}
