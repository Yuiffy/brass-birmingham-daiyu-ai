use rand::seq::SliceRandom;
use rand::Rng;

use crate::core::types::{Card, CardType};
use crate::game::runner::GameRunner;

/// Samples one state consistent with the observer's information.
///
/// Wild cards remain with the player who publicly obtained them through Scout;
/// only normal cards in opponents' hands and the draw deck are exchangeable.
pub fn determinize_hidden_information<R: Rng + ?Sized>(
    runner: &mut GameRunner,
    observer_idx: usize,
    rng: &mut R,
) -> Result<(), String> {
    let state = &mut runner.framework.board.state;
    if observer_idx >= state.players.len() {
        return Err(format!(
            "observer index {observer_idx} out of bounds for {} players",
            state.players.len()
        ));
    }

    if state.deck.cards.iter().any(is_wild_card) {
        return Err("draw deck contains a wild card before determinization".to_string());
    }

    let mut hidden_normal_cards = state.deck.cards.clone();
    let mut opponent_layouts = Vec::<(usize, usize, Vec<Card>)>::new();
    for (player_idx, player) in state.players.iter().enumerate() {
        if player_idx == observer_idx {
            continue;
        }
        let wild_cards = player
            .hand
            .cards
            .iter()
            .filter(|card| is_wild_card(card))
            .cloned()
            .collect::<Vec<_>>();
        let normal_cards = player
            .hand
            .cards
            .iter()
            .filter(|card| !is_wild_card(card))
            .cloned()
            .collect::<Vec<_>>();
        hidden_normal_cards.extend(normal_cards.iter().cloned());
        opponent_layouts.push((player_idx, normal_cards.len(), wild_cards));
    }

    hidden_normal_cards.shuffle(rng);
    let mut cursor = 0usize;
    for (player_idx, normal_count, wild_cards) in opponent_layouts {
        let end = cursor + normal_count;
        if end > hidden_normal_cards.len() {
            return Err("inconsistent hidden card pool during determinization".to_string());
        }
        let hand = &mut state.players[player_idx].hand.cards;
        hand.clear();
        hand.extend_from_slice(&hidden_normal_cards[cursor..end]);
        hand.extend(wild_cards);
        cursor = end;
    }
    state.deck.cards = hidden_normal_cards[cursor..].to_vec();
    Ok(())
}

fn is_wild_card(card: &Card) -> bool {
    matches!(
        card.card_type,
        CardType::WildLocation | CardType::WildIndustry
    )
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use rand::rngs::StdRng;
    use rand::SeedableRng;

    use super::*;

    fn card_multiset<'a>(cards: impl IntoIterator<Item = &'a Card>) -> HashMap<CardType, usize> {
        let mut counts = HashMap::new();
        for card in cards {
            *counts.entry(card.card_type.clone()).or_insert(0) += 1;
        }
        counts
    }

    fn hidden_normal_multiset(runner: &GameRunner, observer: usize) -> HashMap<CardType, usize> {
        let state = &runner.framework.board.state;
        card_multiset(
            state
                .deck
                .cards
                .iter()
                .chain(
                    state
                        .players
                        .iter()
                        .enumerate()
                        .filter(|(player_idx, _)| *player_idx != observer)
                        .flat_map(|(_, player)| player.hand.cards.iter()),
                )
                .filter(|card| !is_wild_card(card)),
        )
    }

    fn wild_card_multiset(cards: &[Card]) -> HashMap<CardType, usize> {
        card_multiset(cards.iter().filter(|card| is_wild_card(card)))
    }

    #[test]
    fn determinization_preserves_all_information_constraints() {
        let mut runner = GameRunner::new(4, Some(41));
        let observer = 0usize;
        runner.framework.board.state.players[0]
            .hand
            .cards
            .push(Card::new(CardType::WildLocation));
        runner.framework.board.state.players[1]
            .hand
            .cards
            .push(Card::new(CardType::WildLocation));
        runner.framework.board.state.players[1]
            .hand
            .cards
            .push(Card::new(CardType::WildIndustry));
        runner.framework.board.state.players[3]
            .hand
            .cards
            .push(Card::new(CardType::WildIndustry));

        let observer_hand = runner.framework.board.state.players[observer]
            .hand
            .cards
            .clone();
        let hand_lengths = runner
            .framework
            .board
            .state
            .players
            .iter()
            .map(|player| player.hand.cards.len())
            .collect::<Vec<_>>();
        let wild_cards_by_player = runner
            .framework
            .board
            .state
            .players
            .iter()
            .map(|player| wild_card_multiset(&player.hand.cards))
            .collect::<Vec<_>>();
        let hidden_normal_cards = hidden_normal_multiset(&runner, observer);

        let mut rng = StdRng::seed_from_u64(99);
        determinize_hidden_information(&mut runner, observer, &mut rng).unwrap();

        assert_eq!(
            runner.framework.board.state.players[observer].hand.cards,
            observer_hand
        );
        assert_eq!(
            hidden_normal_multiset(&runner, observer),
            hidden_normal_cards
        );
        for (player_idx, player) in runner.framework.board.state.players.iter().enumerate() {
            assert_eq!(player.hand.cards.len(), hand_lengths[player_idx]);
            assert_eq!(
                wild_card_multiset(&player.hand.cards),
                wild_cards_by_player[player_idx]
            );
        }
        assert!(!runner
            .framework
            .board
            .state
            .deck
            .cards
            .iter()
            .any(is_wild_card));
    }

    #[test]
    fn determinization_rejects_wild_cards_in_the_draw_deck_without_mutation() {
        let mut runner = GameRunner::new(2, Some(42));
        runner
            .framework
            .board
            .state
            .deck
            .cards
            .push(Card::new(CardType::WildLocation));
        let before_deck = runner.framework.board.state.deck.cards.clone();
        let before_hands = runner
            .framework
            .board
            .state
            .players
            .iter()
            .map(|player| player.hand.cards.clone())
            .collect::<Vec<_>>();

        let mut rng = StdRng::seed_from_u64(100);
        let error = determinize_hidden_information(&mut runner, 0, &mut rng)
            .expect_err("wild cards in the deck must be rejected");

        assert!(error.contains("draw deck contains a wild card"));
        assert_eq!(runner.framework.board.state.deck.cards, before_deck);
        assert_eq!(
            runner
                .framework
                .board
                .state
                .players
                .iter()
                .map(|player| player.hand.cards.clone())
                .collect::<Vec<_>>(),
            before_hands
        );
    }

    #[test]
    fn determinization_rejects_an_invalid_observer_without_mutation() {
        let mut runner = GameRunner::new(2, Some(43));
        let before_deck = runner.framework.board.state.deck.cards.clone();
        let mut rng = StdRng::seed_from_u64(101);

        let error = determinize_hidden_information(&mut runner, 2, &mut rng)
            .expect_err("observer index must be valid");

        assert!(error.contains("observer index 2 out of bounds"));
        assert_eq!(runner.framework.board.state.deck.cards, before_deck);
    }
}
