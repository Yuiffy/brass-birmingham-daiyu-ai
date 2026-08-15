// Extracted from cards.rs
use crate::core::types::*;
use rand::prelude::*;
use rand::seq::SliceRandom;

#[derive(Debug, Clone)]
pub struct Deck {
    pub cards: Vec<Card>,
    pub discard_pile: Vec<Card>,
    pub rng: StdRng,
}

use crate::consts::{STARTING_CARDS_2P, STARTING_CARDS_3P, STARTING_CARDS_4P};
impl Deck {
    /// Creates a new deck based on the number of players
    pub fn new(num_players: usize, mut rng: StdRng) -> Self {
        let mut cards = match num_players {
            2 => STARTING_CARDS_2P.to_vec(),
            3 => STARTING_CARDS_3P.to_vec(),
            4 => STARTING_CARDS_4P.to_vec(),
            _ => panic!("Unsupported number of players: {}", num_players),
        };
        cards.shuffle(&mut rng);

        Deck {
            cards,
            discard_pile: Vec::new(),
            rng,
        }
    }

    pub fn cards_left(&self) -> usize {
        self.cards.len()
    }

    pub fn reshuffle_with_cards(&mut self, cards: Vec<Card>) {
        self.cards = cards;
        self.shuffle();
    }

    pub fn discard(&mut self, card: Card) {
        self.discard_pile.push(card);
    }

    pub fn shuffle(&mut self) {
        self.cards.shuffle(&mut self.rng);
    }

    pub fn draw(&mut self) -> Option<Card> {
        self.cards.pop()
    }

    pub fn draw_n(&mut self, n: usize) -> Vec<Card> {
        let mut drawn = Vec::with_capacity(n);
        for _ in 0..n {
            if let Some(card) = self.draw() {
                drawn.push(card);
            } else {
                break;
            }
        }
        drawn
    }

    pub fn add_cards(&mut self, discarded_cards: Vec<Card>) {
        self.cards.extend(discarded_cards);
    }

    pub fn is_empty(&self) -> bool {
        self.cards.is_empty()
    }

    pub fn len(&self) -> usize {
        self.cards.len()
    }
}
