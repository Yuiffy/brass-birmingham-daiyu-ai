// Extracted from cards.rs
use crate::core::types::*;

#[derive(Debug, Clone)]
pub struct Hand {
    pub cards: Vec<Card>,
}

impl Hand {
    pub fn new(cards: Vec<Card>) -> Self {
        Self { cards }
    }

    pub fn add_card(&mut self, card: Card) {
        self.cards.push(card);
    }

    /// Removes and returns a card at a specific index.
    /// Returns None if the index is out of bounds.
    pub fn remove_card_at(&mut self, index: usize) -> Option<Card> {
        if index < self.cards.len() {
            Some(self.cards.remove(index))
        } else {
            None
        }
    }

    /// Removes the first occurrence of a specific card type.
    /// Returns true if removed, false otherwise.
    pub fn remove_card_by_type(&mut self, card_type_to_remove: CardType) -> bool {
        if let Some(pos) = self
            .cards
            .iter()
            .position(|c| c.card_type == card_type_to_remove)
        {
            self.cards.remove(pos);
            true
        } else {
            false
        }
    }
}
