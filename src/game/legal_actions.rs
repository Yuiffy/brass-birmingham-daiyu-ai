use std::collections::HashSet;

use crate::core::types::ActionType;
use crate::game::framework::{ActionChoice, ActionIntent, ChoiceSet};
use crate::game::runner::GameRunner;

const MAX_STAGED_CHOICES: usize = 64;

/// One complete board-game action, including every staged UI choice and confirm.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LegalAction {
    pub root: ActionType,
    pub choices: Vec<ActionChoice>,
    pub intent: ActionIntent,
}

impl LegalAction {
    /// Stable, human-readable identity for search trees, logs, and API round trips.
    pub fn key(&self) -> String {
        let root = action_type_key(self.root);
        let choices = self
            .choices
            .iter()
            .map(action_choice_key)
            .collect::<Vec<_>>()
            .join(",");
        format!("{root}|{choices}")
    }

    /// Strategic action identity with discard-card choices removed.
    pub fn card_invariant_key(&self) -> String {
        let root = action_type_key(self.root);
        let choices = self
            .choices
            .iter()
            .filter(|choice| !matches!(choice, ActionChoice::Card(_)))
            .map(action_choice_key)
            .collect::<Vec<_>>()
            .join(",");
        format!("{root}|{choices}")
    }

    /// Replays this action against `runner` and advances to the next decision point.
    /// The runner is restored exactly if the action is stale or cannot be committed.
    pub fn apply(&self, runner: &mut GameRunner) -> Result<(), String> {
        let checkpoint = runner.clone();
        if let Err(error) = self.apply_inner(runner) {
            *runner = checkpoint;
            return Err(error);
        }
        Ok(())
    }

    fn apply_inner(&self, runner: &mut GameRunner) -> Result<(), String> {
        ensure_decision_boundary(runner)?;

        let roots = runner.start_turn();
        if !roots.contains(&self.root) {
            return Err(format!("root action {:?} is no longer legal", self.root));
        }
        if self.choices.last() != Some(&ActionChoice::Confirm) {
            return Err("legal action must end with Confirm".to_string());
        }

        runner.framework.start_action_session(self.root);
        for (choice_idx, choice) in self.choices.iter().enumerate() {
            if *choice == ActionChoice::Confirm {
                if choice_idx + 1 != self.choices.len() {
                    return Err("Confirm must be the final staged choice".to_string());
                }
                if !runner.framework.can_confirm() {
                    return Err("action is not ready to confirm".to_string());
                }
                let actual_intent = runner
                    .framework
                    .current_session()
                    .ok_or_else(|| "action session disappeared before confirm".to_string())?
                    .intent;
                if actual_intent != self.intent {
                    return Err("action intent no longer matches the current state".to_string());
                }
                runner.confirm_action()?;
                continue;
            }

            let choice_set = runner
                .framework
                .get_next_choice_set()
                .ok_or_else(|| format!("missing choice set at staged choice {choice_idx}"))?;
            if !choice_set_contains(&choice_set, choice) {
                return Err(format!(
                    "choice {choice:?} is not legal for current set {choice_set:?}"
                ));
            }
            runner
                .framework
                .apply_action_choice(choice.clone())
                .map_err(|error| format!("failed staged choice {choice_idx}: {error}"))?;
        }

        if runner.framework.action_context.is_some() {
            return Err("action sequence ended without committing".to_string());
        }

        if runner.actions_remaining_in_turn == 0 {
            runner.end_turn();
            if !runner.is_game_finished() && !runner.has_pending_shortfall() {
                let _ = runner.start_turn();
            }
        }
        Ok(())
    }
}

/// Enumerates complete, confirmable actions without mutating the supplied runner.
pub fn enumerate_legal_actions(runner: &GameRunner) -> Result<Vec<LegalAction>, String> {
    ensure_decision_boundary(runner)?;
    if runner.is_game_finished() {
        return Ok(Vec::new());
    }

    let mut decision_state = runner.clone();
    let roots = decision_state.start_turn();
    let mut actions = Vec::new();

    for root in roots {
        let mut branch = decision_state.clone();
        branch.framework.start_action_session(root);
        enumerate_session(branch, root, Vec::new(), &mut actions)?;
    }

    Ok(actions)
}

/// Enumerate one root family without expanding unrelated actions.
///
/// `BuildRailroad` intentionally retains both the single- and double-rail
/// branches exposed by the shared network action session.
pub(crate) fn enumerate_legal_actions_for_root(
    runner: &GameRunner,
    root: ActionType,
) -> Result<Vec<LegalAction>, String> {
    ensure_decision_boundary(runner)?;
    if runner.is_game_finished() {
        return Ok(Vec::new());
    }

    let canonical_root = if root == ActionType::BuildDoubleRailroad {
        ActionType::BuildRailroad
    } else {
        root
    };
    let mut decision_state = runner.clone();
    let roots = decision_state.start_turn();
    if !roots.contains(&canonical_root) {
        return Ok(Vec::new());
    }

    let mut branch = decision_state.clone();
    branch.framework.start_action_session(canonical_root);
    let mut actions = Vec::new();
    enumerate_session(branch, canonical_root, Vec::new(), &mut actions)?;
    if root == ActionType::BuildDoubleRailroad {
        actions.retain(|action| action.intent.action_type == ActionType::BuildDoubleRailroad);
    }
    Ok(actions)
}

fn ensure_decision_boundary(runner: &GameRunner) -> Result<(), String> {
    if runner.framework.action_context.is_some() {
        return Err("cannot enumerate or apply during an active action session".to_string());
    }
    if runner.has_pending_shortfall() {
        return Err("income shortfall must be resolved before the next action".to_string());
    }
    Ok(())
}

fn enumerate_session(
    runner: GameRunner,
    root: ActionType,
    choices: Vec<ActionChoice>,
    actions: &mut Vec<LegalAction>,
) -> Result<(), String> {
    if choices.len() >= MAX_STAGED_CHOICES {
        return Err(format!(
            "action {:?} exceeded {MAX_STAGED_CHOICES} staged choices",
            root
        ));
    }

    let session = runner
        .framework
        .current_session()
        .ok_or_else(|| format!("action session for {root:?} ended before confirmation"))?;

    // Sell is the only current action that can either stop or continue at one node.
    if session.can_confirm {
        let mut confirmed_runner = runner.clone();
        confirmed_runner.confirm_action().map_err(|error| {
            format!("enumerated action {root:?} failed confirmation after {choices:?}: {error}")
        })?;
        let mut confirmed_choices = choices.clone();
        confirmed_choices.push(ActionChoice::Confirm);
        actions.push(LegalAction {
            root,
            choices: confirmed_choices,
            intent: session.intent.clone(),
        });
    }

    let choice_set = runner
        .framework
        .get_next_choice_set()
        .ok_or_else(|| format!("missing choice set while enumerating {root:?}"))?;
    if choice_set == ChoiceSet::ConfirmOnly {
        if !session.can_confirm {
            return Err(format!(
                "{root:?} exposed ConfirmOnly before it could be confirmed"
            ));
        }
        return Ok(());
    }

    let is_card_branch = matches!(choice_set, ChoiceSet::Card(_));
    let is_canonical_scout_card_branch = root == ActionType::Scout && is_card_branch;
    let mut next_choices = choices_for_set(choice_set, root, &choices);
    if is_card_branch {
        retain_canonical_card_choices(&runner, &mut next_choices)?;
    }
    if next_choices.is_empty() {
        // Ascending Scout indices remove permutation duplicates. A high partial
        // combination may have no ascending continuation and is simply pruned.
        if is_canonical_scout_card_branch {
            return Ok(());
        }
        return Err(format!("{root:?} exposed an empty non-confirm choice set"));
    }

    for choice in next_choices {
        let mut branch = runner.clone();
        let before = branch
            .framework
            .current_session()
            .expect("session was checked above");
        let after = branch
            .framework
            .apply_action_choice(choice.clone())
            .map_err(|error| format!("failed to enumerate {root:?} choice {choice:?}: {error}"))?
            .ok_or_else(|| format!("choice {choice:?} unexpectedly ended the action session"))?;

        if before.intent == after.intent
            && before.next_choices == after.next_choices
            && before.can_confirm == after.can_confirm
        {
            return Err(format!(
                "choice {choice:?} did not advance the {root:?} action session"
            ));
        }

        let mut branch_choices = choices.clone();
        branch_choices.push(choice);
        enumerate_session(branch, root, branch_choices, actions)?;
    }

    Ok(())
}

fn retain_canonical_card_choices(
    runner: &GameRunner,
    choices: &mut Vec<ActionChoice>,
) -> Result<(), String> {
    let hand = &runner.framework.board.state.players[runner.framework.current_player]
        .hand
        .cards;
    let mut seen_card_types = HashSet::new();
    let mut canonical = Vec::with_capacity(choices.len());
    for choice in choices.drain(..) {
        match &choice {
            ActionChoice::Card(index) => {
                let card = hand
                    .get(*index)
                    .ok_or_else(|| format!("card choice references missing hand index {index}"))?;
                if seen_card_types.insert(card.card_type.clone()) {
                    canonical.push(choice);
                }
            }
            _ => canonical.push(choice),
        }
    }
    *choices = canonical;
    Ok(())
}

pub(crate) fn choices_for_set(
    set: ChoiceSet,
    root: ActionType,
    prior_choices: &[ActionChoice],
) -> Vec<ActionChoice> {
    let mut choices = match set {
        ChoiceSet::Industry(values) => values.into_iter().map(ActionChoice::Industry).collect(),
        ChoiceSet::Card(values) => values.into_iter().map(ActionChoice::Card).collect(),
        ChoiceSet::BuildLocation(values) => values
            .into_iter()
            .map(ActionChoice::BuildLocation)
            .collect(),
        ChoiceSet::Road(values) | ChoiceSet::SecondRoad(values) => {
            values.into_iter().map(ActionChoice::Road).collect()
        }
        ChoiceSet::CoalSource(values) => values.into_iter().map(ActionChoice::CoalSource).collect(),
        ChoiceSet::IronSource(values) => values.into_iter().map(ActionChoice::IronSource).collect(),
        ChoiceSet::BeerSource(values) => values.into_iter().map(ActionChoice::BeerSource).collect(),
        ChoiceSet::ActionBeerSource(values) => values
            .into_iter()
            .map(ActionChoice::ActionBeerSource)
            .collect(),
        ChoiceSet::SellTarget(values) => values.into_iter().map(ActionChoice::SellTarget).collect(),
        ChoiceSet::FreeDevelopment(values) => values
            .into_iter()
            .map(ActionChoice::FreeDevelopment)
            .collect(),
        ChoiceSet::SecondIndustry(values) => {
            values.into_iter().map(ActionChoice::Industry).collect()
        }
        ChoiceSet::NetworkMode(values) => {
            values.into_iter().map(ActionChoice::NetworkMode).collect()
        }
        ChoiceSet::ConfirmOnly => Vec::new(),
    };

    if root == ActionType::Scout {
        let used_cards = prior_choices
            .iter()
            .filter_map(|choice| match choice {
                ActionChoice::Card(index) => Some(*index),
                _ => None,
            })
            .collect::<Vec<_>>();
        choices.retain(|choice| match choice {
            ActionChoice::Card(index) => used_cards
                .last()
                .is_none_or(|previous_index| index > previous_index),
            _ => true,
        });
    }

    choices
}

fn action_type_key(action_type: ActionType) -> &'static str {
    match action_type {
        ActionType::BuildBuilding => "build",
        ActionType::BuildRailroad => "network",
        ActionType::BuildDoubleRailroad => "double_network",
        ActionType::Develop => "develop",
        ActionType::DevelopDouble => "develop_double",
        ActionType::Sell => "sell",
        ActionType::Loan => "loan",
        ActionType::Scout => "scout",
        ActionType::Pass => "pass",
    }
}

fn action_choice_key(choice: &ActionChoice) -> String {
    match choice {
        ActionChoice::Industry(value) => format!("i{}", *value as usize),
        ActionChoice::Card(value) => format!("c{value}"),
        ActionChoice::BuildLocation(value) => format!("b{value}"),
        ActionChoice::Road(value) => format!("r{value}"),
        ActionChoice::SellTarget(value) => format!("s{value}"),
        ActionChoice::CoalSource(value) => format!("coal{}", resource_source_key(*value)),
        ActionChoice::IronSource(value) => format!("iron{}", resource_source_key(*value)),
        ActionChoice::BeerSource(value) => match value {
            crate::board::resources::BeerSellSource::Building(location) => {
                format!("beerb{location}")
            }
            crate::board::resources::BeerSellSource::TradePost(slot) => {
                format!("beerm{slot}")
            }
        },
        ActionChoice::ActionBeerSource(value) => match value {
            crate::board::resources::BreweryBeerSource::OwnBrewery(location) => {
                format!("action_beer_own{location}")
            }
            crate::board::resources::BreweryBeerSource::OpponentBrewery(location) => {
                format!("action_beer_opponent{location}")
            }
        },
        ActionChoice::FreeDevelopment(value) => format!("free{}", *value as usize),
        ActionChoice::NetworkMode(value) => match value {
            crate::game::framework::NetworkMode::Single => "mode_single".to_string(),
            crate::game::framework::NetworkMode::Double => "mode_double".to_string(),
        },
        ActionChoice::Confirm => "confirm".to_string(),
        ActionChoice::Cancel => "cancel".to_string(),
    }
}

fn resource_source_key(source: crate::board::resources::ResourceSource) -> String {
    match source {
        crate::board::resources::ResourceSource::Building(location) => format!("b{location}"),
        crate::board::resources::ResourceSource::Market => "m".to_string(),
    }
}

pub(crate) fn choice_set_contains(set: &ChoiceSet, choice: &ActionChoice) -> bool {
    match (set, choice) {
        (ChoiceSet::Industry(values), ActionChoice::Industry(value))
        | (ChoiceSet::SecondIndustry(values), ActionChoice::Industry(value)) => {
            values.contains(value)
        }
        // The web protocol represents the second industry as a free-development
        // choice, while the Python composite-action bridge still uses the
        // historical `Industry` variant. Accept both encodings for the same
        // advertised choice set.
        (ChoiceSet::SecondIndustry(values), ActionChoice::FreeDevelopment(value)) => {
            values.contains(value)
        }
        (ChoiceSet::Card(values), ActionChoice::Card(value))
        | (ChoiceSet::BuildLocation(values), ActionChoice::BuildLocation(value))
        | (ChoiceSet::Road(values), ActionChoice::Road(value))
        | (ChoiceSet::SecondRoad(values), ActionChoice::Road(value))
        | (ChoiceSet::SellTarget(values), ActionChoice::SellTarget(value)) => {
            values.contains(value)
        }
        (ChoiceSet::CoalSource(values), ActionChoice::CoalSource(value))
        | (ChoiceSet::IronSource(values), ActionChoice::IronSource(value)) => {
            values.contains(value)
        }
        (ChoiceSet::BeerSource(values), ActionChoice::BeerSource(value)) => values.contains(value),
        (ChoiceSet::ActionBeerSource(values), ActionChoice::ActionBeerSource(value)) => {
            values.contains(value)
        }
        (ChoiceSet::FreeDevelopment(values), ActionChoice::FreeDevelopment(value)) => {
            values.contains(value)
        }
        (ChoiceSet::NetworkMode(values), ActionChoice::NetworkMode(value)) => {
            values.contains(value)
        }
        (ChoiceSet::ConfirmOnly, ActionChoice::Confirm) => true,
        _ => false,
    }
}
