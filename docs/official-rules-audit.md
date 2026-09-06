# Economic Rules Audit (2026-09-06)

Source: the [Brass: Birmingham rulebook](https://cdn.1j1ju.com/medias/60/39/64-brass-birmingham-rulebook.pdf).
Page numbers below are printed page numbers; the PDF contains two-page spreads.
The local source and scoring illustration are retained in ignored `output/`.

| Rule | Previous behavior | Corrected behavior | Source |
| --- | --- | --- | --- |
| Loan cost | Repeated marker decrements could lose more than three income bands. | Move down exactly three displayed income bands, landing on the highest space in the target band. | p. 10, Loan Action |
| Loan floor | Allowed a loan at -8 or -9, clamping its cost at -10. | A loan requires income of at least -7. | p. 10, Loan Action exception |
| Hand refill | Drew after each action, making an unknown draw available for the second action. | Refill after the turn's last action; the opening single-action turn still refills immediately. | p. 6, Player Turns |
| Setup discards | One face-down card for the entire table. | One face-down discard per player. Passing-only games now have exactly 10/9/8 rounds per era for 2/3/4 players. | p. 5, Player Area Setup 9; p. 6, Rounds |
| Scout | Allowed with one Wild card already held. | Either type of held Wild card prevents Scout. | p. 10, Scout Action |
| Link scoring | Counted flipped industries but omitted printed merchant-location icons. | Add the two printed icons at every adjacent merchant location, including inactive locations. | p. 7, Score Canal/Rail Links and board illustration |
| Ties | Compared the income marker's space before money. | Compare displayed income, then money. Two spaces in the same income band do not break the tie. | p. 7, Winning the Game |

`BoardState::link_victory_points` is shared by era settlement, search estimates and Python
observations. `Player::final_ranking_key` is shared by search winners, self-play placements,
training targets and Python outcomes. This keeps the consumer interfaces on the same rules.

Regression coverage includes every income marker space, the loan floor, either Wild card,
inactive merchant scoring, card draw timing, setup deck sizes and complete passing-only games
for all player counts. Existing assertions that explicitly preserved the incorrect rules were
updated to the rulebook behavior.

## Compatibility

These changes alter legal actions, second-action hands, game length and scores. Old fixed-seed
outcomes cannot be compared directly with new scores to isolate AI improvement. All v2 policy
comparisons must run both policies under the corrected engine.

Historical browser saves store actions and rebuild from the seed. An old action stream may
replay differently or become illegal under these corrections. Existing saves are retained;
use fresh games for the new preview and retain an older executable when auditing old replays.
Old human replay shards and neural checkpoints were not regenerated or retrained. They must
not be described as trained on these corrected rules.

This is a focused audit of economic decisions, turn flow and scoring, not certification of
every rule or every UI workflow against an independent implementation.
