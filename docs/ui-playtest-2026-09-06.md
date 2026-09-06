# UI playtest: 2026-09-06

Scope: actual Svelte UI and Rust game flow, using an isolated backend on port 3012
and a separate database. The existing game on port 3011 was not changed.
Seed 2026090700, two players, manual control. Desktop: 1440x1000. Mobile: 390x844
and 320x844. Browser automation and screenshots were both reviewed.

## Confirmed issues and fixes

| Priority | Reproduction and player impact | Result |
| --- | --- | --- |
| P1 | Start Build, select Coal, immediately cancel, then start Build again. The previous animation submitted its delayed industry choice into the new action, advancing it to card selection without another choice. | Submit immediately, disable duplicate selection while pending, and invalidate the closed dialog's continuation. Slow-response cancellation and restart now leave the new action at industry selection. |
| P2 | An industry choice waited approximately 972 ms before its API request. Every selectable tile also bounced continuously. | Removed both fixed waits and the continuous bounce. Measured request dispatch was 2 ms after the click in the final run. This measures local dispatch, not network completion. |
| P2 | On a 390 px viewport, the three-column Industry Mat squeezed out the center illustrations and overflowed its right edge. Expanded industry levels had no bounded vertical scrolling. | Responsive columns, a scrollable modal with a persistent header, and readable minimum illustration width. Reviewed every industry and the expanded view at 320 and 390 px. |
| P2 | On a 390 px viewport, the discard viewer's Next button extended to x=432, outside the screen. It exposed strings such as Location(Stafford), lacked a position counter, and had nonfunctional keyboard arrows. | A bounded single-card viewer with card assets, display labels, a counter, keyboard and wheel navigation, and disabled buttons at the ends. |
| P2 | Open Industry Mat or Discard Pile with Enter. Focus remained on the background trigger, so Escape did not close the overlay and keyboard navigation could reach the background. | Native modal dialogs manage focus and background interaction. Closing restores the trigger; Escape in an expanded industry first returns to the main mat. |

Keeping the industry panel open across chained development choices also required explicit
reactive dependencies for projected tile levels and counts. This avoids a Svelte update
exception when the first development changes the displayed tile from level I to level II.
The final test developed Coal twice, committed the action, and verified the remaining stack.

## Verification

- Completed Build, Undo, Loan, End Turn, and Develop x2 through the actual browser and API.
- Verified immediate industry dispatch, cancellation with a delayed response, and retry
  following an intentionally failed request.
- Verified modal focus, background isolation, Escape, expanded-view return, and focus restoration.
- Verified empty/single/multiple discard presentation, card images, navigation boundaries,
  wheel handling, and reset when reopening. Multiple card kinds used a browser-only fixture.
- Checked mobile bounds, minimum icon area, scrolling, and close-button visibility.
- Browser console errors: zero in the final verification run.
- Production build passed without warnings. The 20-city hover regression and standard
  web-game client startup check passed after these changes.
- Checked cross-level pending development counts and live updates in the expanded level view.

Local evidence: `output/ux-audit/before-findings.json`, `after-results.json`, the
`before-*.png` and `after-*.png` screenshots, and `audit.mjs` / `verify.mjs`.
The original city-card hover regression remains in `output/city-card-preview/playtest.mjs`.
The temporary test backend was stopped after verification; the preview remains on port 5176.

## Further improvements

1. Add board-object inspection. Built industries currently show only abbreviations, level,
   and resource marks; inspecting owner, flipped state, income, and scoring values would
   reduce trips to the Industry Mat. Distinguish inspection from selecting a legal action.
2. Add touch-friendly board zoom and pan. Fitting the entire 1200x1200 board into a phone
   makes tile text and legal targets small. Keep the hand and current action reachable.
3. Bring active action controls closer to the board on phones. The current vertical layout
   places player and market information between the board and confirmation/cancellation.
4. Unify player-facing language. The sidebar and action flow are English while control mode
   and analysis use Chinese; use a shared label catalog before adding more translated copy.

Items 1-3 were implemented in the follow-up below. Language unification remains a separate
design task. Endgame and complete AI self-play were outside this UI pass; game rules and
AI code were unchanged.

## Follow-up: map interaction

- Hovering a built industry shows its city, owner, flipped state, resources, scoring values,
  and flip income. Clicking or tapping pins the detail panel; closing restores map focus.
  Links show their connected towns and owner. Merchants show accepted goods and Beer state.
- The map supports 100%-400% zoom relative to its fitted view, wheel zoom anchored at the
  pointer, drag to pan, native two-finger pinch, and a fit control. Arrow keys inspect targets;
  Enter selects a legal target; plus/minus zoom and zero resets the view.
- Rendering, selection masks, pointer hit testing, and the diagnostic state share the same
  camera transform. Drag, pinch, cancelled gestures, changed choice states, AI turns, and
  replay views cannot accidentally submit a pointer selection.
- City-card hover reveals an off-screen town without resetting the chosen zoom. Highlight
  strokes stay bounded in screen pixels at high zoom, and labels avoid the map toolbar.
- Phone layout removes the spare vertical space around a fitted board and places active
  action controls ahead of player and market information. The first action row is visible
  at 390x844; 320x740 has no horizontal page overflow.

Verification: `output/map-navigation/playtest.mjs` and `result.json` cover desktop and mobile
canvas pixels, object details, zoom anchoring, drag suppression, native Chromium pinch/tap,
zoomed selection, keyboard selection, camera reset, card-driven reveal, and replay/AI locks.
Mock action submissions verify unchanged values for building, road, second road, sale, and
Beer-source choices. The real-game UI regression separately completed Build, Undo, Loan,
End Turn, and Develop x2 against the isolated backend. The 20-city preview test also passed.
Screenshots in the same directory were reviewed; browser errors were zero.
The production build and standard startup client passed. Keyboard inspection also updates
the accessible status region. The temporary backend is stopped; the preview remains on 5176.
