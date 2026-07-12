# Limbus Ripoff

A homebrew turn-based RPG battle system (3-person party vs. a boss), inspired
by Limbus Company/Library of Ruina-style clashing combat: each side rolls
dice within a skill-specific range, higher roll wins, and losing a clash
means eating the other side's hit instead of landing your own.

## File overview

**Core**
- `engine.py` — the whole rules engine: units, skills, clashing, damage,
  stagger, all status effects (tremor/burn/dark flame/magic bullets), and
  the skill-effect + passive-effect function libraries. No pygame, no
  display, no game-specific numbers — just mechanics. Every other file
  builds on this one. `Battle.clone()` makes a fully independent deep copy
  of an in-progress battle (state, RNG, everything) — the tool an AI needs
  to simulate a few turns ahead ("what happens if I try X?") on a throwaway
  copy before committing to a real decision. See its docstring for details,
  including how to fork off an independent random future vs. replaying the
  same one.
- `skill_queue.py` — the 8-slot bottom-to-top skill queue each party unit
  cycles through (which of its two skills are pickable this turn, which two
  are visible-but-not-yet, etc).
- `game_data.py` — **the single source of truth** for every tunable number
  (HP, stagger thresholds, base damage, roll ranges, the boss's clash-bonus
  values) and all the skill/passive tooltip text. `build_party_units()` and
  `build_boss()` are the only place party members and the boss actually get
  constructed — both the real game (`game_ui.py`) and the tuning harness
  (`balance_sim.py`) call into this instead of keeping their own copies, so
  they can never quietly drift out of sync with each other. Status-effect
  *magnitudes* (how much tremor/burn a skill applies, burst/conversion
  mechanics, etc.) are considered fixed design and live directly in
  `engine.py`'s effect functions instead — `game_data.py` only covers what
  balance tuning is meant to touch.

**Ways to play**
- `game_ui.py` — the visual, human-playable pygame version.
  Run with `python game_ui.py`. See the module docstring at the top of the
  file for full controls (arming skills, clashing, manual boss-targeting
  mode, etc).
- `ai_interface.py` — a headless, command-driven interface for scripting a
  bot/AI against the game — no pygame, no display, nothing visual at all.
  See its module docstring for the exact state/command format. Typical use:

  ```python
  from ai_interface import run_headless_battle

  def my_policy(state):
      return [[0, 0] for unit in state["units"] if unit["available_skills"]]

  result = run_headless_battle(my_policy, seed=42)
  print(result["outcome"], result["turns"])
  ```

  A policy function can optionally accept the live `Battle` object (and the
  boss's already-chosen skills for the turn) as extra arguments, to clone
  the battle and simulate a few moves ahead before deciding for real — see
  the "LOOKING AHEAD" section of the module docstring for a worked example.

**Tuning**
- `balance_sim.py` — simulates thousands of full battles against a
  heuristic "skilled" policy (and a deliberately unskilled "naive" one) to
  measure actual win rate for a given set of numbers, rather than guessing.
  Run with `python balance_sim.py` to see the current shipped numbers'
  measured win rate, or import `run_batch`/`DEFAULT_PARAMS` to experiment
  with different values yourself (see the module docstring for details on
  what the heuristic AI does and why).

**Tests**
- `run_tests.py` — runs every `test_*.py` file and prints one clean
  pass/fail summary. Run with `python run_tests.py`.
- `test_stagger.py`, `test_status_effects.py`, `test_turn_steps.py`,
  `test_clash_bonus.py`, `test_passives.py`, `test_battle_clone.py` —
  engine-level tests, each using small hand-built units/skills rather than
  the real game data, so they stay focused on one mechanic at a time
  (stagger, tremor/burn/dark-flame/bullets, the step-by-step
  turn-animation generator, the Clash Baiter clash bonus, the three party
  passives, and `Battle.clone()`'s lookahead-simulation guarantees,
  respectively).
- `test_ai_interface.py` — tests the state-building/command-parsing/
  validation logic in `ai_interface.py`.
- `test_ui_headless.py` — drives the actual `GameUI` class (clicks, arming,
  committing, the turn-resolution animation) using pygame's dummy video
  driver, so the whole UI can be tested without a real display.

**Assets**
- `assets/` — portrait sprites for the boss and the three party units,
  referenced by `game_ui.py`'s `SPRITE_FILES` mapping.

## Quick start

```bash
pip install pygame

python game_ui.py        # play it yourself
python balance_sim.py     # check the current win-rate numbers
python run_tests.py       # run the whole test suite
```
