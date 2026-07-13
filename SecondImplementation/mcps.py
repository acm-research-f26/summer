"""
Monte Carlo Permutation Search (MCPS), from Cazenave, "Monte Carlo
Permutation Search" (2025) - the attached paper. MCPS extends the GRAVE
algorithm (itself a variant of MCTS/RAVE) with a THIRD statistic on top of
the usual two:

  1. Q(s, a)  - direct: playouts that took exactly this path, then move a.
  2. Q~(s, a) - AMAF/GRAVE: playouts that took this path, with move a
                appearing SOMEWHERE later (not necessarily immediately next),
                read from the nearest sufficiently-visited ANCESTOR (that's
                the "GRAVE" part - RAVE alone would use the node's own stats).
  3. Q^(s, a) - permutation (new in MCPS): pools ALL recorded playouts, from
                the very start of the search, that contain the SAME SET of
                moves as this path plus `a` - in ANY order, anywhere in the
                playout. Valid when permuting move order doesn't change the
                resulting state (exactly true for our game's "which unit do
                I decide first this turn" ordering - see below).

The three are combined with weights proportional to their sample counts
(the variance-minimizing convex combination under independence - see the
paper's derivation), so there's no bias hyperparameter to tune, only the
GRAVE-style visit threshold `rho`.

WHY THIS FITS THIS GAME WELL: a "move" here is one unit's [skill_idx,
target_idx] decision. A whole turn is 1-3 such decisions (one per living,
non-staggered unit) made in some order, after which the FULL joint command
list gets applied via Battle.resolve_turn() - and resolve_turn() only cares
about the final joint assignment, not what order we mentally considered the
units in. So a playout that decides unit A then unit B, and one that decides
B then A but reaches the same final assignment, are genuinely equivalent -
exactly the permutation-invariance MCPS is built to exploit (the same way
Hex doesn't care what order stones were placed in).

WHY MOVE CODES USE skill_number, NOT THE POSITIONAL SLOT INDEX: the boss
picks 3 *random* skills each turn, so "clash slot 1" doesn't mean the same
thing from one turn to the next. A move's code is
(unit_name, skill_idx, boss_skill_number-or-None) - using the boss's actual,
stable skill_number (from chosen_skills) instead of the turn-specific slot
position - so statistics about "clash Clash Baiter with my top skill" pool
correctly across turns regardless of which slot Clash Baiter happened to
land in that particular turn.

WHY THERE'S NO VALUE NEGATION (unlike the paper's Hex/Go/Wargame settings):
this game is single-decision-maker-with-chance-nodes, not 2-player-
adversarial. The boss is never tree-searched or "played against" - its
turn is sampled by Battle.boss_choose_turn()'s own fixed randomness, exactly
like a chance/environment node. Every value in the tree is from the same
(the party's) perspective throughout, so the paper's "negated for the
opponent's turn" doesn't apply here.

WHAT'S DELIBERATELY SIMPLIFIED vs. the paper: the paper's bitset/sliding-
window machinery (POPCOUNT over cyclic per-code bitsets) is a SPEED
optimization for engines doing hundreds of thousands of playouts/second
(bitboard Go/Hex). This engine does real per-turn Python bookkeeping
(status effects, stagger checks, logging), so realistic budgets here are
more like hundreds to a few thousand playouts per decision - at that scale,
plain Python sets for permutation matching are simpler, obviously correct,
and not the bottleneck. Likewise, the paper freezes a node's permutation
stats once at rho visits and caches them for its whole subtree; here Q^ is
just recomputed on demand from the global playout list, using the same
"nearest ancestor past rho visits" reference logic as Q~ for simplicity.
"""

import json
import random
from collections import defaultdict
import time

from ai_interface import run_headless_battle
from ai_interface import commands_to_player_actions


# ----------------------------------------------------------------------------
# Turn cursor: tracks "where we are" within a single turn's intra-turn
# sequence of per-unit decisions, for one in-progress simulated playout.
# Immutable-ish: apply_move() returns a NEW cursor rather than mutating.
# ----------------------------------------------------------------------------
class _TurnCursor:
    __slots__ = ("battle", "boss_slots", "actionable_names", "decided")

    def __init__(self, battle, boss_slots, decided=None):
        self.battle = battle
        self.boss_slots = boss_slots
        self.actionable_names = [u.name for u in battle.alive_units() if not u.is_staggered]
        self.decided = decided if decided is not None else {}

    def next_unit_name(self):
        """The next actionable unit (this turn) that still needs a decision,
        or None if everyone has one (the turn is ready to commit)."""
        for name in self.actionable_names:
            if name not in self.decided:
                return name
        return None

    def claimed_slots(self):
        return {target_idx - 1 for (_, target_idx) in self.decided.values() if target_idx != 0}

    def legal_moves_for(self, unit_name):
        """All legal (skill_idx, target_idx) pairs for this unit right now,
        given what's already been decided by other units earlier THIS turn."""
        claimed = self.claimed_slots()
        num_real_slots = len(self.boss_slots)
        moves = []
        for skill_idx in (0, 1):
            moves.append((skill_idx, 0))  # unopposed is always legal
            for slot_idx in range(num_real_slots):
                if slot_idx not in claimed:
                    moves.append((skill_idx, slot_idx + 1))
        return moves

    def move_code(self, unit_name, move):
        """(unit_name, skill_idx, boss_skill_number-or-None) - see module
        docstring for why this is used instead of the raw (skill_idx, target_idx)."""
        skill_idx, target_idx = move
        if target_idx == 0:
            boss_skill_number = None
        else:
            slot = self.boss_slots[target_idx - 1]
            boss_skill_number = self.battle.boss.skills.index(slot.skill_def)
        return (unit_name, skill_idx, boss_skill_number)

    def apply_move(self, unit_name, move):
        """Returns a new cursor with this unit's decision recorded (does not
        mutate self or touch the battle - nothing is "real" until commit_and_advance)."""
        new_decided = dict(self.decided)
        new_decided[unit_name] = move
        return _TurnCursor(self.battle, self.boss_slots, new_decided)

    def commit_and_advance(self):
        """
        Everyone's decided this turn - actually apply the commands to
        self.battle (a real mutation, but self.battle is always a throwaway
        clone during search) and return a cursor for the NEXT turn, or None
        if the battle is now over.
        """
        commands = [list(self.decided[name]) for name in self.actionable_names]
        actions = commands_to_player_actions(self.battle, self.boss_slots, commands)
        self.battle.resolve_turn(self.boss_slots, actions)
        if not self.battle.boss.is_alive() or not self.battle.alive_units():
            return None
        next_boss_slots = self.battle.boss_choose_turn()
        return _TurnCursor(self.battle, next_boss_slots)


def _default_rollout_policy(cursor, unit_name, legal_moves):
    """Uniform random - the simplest possible playout policy. Pass a smarter
    one (e.g. adapting balance_sim.py's heuristic) via MCPSSearch(rollout_policy=...).
    Draws from the battle's OWN rng (not the global `random` module) so that
    a whole search is reproducible given a fixed seed on the source battle."""
    return cursor.battle.rng.choice(legal_moves)


# ----------------------------------------------------------------------------
# Tree node
# ----------------------------------------------------------------------------
class MCPSNode:
    __slots__ = ("parent", "n", "total_reward", "children", "amaf_n", "amaf_total")

    def __init__(self, parent):
        self.parent = parent
        self.n = 0
        self.total_reward = 0.0
        self.children = {}  # concrete move (skill_idx, target_idx) -> MCPSNode
        self.amaf_n = defaultdict(int)          # code -> count (GRAVE/AMAF, Q~)
        self.amaf_total = defaultdict(float)

    def q(self):
        return self.total_reward / self.n if self.n else 0.0


def _find_reference_node(node, rho):
    """Nearest ancestor (inclusive) with visit count > rho - used for both
    Q~ (AMAF) and Q^ (permutation) lookups, same as GRAVE's own reference rule."""
    cur = node
    while cur.parent is not None and cur.n <= rho:
        cur = cur.parent
    return cur


# ----------------------------------------------------------------------------
# The search itself
# ----------------------------------------------------------------------------
class MCPSSearch:
    def __init__(self, rho=10, rollout_policy=None,
                 reward_win=1.0, reward_loss=-1.0, reward_draw=0.0, seed=None,
                 max_turns=60):
        self.rho = rho
        self.rollout_policy = rollout_policy or _default_rollout_policy
        self.reward_win = reward_win
        self.reward_loss = reward_loss
        self.reward_draw = reward_draw
        self.max_turns = max_turns  # safety cap - a playout that never
        # resolves (e.g. an unwinnable stalemate) is scored as a draw rather
        # than looping forever.
        self.root = MCPSNode(parent=None)
        # every completed playout's full set of codes + its reward - the
        # permutation statistics Q^ are computed on demand from this list.
        self.playout_records = []
        # Battle.clone() with no new_rng continues the SAME random sequence
        # the source would have - great for exact replay, but wrong here:
        # every playout needs its OWN independent stream, or repeated clones
        # of the same source battle would all draw identical "random" boss
        # turns/dice rolls, badly biasing the search's statistics. This
        # master_rng exists purely to hand each playout a fresh, independent
        # seed, while keeping the WHOLE search reproducible given `seed`.
        self.master_rng = random.Random(seed)

    def _q_hat(self, path_codes, candidate_code):
        needed = set(path_codes)
        needed.add(candidate_code)
        matching = [r for (codes, r) in self.playout_records if needed <= codes]
        if not matching:
            return 0, 0.0
        return len(matching), sum(matching) / len(matching)

    def _select_move(self, node, cursor, unit_name, legal_moves, path_codes_so_far):
        
        ref = _find_reference_node(node, self.rho)
        # Shuffle so that ties (e.g. every candidate having zero data early
        # in the search) resolve to a DIFFERENT move across playouts, using
        # this playout's own independent rng - not deterministically to
        # whichever move happens to be listed first every single time,
        # which would prevent ever exploring the alternatives at all.
        shuffled_moves = list(legal_moves)
        cursor.battle.rng.shuffle(shuffled_moves)
        best_key, best_move = None, None
        for move in shuffled_moves:
            code = cursor.move_code(unit_name, move)
            child = node.children.get(move)
            n = child.n if child else 0
            q = child.q() if child else 0.0

            n_tilde = ref.amaf_n.get(code, 0)
            q_tilde = (ref.amaf_total.get(code, 0.0) / n_tilde) if n_tilde else 0.0

            n_hat, q_hat = self._q_hat(path_codes_so_far, code)

            denom = n + n_tilde + n_hat
            val = (n * q + n_tilde * q_tilde + n_hat * q_hat) / denom if denom else 0.0

            # "A move with n~ = 0 is given priority over all others" (paper) -
            # explore totally AMAF-unseen moves before trusting val() at all.
            key = (0 if n_tilde == 0 else 1, -val)
            if best_key is None or key < best_key:
                best_key, best_move = key, move

        return best_move

    def run_one_playout(self, source_battle, boss_slots):
        """
        source_battle is cloned internally - with its OWN independent random
        stream (see master_rng above) - so it's never mutated itself, and
        no two playouts accidentally share identical "random" outcomes.
        """
        battle = source_battle.clone(new_rng=random.Random(self.master_rng.random()))
        cursor = _TurnCursor(battle, boss_slots)
        node = self.root
        tree_path = []       # [(node_before, move, code), ...] - the in-tree portion only
        all_codes = []       # every code seen, in order, across tree AND rollout
        expanded = False

        # --- Phase 1: selection, then exactly one expansion ---
        while not expanded:
            unit_name = cursor.next_unit_name()
            if unit_name is None:
                if battle.turn_number >= self.max_turns:
                    return self._finish_playout(battle, tree_path, all_codes, capped=True)
                next_cursor = cursor.commit_and_advance()
                if next_cursor is None:
                    return self._finish_playout(battle, tree_path, all_codes)
                cursor = next_cursor
                continue

            legal = cursor.legal_moves_for(unit_name)
            path_codes_so_far = [c for (_, _, c) in tree_path]
            move = self._select_move(node, cursor, unit_name, legal, path_codes_so_far)
            code = cursor.move_code(unit_name, move)
            all_codes.append(code)
            tree_path.append((node, move, code))

            if move in node.children:
                node = node.children[move]
            else:
                node.children[move] = MCPSNode(parent=node)
                node = node.children[move]
                expanded = True
            cursor = cursor.apply_move(unit_name, move)

        # --- Phase 2: rollout to a terminal state (or the turn cap) ---
        while True:
            unit_name = cursor.next_unit_name()
            if unit_name is None:
                if battle.turn_number >= self.max_turns:
                    return self._finish_playout(battle, tree_path, all_codes, capped=True)
                next_cursor = cursor.commit_and_advance()
                if next_cursor is None:
                    break
                cursor = next_cursor
                continue
            legal = cursor.legal_moves_for(unit_name)
            move = self.rollout_policy(cursor, unit_name, legal)
            all_codes.append(cursor.move_code(unit_name, move))
            cursor = cursor.apply_move(unit_name, move)

        return self._finish_playout(battle, tree_path, all_codes)

    def _finish_playout(self, battle, tree_path, all_codes, capped=False):
        if capped:
            reward = self.reward_draw  # hit max_turns without resolving - treat as a draw
        elif not battle.boss.is_alive():
            reward = self.reward_win
        elif not battle.alive_units():
            reward = self.reward_loss
        else:
            reward = self.reward_draw

        self.playout_records.append((frozenset(all_codes), reward))

        self.root.n += 1
        self.root.total_reward += reward
        for depth, (node_before, move, code) in enumerate(tree_path):
            child = node_before.children[move]
            child.n += 1
            child.total_reward += reward
            # AMAF: every code that appears AFTER this point in the full
            # played sequence (later tree moves + the whole rollout).
            later_codes = {c for (_, _, c) in tree_path[depth + 1:]}
            later_codes.update(all_codes[len(tree_path):])
            for lc in later_codes:
                node_before.amaf_n[lc] += 1
                node_before.amaf_total[lc] += reward
        return reward


# ----------------------------------------------------------------------------
# Public entry point: an ai_interface-compatible policy function
# ----------------------------------------------------------------------------
def make_mcps_policy(num_playouts
=500, rho=10, rollout_policy=None,
                      reward_win=1.0, reward_loss=-1.0, reward_draw=0.0, seed=None):
    """
    Returns a policy_fn(state, battle, boss_slots) suitable for
    ai_interface.run_headless_battle(...), backed by a FRESH Monte Carlo
    Permutation Search run from scratch at every real decision point (no
    tree reuse across turns, for simplicity).

    num_playouts: how many simulated games to run per real decision. More is
        stronger but slower - unlike the paper's per-move wall-clock time
        budgets, this is a fixed playout count, simplest to reason about.
    rho: GRAVE-style visit threshold for both the AMAF and permutation
        reference lookups (paper uses 30 for board games; this game's much
        smaller branching factor means a smaller value is often more
        appropriate - tune it for your situation).
    rollout_policy(cursor, unit_name, legal_moves) -> move: defaults to
        uniform random. Pass something smarter (e.g. adapted from
        balance_sim.py's heuristic) for stronger, more sample-efficient search.
    seed: makes the WHOLE policy's behavior reproducible across an episode -
        every turn's search gets its own seed, deterministically derived
        from this one, rather than sharing a single search's RNG across
        turns (each turn is a brand new MCPSSearch instance).
    """
    seed_rng = random.Random(seed)

    def policy_fn(state, battle, boss_slots):
        startTime = time.time()
        
        search = MCPSSearch(rho=rho, rollout_policy=rollout_policy,
                             reward_win=reward_win, reward_loss=reward_loss,
                             reward_draw=reward_draw, seed=seed_rng.random())
        for _ in range(num_playouts):
            search.run_one_playout(battle, boss_slots)

        # print(json.dumps(state, indent = 4))

        # Walk the tree along the most-visited ("robust") child at each of
        # this turn's decision levels to extract the actual commands -
        # using the REAL battle read-only here; nothing gets mutated since
        # we only ever call apply_move (never commit_and_advance) below.
        cursor = _TurnCursor(battle, boss_slots)
        node = search.root
        commands_by_unit = {}
        while True:
            unit_name = cursor.next_unit_name()
            if unit_name is None:
                break
            legal = cursor.legal_moves_for(unit_name)
            best_move, best_n = None, -1
            for move in legal:
                child = node.children.get(move)
                n = child.n if child else 0
                if n > best_n:
                    best_n, best_move = n, move
            commands_by_unit[unit_name] = list(best_move)
            node = node.children.get(best_move) or MCPSNode(parent=node)
            cursor = cursor.apply_move(unit_name, best_move)

        endTime = time.time()
        return [commands_by_unit[name] for name in cursor.actionable_names], endTime - startTime

    return policy_fn

numIterations = 20
numWins = 0
avgTurns = 0
avgTime = 0
for _ in range(numIterations):
    result = run_headless_battle(make_mcps_policy(), seed=42)
    if result["outcome"] == "win":
        numWins += 1
    
    avgTurns += result["turns"] / numIterations

    avgTime += result["timePerTurn"] / numIterations

print(f"Winrate was {numWins / numIterations}%, avg turns is {avgTurns}, avg time per turn was {avgTime}")