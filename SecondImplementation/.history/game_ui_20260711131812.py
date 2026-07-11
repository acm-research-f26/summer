"""
Limbus Ripoff - Pygame UI

Controls:
  - Each unit shows a vertical stack of 4 skill circles: from bottom to top, that's
    BOTTOM (available now), TOP (available now), then two dimmed NEXT UP circles
    above those. Only BOTTOM/TOP are clickable. Colors mark skill1 (blue) vs
    skill2 (red) — including for the dimmed next-up circles.
  - Hover over ANY skill circle (available or next-up) to see its full effect
    text, including status-effect parts not yet implemented in the engine.
  - Hover over a unit's portrait (or the boss's) to see its passive.
  - Click BOTTOM or TOP to "arm" that specific skill (highlights yellow).
  - With a skill armed, click one of the boss's 3 skill slot boxes to CLASH it,
    OR click the "ATTACK UNOPPOSED" button under a unit to send it in unopposed.
  - Once committed, a line connects the armed circle to its target, and a
    small [x] next to the status text cancels the commitment.
  - Click "START TURN" once every living unit has an action assigned.
  - "MANUAL BOSS" checkbox: OFF = boss picks 3 random skills + random targets
    each turn (default). ON = you can click a slot's skill/target text to
    cycle through options BEFORE committing a clash to it. Once a slot has
    been claimed by a clash, it locks — you can no longer change its skill
    or target. This matches the actual rule: the boss keeps attacking its
    originally-chosen target unless someone clashes it, in which case the
    attack (if the boss wins the clash) redirects onto whoever clashed it.
  - Yellow ticks on an HP bar mark stagger thresholds; a unit/boss shows an
    orange ring + "STAGGERED" label while staggered (takes 1.5x damage, can't
    act, for the rest of that turn plus the entire next turn). A staggered
    unit doesn't need an action to START TURN — it's automatically skipped.
  - Once a turn starts, it plays out one atomic action at a time (each boss
    slot, then each unopposed player action, then end-of-turn status effects),
    with a white ring around whoever's involved and a banner describing what's
    happening (rolls + winner, for a clash). Click anywhere to advance to the
    next step; all normal controls are frozen until the turn finishes playing out.

Run locally with:  python game_ui.py
Requires: pygame  (pip install pygame)
"""

import sys
import os
import random
import pygame

from engine import (
    PlayerUnit, SkillDef, Boss, BossSkillDef, Battle, PlayerAction,
    effect_tremor_scorch_skill1, effect_tremor_scorch_skill2,
    effect_dark_flame_skill1, effect_dark_flame_skill2,
    effect_self_status_skill1, effect_self_status_skill2,
    effect_boss_tremor_slam, effect_boss_burn_wave, effect_boss_clash_baiter,
    effect_boss_scorch_point, effect_boss_amplitude_cascade,
)

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
MANUAL_BOSS_MODE = True  # <-- the bool toggle. Can also be flipped in-app via checkbox.

SCREEN_W, SCREEN_H = 1500, 720

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
SPRITE_FILES = {
    "Thumb East Capo III": "Meursault.png",
    "Lobotomy EGO: Magic Bullet": "Outis.png",
    "You Branch Adept": "Heathcliff.png",
    "Thumb East Capo II": "boss.png",
}
FPS = 60

COLOR_BG = (30, 30, 35)
COLOR_TEXT = (235, 235, 235)
COLOR_DIM_TEXT = (150, 150, 150)
COLOR_HP_BG = (70, 20, 20)
COLOR_HP_FG = (200, 60, 60)
COLOR_BOSS_HP_FG = (160, 40, 160)
COLOR_SKILL1 = (70, 130, 220)   # blue circles = skill1
COLOR_SKILL2 = (220, 90, 70)    # red circles = skill2
COLOR_ARMED = (250, 220, 60)
COLOR_SLOT_BG = (55, 55, 65)
COLOR_SLOT_CLAIMED = (90, 140, 90)
COLOR_SLOT_LOCKED_HINT = (140, 90, 90)
COLOR_BUTTON = (60, 120, 60)
COLOR_BUTTON_OFF = (90, 90, 90)
COLOR_PANEL = (45, 45, 52)
COLOR_CANCEL = (200, 70, 70)
COLOR_LINE = (250, 220, 60)
COLOR_UNOPPOSED_BTN = (80, 100, 140)
COLOR_TOOLTIP_BG = (20, 20, 24)
COLOR_TOOLTIP_BORDER = (200, 200, 100)
COLOR_STAGGER_TICK = (250, 210, 40)
COLOR_STAGGERED_RING = (255, 160, 30)
COLOR_STEP_HIGHLIGHT = (255, 255, 255)
COLOR_STAGGERED_TEXT = (255, 170, 40)
COLOR_TREMOR_NORMAL = (160, 130, 70)
COLOR_TREMOR_SCORCH = (210, 40, 130)
COLOR_BURN = (235, 120, 40)
COLOR_DARK_FLAME = (100, 50, 140)
COLOR_MAGIC_BULLET = (70, 170, 200)

FONT_NAME = None  # default pygame font


# ----------------------------------------------------------------------------
# GAME DATA — includes full effect text (even not-yet-implemented status
# effects) so tooltips can show the complete design intent.
# ----------------------------------------------------------------------------
def make_units():
    return [
        PlayerUnit(
            name="Thumb East Capo III",
            max_hp=130,
            skill1=SkillDef(
                "Tremor Jab", 14, 20, 25,
                description="Inflict 3 tremor potency and 2 tremor count. If tremor count "
                            "is now more than 3, tremor burst once. Rolls 14-20. Base damage 25.",
                effect=effect_tremor_scorch_skill1,
            ),
            skill2=SkillDef(
                "Tremor Burst Strike", 10, 16, 50,
                description="Inflict 2 tremor potency, perform tremor burst twice, and trigger "
                            "amplitude conversion of the tremor type to tremor scorch. "
                            "Rolls 10-16. Base damage 50.",
                effect=effect_tremor_scorch_skill2,
            ),
            passive_description="If opponent has +15 burn potency, roll +1.5 more (added to both "
                                 "lower and bigger bound). If opponent has +15 tremor potency, "
                                 "roll +1.5 more in the same way.",
            stagger_thresholds=[110],
        ),
        PlayerUnit(
            name="Lobotomy EGO: Magic Bullet",
            max_hp=150,
            skill1=SkillDef(
                "Flame Tag", 14, 20, 25,
                description="Inflict burn equal to current magic bullets, then inflict 1 dark "
                            "flame on target. Gain 2 magic bullets (max 7). Rolls 14-20. Base damage 25.",
                effect=effect_dark_flame_skill1,
            ),
            skill2=SkillDef(
                "Dark Flame Surge", 10, 16, 50,
                description="Gain 1 magic bullet (max 7), then inflict dark flame equal to "
                            "current magic bullets. Rolls 10-16. Base damage 50.",
                effect=effect_dark_flame_skill2,
            ),
            passive_description="If currently has 5+ magic bullets, roll +1.5 more (added to both "
                                 "lower and bigger bound). If opponent has +15 burn potency, "
                                 "roll +1.5 more (added to both lower and bigger bound).",
            stagger_thresholds=[160, 70, 30],
            max_magic_bullets=7,
        ),
        PlayerUnit(
            name="You Branch Adept",
            max_hp=100,
            skill1=SkillDef(
                "Shared Tremor", 14, 20, 25,
                description="Inflict 16 tremor potency and 8 tremor count on opponent, while "
                            "applying 1 tremor count and 5 tremor potency to self. "
                            "Rolls 14-20. Base damage 25.",
                effect=effect_self_status_skill1,
            ),
            skill2=SkillDef(
                "Shared Burn", 10, 16, 50,
                description="Inflict 10 burn potency and 5 burn count on self and target. Burn "
                            "cannot cause HP to go below 1. Base damage is 50 + burn potency. Rolls 10-16.",
                effect=effect_self_status_skill2,
            ),
            passive_description="If self has +10 tremor potency, roll +1.5 more and take 20% less "
                                 "damage. If self has +15 burn potency, roll +1.5 more and take "
                                 "20% less damage. Also, once per battle, if HP drops below zero "
                                 "(from any cause other than this unit's own self-burn), remove "
                                 "all burn and tremor on self and heal back to 80 HP.",
            stagger_thresholds=[150, 50],
        ),
    ]


def make_boss():
    skills = [
        BossSkillDef(
            "Tremor Slam", 12, 16, 15,
            description="On hit, inflict 5 tremor potency and 3 tremor count, then trigger "
                        "tremor burst. Damage 15. Rolls 12-16.",
            effect=effect_boss_tremor_slam,
        ),
        BossSkillDef(
            "Burn Wave", 10, 13, 10, hits_all=True,
            description="Hits ALL party members, inflicting 10 burn potency and 3 burn count "
                        "on each. Damage 10 to all members. Rolls 10-13.",
            effect=effect_boss_burn_wave,
        ),
        BossSkillDef(
            "Clash Baiter", 10, 14, 15,
            description="If this attack is clashed by another skill, gain +5 base power to "
                        "rolls and deal 900% more damage (not yet implemented). Inflicts 3 "
                        "tremor potency. Base damage 15. Rolls 10-14.",
            effect=effect_boss_clash_baiter,
        ),
        BossSkillDef(
            "Scorch Point", 15, 18, 15,
            description="Deal 10 burn potency to target. Base damage 15. Rolls 15-18.",
            effect=effect_boss_scorch_point,
        ),
        BossSkillDef(
            "Amplitude Cascade", 18, 20, 30, hits_all=True,
            description="Targets all enemies. Inflicts 1 tremor count and 1 tremor potency on "
                        "all, then bursts, while also triggering amplitude conversion into "
                        "tremor scorch. Base damage 30. Rolls 14-16.",
            effect=effect_boss_amplitude_cascade,
        ),
    ]
    return Boss(
        name="Boss", max_hp=1000, skills=skills,
        passive_description="Max HP 1500. Stagger thresholds at 1000 and 500. Picks 3 of its "
                             "5 skills at random each turn, each with a random target, unless "
                             "manually overridden.",
        stagger_thresholds=[1000, 500],
    )


# ----------------------------------------------------------------------------
# SMALL UI HELPERS
# ----------------------------------------------------------------------------
class Rect:
    """Thin wrapper so we can store arbitrary payloads alongside a pygame.Rect."""
    def __init__(self, rect, payload=None):
        self.rect = rect
        self.payload = payload

    def collide(self, pos):
        return self.rect.collidepoint(pos)


def draw_text(surface, font, text, pos, color=COLOR_TEXT, center=False):
    img = font.render(text, True, color)
    r = img.get_rect()
    if center:
        r.center = pos
    else:
        r.topleft = pos
    surface.blit(img, r)
    return r


def load_circular_sprite(path, diameter):
    """
    Loads an image, center-crops it to a square (so wide/tall source images
    don't get squished), scales it to `diameter`, and masks it to a circle
    so it drops cleanly into the existing round-portrait layout.
    """
    img = pygame.image.load(path).convert_alpha()
    w, h = img.get_size()
    side = min(w, h)
    crop_rect = pygame.Rect((w - side) // 2, (h - side) // 2, side, side)
    img = img.subsurface(crop_rect).copy()
    img = pygame.transform.smoothscale(img, (diameter, diameter))

    mask = pygame.Surface((diameter, diameter), pygame.SRCALPHA)
    pygame.draw.circle(mask, (255, 255, 255, 255), (diameter // 2, diameter // 2), diameter // 2)

    result = pygame.Surface((diameter, diameter), pygame.SRCALPHA)
    result.blit(img, (0, 0))
    result.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    return result


def draw_hp_bar(surface, x, y, w, h, hp, max_hp, fg_color, stagger_thresholds=None):
    hp_clamped = max(0, min(hp, max_hp))
    pygame.draw.rect(surface, COLOR_HP_BG, (x, y, w, h))
    if max_hp > 0:
        fill_w = int(w * (hp_clamped / max_hp))
        pygame.draw.rect(surface, fg_color, (x, y, fill_w, h))
    if stagger_thresholds and max_hp > 0:
        for threshold in stagger_thresholds:
            tx = x + int(w * (threshold / max_hp))
            tx = max(x, min(tx, x + w))
            pygame.draw.line(surface, COLOR_STAGGER_TICK, (tx, y - 2), (tx, y + h + 2), 3)
    pygame.draw.rect(surface, COLOR_TEXT, (x, y, w, h), 1)


def wrap_text(text, font, max_width):
    words = text.split(" ")
    lines = []
    current = ""
    for word in words:
        candidate = (current + " " + word).strip()
        if font.size(candidate)[0] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_tooltip(surface, font, text, mouse_pos, max_width=320):
    lines = wrap_text(text, font, max_width)
    line_h = font.get_height() + 3
    pad = 8
    box_w = max_width + pad * 2
    box_h = line_h * len(lines) + pad * 2

    x, y = mouse_pos[0] + 16, mouse_pos[1] + 16
    if x + box_w > SCREEN_W:
        x = SCREEN_W - box_w - 5
    if y + box_h > SCREEN_H:
        y = SCREEN_H - box_h - 5

    pygame.draw.rect(surface, COLOR_TOOLTIP_BG, (x, y, box_w, box_h))
    pygame.draw.rect(surface, COLOR_TOOLTIP_BORDER, (x, y, box_w, box_h), 1)
    for i, line in enumerate(lines):
        draw_text(surface, font, line, (x + pad, y + pad + i * line_h), color=COLOR_TEXT)


# ----------------------------------------------------------------------------
# MAIN GAME
# ----------------------------------------------------------------------------
class GameUI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("Limbus Ripoff - Prototype")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(FONT_NAME, 17)
        self.font_small = pygame.font.SysFont(FONT_NAME, 13)
        self.font_big = pygame.font.SysFont(FONT_NAME, 24, bold=True)

        self.rng = random.Random()
        self.units = make_units()
        self.boss = make_boss()
        self.battle = Battle(self.boss, self.units, rng=self.rng)

        self.manual_boss_mode = MANUAL_BOSS_MODE

        # Portrait sprites: keyed by entity name, pre-masked to circles at the
        # exact diameter each portrait is drawn at (boss r=38, party r=34).
        # Falls back to the plain colored-circle look if an image is missing.
        self.sprites = {}
        for name, filename in SPRITE_FILES.items():
            path = os.path.join(ASSETS_DIR, filename)
            diameter = (38 if name == "Boss" else 34) * 2
            try:
                self.sprites[name] = load_circular_sprite(path, diameter)
            except (pygame.error, FileNotFoundError) as e:
                print(f"[sprite] could not load {path}: {e}")
                self.sprites[name] = None

        self.boss_slots = None
        self.player_actions = {}  # unit_name -> PlayerAction
        self.armed = None  # (unit_name, position) currently armed, or None. position is 0 (bottom) or 1 (top).

        self.log_lines = []
        self.running = True

        self.clickables = []
        self.hover_regions = []  # list of Rect(), payload = tooltip text string

        self.skill_circle_pos = {}
        self.boss_slot_rects = {}
        self.unopposed_btn_rects = {}

        # --- turn resolution animation state ---
        self.phase = "planning"  # "planning" | "resolving"
        self.resolving_step_iter = None
        self.current_step = None

        self._start_new_planning_phase()

    # ---------------- turn / planning management ----------------

    def _start_new_planning_phase(self):
        self.boss_slots = self.battle.boss_choose_turn()
        self.player_actions = {}
        self.armed = None

    def _alive_unit_names(self):
        return [u.name for u in self.battle.alive_units()]

    def _all_actions_ready(self):
        needs_action = [
            u.name for u in self.battle.alive_units() if not u.is_staggered
        ]
        return all(name in self.player_actions for name in needs_action)

    def _claimed_slot_indices(self):
        return {
            a.clash_slot_index
            for a in self.player_actions.values()
            if a.clash_slot_index is not None
        }

    def _arm(self, unit_name, position):
        if unit_name in self.player_actions:
            del self.player_actions[unit_name]
        self.armed = (unit_name, position)

    def _commit_action(self, clash_slot_index):
        if self.armed is None:
            return
        unit_name, position = self.armed
        unit = self.battle.get_unit(unit_name)
        skill_type = unit.queue.available[position]
        self.player_actions[unit_name] = PlayerAction(
            unit_name, skill_type, clash_slot_index, armed_position=position
        )
        self.armed = None

    def _cancel_action(self, unit_name):
        if unit_name in self.player_actions:
            del self.player_actions[unit_name]
        if self.armed is not None and self.armed[0] == unit_name:
            self.armed = None

    def _start_turn(self):
        if not self._all_actions_ready() or not self.boss.is_alive():
            return
        actions = list(self.player_actions.values())
        self.resolving_step_iter = self.battle.resolve_turn_steps(self.boss_slots, actions)
        self.phase = "resolving"
        self._advance_step()

    def _advance_step(self):
        """Pulls the next TurnStep from the resolution generator and applies
        its log lines. Called once when a turn starts, then once per click
        while resolving. If the generator is exhausted, wraps up the turn
        and returns to planning."""
        try:
            step = next(self.resolving_step_iter)
        except StopIteration:
            self._finish_resolving_turn()
            return
        self.current_step = step
        for line in step.log_lines:
            self.log_lines.append(line)
        self.log_lines = self.log_lines[-200:]

    def _finish_resolving_turn(self):
        self.phase = "planning"
        self.current_step = None
        self.resolving_step_iter = None
        self._start_new_planning_phase()

    def _highlighted_names(self):
        if self.phase == "resolving" and self.current_step is not None:
            return set(self.current_step.participants)
        return set()

    def _highlighted_slot_index(self):
        if self.phase == "resolving" and self.current_step is not None:
            return self.current_step.slot_index
        return None

    def _cycle_boss_skill(self, slot_index):
        if not self.manual_boss_mode or slot_index in self._claimed_slot_indices():
            return
        slot = self.boss_slots[slot_index]
        current_idx = self.boss.skills.index(slot.skill_def)
        new_def = self.boss.skills[(current_idx + 1) % len(self.boss.skills)]
        alive_names = self._alive_unit_names()
        if new_def.hits_all:
            targets = list(alive_names)
        else:
            old_target = slot.target_names[0] if slot.target_names else None
            targets = [old_target] if old_target in alive_names else [alive_names[0]]
        from engine import PlannedBossSkill
        self.boss_slots[slot_index] = PlannedBossSkill(new_def, targets)

    def _cycle_boss_target(self, slot_index):
        if not self.manual_boss_mode or slot_index in self._claimed_slot_indices():
            return
        slot = self.boss_slots[slot_index]
        if slot.skill_def.hits_all:
            return
        alive_names = self._alive_unit_names()
        if not alive_names:
            return
        current = slot.target_names[0]
        if current in alive_names:
            idx = (alive_names.index(current) + 1) % len(alive_names)
        else:
            idx = 0
        slot.target_names = [alive_names[idx]]

    # ---------------- drawing ----------------

    def draw(self):
        self.screen.fill(COLOR_BG)
        self.clickables = []
        self.hover_regions = []
        self.skill_circle_pos = {}
        self.boss_slot_rects = {}
        self.unopposed_btn_rects = {}

        self._draw_header()
        self._draw_boss_area()
        self._draw_boss_slots()
        self._draw_party()
        self._draw_connections()
        self._draw_log_panel()
        self._draw_start_button()
        self._draw_current_step_banner()
        self._draw_tooltip_if_hovering()

        if self.phase == "resolving":
            # Freeze interaction during animation — visuals still fully render above,
            # but nothing should be clickable while a turn is playing out.
            self.clickables = []

        pygame.display.flip()

    def _draw_current_step_banner(self):
        if self.phase != "resolving" or self.current_step is None:
            return
        step = self.current_step

        if step.kind == "clash":
            clasher = step.participants[1]
            winner_text = "Player wins!" if step.winner == "player" else "Boss wins!"
            text = (f"CLASH — Boss's {step.boss_skill_name} rolled {step.boss_roll}  vs  "
                     f"{clasher}'s {step.player_skill_name} rolled {step.player_roll}  →  {winner_text}")
        elif step.kind == "boss_unopposed":
            targets = ", ".join(step.participants[1:])
            text = f"BOSS uses {step.boss_skill_name} unopposed on {targets}"
        elif step.kind == "player_unopposed":
            text = f"{step.participants[0]} attacks unopposed!"
        elif step.kind == "turn_end":
            text = "End of turn — status effects resolve..."
        else:
            text = ""

        banner_w, banner_h = 1000, 62
        x = (SCREEN_W - banner_w) // 2
        y = 390
        surf = pygame.Surface((banner_w, banner_h), pygame.SRCALPHA)
        surf.fill((20, 20, 24, 235))
        pygame.draw.rect(surf, COLOR_ARMED, surf.get_rect(), 2)
        self.screen.blit(surf, (x, y))
        draw_text(self.screen, self.font, text, (SCREEN_W // 2, y + 22), center=True)
        draw_text(self.screen, self.font_small, "(click anywhere to continue)",
                  (SCREEN_W // 2, y + 44), center=True, color=COLOR_DIM_TEXT)

    def _draw_tooltip_if_hovering(self):
        mouse_pos = pygame.mouse.get_pos()
        for hr in reversed(self.hover_regions):
            if hr.collide(mouse_pos):
                draw_tooltip(self.screen, self.font_small, hr.payload, mouse_pos)
                return

    def _draw_status_badges(self, cx, y, entity):
        """
        Draws small colored dot+label badges for active status effects, centered
        horizontally on cx, each with a hover tooltip explaining what it does.
        A tremor/burn badge disappears once its COUNT reaches 0 (an expired
        stack), even if some potency value happens to still be lingering.
        Tremor's dot color distinguishes normal tremor from tremor scorch.
        Magic bullets always shows for units that have the mechanic.
        """
        badges = []  # (color, text, tooltip)
        if entity.tremor_count > 0:
            is_scorch = entity.tremor_type == "scorch"
            color = COLOR_TREMOR_SCORCH if is_scorch else COLOR_TREMOR_NORMAL
            if is_scorch:
                tooltip = (
                    "TREMOR SCORCH (potency/count). An upgraded tremor from amplitude "
                    "conversion: every time it bursts (thresholds raised by potency, "
                    "staggering if HP now falls below one, count -1), it ALSO deals "
                    "physical damage equal to tremor potency + burn potency, and burn "
                    "count drops by 1. Reverts to normal tremor a couple turns after "
                    "conversion. Count also ticks down by 1 at the end of every turn; "
                    "hits 0 -> effect clears."
                )
            else:
                tooltip = (
                    "TREMOR (potency/count). When it 'bursts', raises all not-yet-passed "
                    "stagger thresholds by its potency (staggering the target if HP is "
                    "now below one), then count drops by 1. Count also ticks down by 1 "
                    "at the end of every turn. Hits 0 -> effect clears."
                )
            badges.append((color, f"{entity.tremor_potency}/{entity.tremor_count}", tooltip))

        if entity.burn_count > 0:
            tooltip = (
                "BURN (potency/count). At the end of every turn, deals damage equal to "
                "burn potency — unless dark flame is present, in which case it instead "
                "deals (dark flame x burn potency) damage and consumes the dark flame. "
                "Count drops by 1 each turn end. Hits 0 -> effect clears."
            )
            badges.append((COLOR_BURN, f"{entity.burn_potency}/{entity.burn_count}", tooltip))

        if entity.dark_flame_count != 0:
            tooltip = (
                "DARK FLAME (count). At the next end-of-turn burn tick, this replaces "
                "normal burn damage with (dark flame count x burn potency) damage "
                "instead, then is fully consumed."
            )
            badges.append((COLOR_DARK_FLAME, f"{entity.dark_flame_count}", tooltip))

        if getattr(entity, "max_magic_bullets", 0) > 0:
            tooltip = (
                f"MAGIC BULLETS (current/max {entity.max_magic_bullets}). A resource "
                "this unit spends to fuel its own skills — one skill converts current "
                "bullets into burn/dark flame amounts, the other grants more bullets."
            )
            badges.append((COLOR_MAGIC_BULLET, f"{entity.magic_bullets}/{entity.max_magic_bullets}", tooltip))

        if not badges:
            return

        dot_r = 6
        gap = 4
        group_gap = 14
        widths = []
        for color, text, tooltip in badges:
            widths.append(dot_r * 2 + gap + self.font_small.size(text)[0])
        total_w = sum(widths) + group_gap * (len(badges) - 1)
        x = cx - total_w // 2

        for (color, text, tooltip), w in zip(badges, widths):
            pygame.draw.circle(self.screen, color, (x + dot_r, y), dot_r)
            draw_text(self.screen, self.font_small, text, (x + dot_r * 2 + gap, y - 7), color=COLOR_TEXT)
            badge_rect = pygame.Rect(x - 2, y - dot_r - 2, w + 4, dot_r * 2 + 4)
            self.hover_regions.append(Rect(badge_rect, tooltip))
            x += w + group_gap
            x += w + group_gap

    def _draw_connections(self):
        for unit_name, action in self.player_actions.items():
            position = action.armed_position
            start = self.skill_circle_pos.get((unit_name, position))
            if start is None:
                continue
            if action.clash_slot_index is not None:
                target_rect = self.boss_slot_rects.get(action.clash_slot_index)
                if target_rect is None:
                    continue
                end = target_rect.midbottom
            else:
                target_rect = self.unopposed_btn_rects.get(unit_name)
                if target_rect is None:
                    continue
                end = target_rect.midtop
            pygame.draw.line(self.screen, COLOR_LINE, start, end, 2)

    def _draw_header(self):
        draw_text(self.screen, self.font_big, f"Turn {self.battle.turn_number + 1}", (20, 8))

        box_rect = pygame.Rect(20, 40, 18, 18)
        pygame.draw.rect(self.screen, COLOR_TEXT, box_rect, 2)
        if self.manual_boss_mode:
            pygame.draw.rect(self.screen, COLOR_ARMED, box_rect.inflate(-6, -6))
        draw_text(self.screen, self.font_small, "Manual Boss Targeting", (44, 41))
        self.clickables.append(Rect(box_rect.inflate(200, 8), ("toggle_manual", None)))

    def _draw_boss_area(self):
        cx, cy, r = SCREEN_W // 2, 85, 38
        sprite = self.sprites.get("Boss")
        if sprite is not None:
            self.screen.blit(sprite, (cx - r, cy - r))
        else:
            pygame.draw.circle(self.screen, (120, 40, 140), (cx, cy), r)
        if self.boss.is_staggered:
            pygame.draw.circle(self.screen, COLOR_STAGGERED_RING, (cx, cy), r + 5, 4)
        if self.boss.name in self._highlighted_names():
            pygame.draw.circle(self.screen, COLOR_STEP_HIGHLIGHT, (cx, cy), r + 10, 4)
        draw_text(self.screen, self.font, self.boss.name, (cx, cy - r - 14), center=True)
        draw_hp_bar(self.screen, cx - 100, cy + r + 8, 200, 12, self.boss.hp, self.boss.max_hp,
                    COLOR_BOSS_HP_FG, stagger_thresholds=self.boss.active_stagger_thresholds())
        draw_text(self.screen, self.font_small, f"{max(self.boss.hp,0)}/{self.boss.max_hp}",
                   (cx, cy + r + 24), center=True)
        self._draw_status_badges(cx, cy + r + 40, self.boss)
        if self.boss.is_staggered:
            draw_text(self.screen, self.font_small, "STAGGERED", (cx, cy), center=True,
                      color=COLOR_STAGGERED_TEXT)

        boss_rect = pygame.Rect(cx - r, cy - r, r * 2, r * 2)
        self.hover_regions.append(Rect(boss_rect, self.boss.passive_description))

    def _draw_boss_slots(self):
        y = 180
        slot_w, slot_h = 440, 78
        gap = 30
        total_w = slot_w * 3 + gap * 2
        start_x = (SCREEN_W - total_w) // 2

        claimed = self._claimed_slot_indices()

        for i, slot in enumerate(self.boss_slots):
            x = start_x + i * (slot_w + gap)
            rect = pygame.Rect(x, y, slot_w, slot_h)
            is_claimed = i in claimed
            bg = COLOR_SLOT_CLAIMED if is_claimed else COLOR_SLOT_BG
            pygame.draw.rect(self.screen, bg, rect, border_radius=8)
            pygame.draw.rect(self.screen, COLOR_TEXT, rect, 2, border_radius=8)
            if i == self._highlighted_slot_index():
                pygame.draw.rect(self.screen, COLOR_STEP_HIGHLIGHT, rect.inflate(8, 8), 3, border_radius=10)

            skill_label = f"{slot.skill_def.name}"
            roll_label = f"roll {slot.skill_def.roll_lo}-{slot.skill_def.roll_hi}   dmg {slot.skill_def.base_damage}"
            target_label = "ALL" if slot.skill_def.hits_all else ", ".join(slot.target_names)

            skill_rect = draw_text(self.screen, self.font, skill_label, (x + 12, y + 6))
            draw_text(self.screen, self.font_small, roll_label, (x + 12, y + 27), color=COLOR_DIM_TEXT)
            target_rect = draw_text(self.screen, self.font_small, f"-> {target_label}", (x + 12, y + 45))

            self.clickables.append(Rect(rect, ("clash_slot", i)))
            self.boss_slot_rects[i] = rect
            self.hover_regions.append(Rect(rect, slot.skill_def.description))

            if is_claimed:
                draw_text(self.screen, self.font_small, "(locked: claimed by a clash)",
                          (x + 12, y + 62), color=COLOR_SLOT_LOCKED_HINT)
            elif self.manual_boss_mode:
                draw_text(self.screen, self.font_small, "(click name/target to cycle)",
                          (x + 12, y + 62), color=COLOR_DIM_TEXT)
                self.clickables.append(Rect(skill_rect.inflate(14, 8), ("cycle_boss_skill", i)))
                self.clickables.append(Rect(target_rect.inflate(14, 8), ("cycle_boss_target", i)))

    def _draw_party(self):
        y_portrait = 325
        r = 34
        spacing = SCREEN_W // (len(self.units) + 1)

        for idx, unit in enumerate(self.units):
            cx = spacing * (idx + 1)
            alive = unit.is_alive()
            sprite = self.sprites.get(unit.name)
            if sprite is not None:
                self.screen.blit(sprite, (cx - r, y_portrait - r))
                if not alive:
                    # dim dead units with a translucent dark overlay on top of the sprite
                    overlay = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
                    pygame.draw.circle(overlay, (20, 20, 20, 190), (r, r), r)
                    self.screen.blit(overlay, (cx - r, y_portrait - r))
            else:
                color = (70, 150, 200) if alive else (60, 60, 60)
                pygame.draw.circle(self.screen, color, (cx, y_portrait), r)
            if alive and unit.is_staggered:
                pygame.draw.circle(self.screen, COLOR_STAGGERED_RING, (cx, y_portrait), r + 5, 4)
            if unit.name in self._highlighted_names():
                pygame.draw.circle(self.screen, COLOR_STEP_HIGHLIGHT, (cx, y_portrait), r + 10, 4)
            draw_text(self.screen, self.font, unit.name, (cx, y_portrait - r - 15), center=True)
            draw_hp_bar(self.screen, cx - 65, y_portrait + r + 7, 130, 11, unit.hp, unit.max_hp,
                        COLOR_HP_FG, stagger_thresholds=unit.active_stagger_thresholds())
            draw_text(self.screen, self.font_small, f"{max(unit.hp,0)}/{unit.max_hp}",
                       (cx, y_portrait + r + 21), center=True)
            self._draw_status_badges(cx, y_portrait + r + 37, unit)

            portrait_rect = pygame.Rect(cx - r, y_portrait - r, r * 2, r * 2)
            self.hover_regions.append(Rect(portrait_rect, unit.passive_description))

            if not alive:
                draw_text(self.screen, self.font_small, "DOWN", (cx, y_portrait), center=True, color=(255, 80, 80))
                continue

            status_y = y_portrait + r + 56

            if unit.is_staggered:
                draw_text(self.screen, self.font_small, "STAGGERED", (cx, y_portrait), center=True,
                          color=COLOR_STAGGERED_TEXT)
                draw_text(self.screen, self.font_small, "(cannot act this turn)",
                          (cx, status_y), center=True, color=COLOR_DIM_TEXT)
                continue

            action = self.player_actions.get(unit.name)
            if action is not None:
                status = f"-> Clash slot {action.clash_slot_index}" if action.clash_slot_index is not None else "-> Unopposed"
                status_rect = draw_text(self.screen, self.font_small, status, (cx, status_y), center=True)
                cancel_rect = pygame.Rect(status_rect.right + 6, status_rect.top - 2, 17, 17)
                pygame.draw.rect(self.screen, COLOR_CANCEL, cancel_rect, border_radius=3)
                draw_text(self.screen, self.font_small, "x", cancel_rect.center, center=True)
                self.clickables.append(Rect(cancel_rect, ("cancel_action", unit.name)))
            else:
                draw_text(self.screen, self.font_small, "(no action yet)", (cx, status_y), center=True,
                          color=COLOR_DIM_TEXT)

            self._draw_unit_skills(unit, cx, status_y + 20)

            btn_y = status_y + 20 + 4 * 34 + 8
            btn_rect = pygame.Rect(cx - 68, btn_y, 136, 26)
            already_acted = unit.name in self.player_actions
            can_click = (self.armed is not None and self.armed[0] == unit.name) and not already_acted
            btn_color = COLOR_UNOPPOSED_BTN if can_click else COLOR_BUTTON_OFF
            pygame.draw.rect(self.screen, btn_color, btn_rect, border_radius=6)
            draw_text(self.screen, self.font_small, "ATTACK UNOPPOSED", btn_rect.center, center=True)
            self.unopposed_btn_rects[unit.name] = btn_rect
            if can_click:
                self.clickables.append(Rect(btn_rect, ("attack_unopposed", unit.name)))

    def _draw_unit_skills(self, unit, cx, top_y):
        """
        Vertical stack. Screen top-to-bottom: next_up[1], next_up[0], TOP(available[1]),
        BOTTOM(available[0]) — bottom-of-screen = bottom-of-queue, matching the
        "bottom gets used/shifted first" semantics.
        """
        sr = 13
        row_h = 34
        available = unit.queue.available   # [bottom, top]
        next_up = unit.queue.next_up       # [next-bottom, next-top]

        rows = [
            (next_up[1], None, "next"),
            (next_up[0], None, "next"),
            (available[1], 1, "TOP"),
            (available[0], 0, "BOTTOM"),
        ]

        already_acted = unit.name in self.player_actions

        for i, (skill_type, position, label) in enumerate(rows):
            y = top_y + i * row_h
            base_color = COLOR_SKILL1 if skill_type == "skill1" else COLOR_SKILL2
            is_clickable = position is not None
            skill_def = unit.get_skill_def(skill_type)

            circle_rect = pygame.Rect(cx - sr, y - sr, sr * 2, sr * 2)

            if is_clickable:
                is_armed = self.armed == (unit.name, position)
                ring_color = COLOR_ARMED if is_armed else COLOR_TEXT
                pygame.draw.circle(self.screen, base_color, (cx, y), sr)
                pygame.draw.circle(self.screen, ring_color, (cx, y), sr, 3 if is_armed else 1)
                self.skill_circle_pos[(unit.name, position)] = (cx, y)
                if not already_acted:
                    self.clickables.append(Rect(circle_rect, ("arm_skill", (unit.name, position))))
            else:
                dim_color = tuple(int(c * 0.45) for c in base_color)
                pygame.draw.circle(self.screen, dim_color, (cx, y), sr - 4)

            self.hover_regions.append(Rect(circle_rect.inflate(6, 6), skill_def.description))

            label_text = f"{label}: {skill_def.name[:14]}" if is_clickable else f"({label})"
            draw_text(self.screen, self.font_small, label_text, (cx + sr + 8, y - 7),
                      color=COLOR_TEXT if is_clickable else COLOR_DIM_TEXT)

    def _draw_log_panel(self):
        panel_h = 70
        panel_y = SCREEN_H - panel_h - 10
        panel_rect = pygame.Rect(20, panel_y, SCREEN_W - 40, panel_h)
        pygame.draw.rect(self.screen, COLOR_PANEL, panel_rect)
        pygame.draw.rect(self.screen, COLOR_TEXT, panel_rect, 1)
        lines = self.log_lines[-4:]
        for i, line in enumerate(lines):
            draw_text(self.screen, self.font_small, line, (30, panel_y + 6 + i * 15), color=COLOR_DIM_TEXT)

    def _draw_start_button(self):
        ready = self._all_actions_ready() and self.boss.is_alive()
        rect = pygame.Rect(SCREEN_W - 180, 12, 160, 38)
        color = COLOR_BUTTON if ready else COLOR_BUTTON_OFF
        pygame.draw.rect(self.screen, color, rect, border_radius=6)
        draw_text(self.screen, self.font, "START TURN", rect.center, center=True)
        if ready:
            self.clickables.append(Rect(rect, ("start_turn", None)))

        if not self.boss.is_alive():
            draw_text(self.screen, self.font_big, "BOSS DEFEATED", (SCREEN_W // 2, 250), center=True, color=(255, 220, 80))

    # ---------------- event handling ----------------

    def handle_click(self, pos):
        if self.phase == "resolving":
            # Animation is click-to-advance: clickables are frozen during
            # resolution (see draw()), so any click just moves to the next step.
            self._advance_step()
            return

        for c in reversed(self.clickables):
            if c.collide(pos):
                kind, payload = c.payload
                if kind == "toggle_manual":
                    self.manual_boss_mode = not self.manual_boss_mode
                elif kind == "arm_skill":
                    unit_name, position = payload
                    self._arm(unit_name, position)
                elif kind == "clash_slot":
                    self._commit_action(clash_slot_index=payload)
                elif kind == "attack_unopposed":
                    self._commit_action(clash_slot_index=None)
                elif kind == "cancel_action":
                    self._cancel_action(payload)
                elif kind == "start_turn":
                    self._start_turn()
                elif kind == "cycle_boss_skill":
                    self._cycle_boss_skill(payload)
                elif kind == "cycle_boss_target":
                    self._cycle_boss_target(payload)
                return

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.handle_click(event.pos)

            self.draw()
            self.clock.tick(FPS)

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    GameUI().run()
