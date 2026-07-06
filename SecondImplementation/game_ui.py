"""
Limbus Ripoff - Pygame UI

Controls:
  - Each unit shows a vertical stack of 4 skill circles: from bottom to top, that's
    BOTTOM (available now), TOP (available now), then two dimmed NEXT UP circles
    above those. Only BOTTOM/TOP are clickable. Colors mark skill1 (blue) vs
    skill2 (red) — including for the dimmed next-up circles, so you can always
    tell what's coming.
  - Click BOTTOM or TOP to "arm" that specific skill slot (it'll highlight yellow).
    Even if both slots happen to be the same skill type, arming one only
    highlights that one, not both.
  - With a skill armed, click one of the boss's 3 skill slot boxes to CLASH it,
    OR click the "ATTACK UNOPPOSED" button under a unit to send it in unopposed.
  - Once a unit has a committed action, a line is drawn from its armed skill
    circle to whatever it's tied to (a boss slot, or the unopposed button), and
    a small [x] appears next to its status text — click that to undo the
    commitment and re-arm.
  - Click "START TURN" once every living unit has an action assigned.
  - Click the "MANUAL BOSS" checkbox to toggle manual boss targeting.
      - OFF (default): boss picks 3 random skills + random targets each turn (existing engine logic).
      - ON: boss still starts from a random pick, but you can click a slot's skill name to
        cycle through the 5 boss skills, and click its target name to cycle through targets,
        before starting the turn.

Run locally with:  python game_ui.py
Requires: pygame  (pip install pygame)
"""

import sys
import random
import pygame

from engine import PlayerUnit, SkillDef, Boss, BossSkillDef, Battle, PlayerAction

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
MANUAL_BOSS_MODE = True  # <-- the bool toggle. Can also be flipped in-app via checkbox.

SCREEN_W, SCREEN_H = 1100, 820
FPS = 60

COLOR_BG = (30, 30, 35)
COLOR_TEXT = (235, 235, 235)
COLOR_DIM_TEXT = (150, 150, 150)
COLOR_HP_BG = (70, 20, 20)
COLOR_HP_FG = (200, 60, 60)
COLOR_BOSS_HP_FG = (160, 40, 160)
COLOR_SKILL1 = (70, 130, 220)   # blue circles = skill1
COLOR_SKILL2 = (220, 90, 70)    # red circles = skill2
COLOR_NEXTUP = (90, 90, 100)
COLOR_ARMED = (250, 220, 60)
COLOR_SLOT_BG = (55, 55, 65)
COLOR_SLOT_CLAIMED = (90, 140, 90)
COLOR_BUTTON = (60, 120, 60)
COLOR_BUTTON_OFF = (90, 90, 90)
COLOR_PANEL = (45, 45, 52)
COLOR_CANCEL = (200, 70, 70)
COLOR_LINE = (250, 220, 60)
COLOR_UNOPPOSED_BTN = (80, 100, 140)

FONT_NAME = None  # default pygame font


# ----------------------------------------------------------------------------
# GAME DATA (same as demo.py)
# ----------------------------------------------------------------------------
def make_units():
    return [
        PlayerUnit(
            name="TremorScorch",
            max_hp=130,
            skill1=SkillDef("Tremor Jab", 14, 20, 25),
            skill2=SkillDef("Tremor Burst Strike", 10, 16, 50),
        ),
        PlayerUnit(
            name="DarkFlameInflictor",
            max_hp=150,
            skill1=SkillDef("Flame Tag", 14, 20, 25),
            skill2=SkillDef("Dark Flame Surge", 10, 16, 50),
        ),
        PlayerUnit(
            name="BurnTremorInflictor",
            max_hp=100,
            skill1=SkillDef("Shared Tremor", 14, 20, 25),
            skill2=SkillDef("Shared Burn", 10, 16, 50),
        ),
    ]


def make_boss():
    skills = [
        BossSkillDef("Tremor Slam", 12, 16, 25),
        BossSkillDef("Burn Wave", 10, 13, 10, hits_all=True),
        BossSkillDef("Clash Baiter", 10, 14, 15),
        BossSkillDef("Scorch Point", 15, 18, 20),
        BossSkillDef("Amplitude Cascade", 18, 20, 40, hits_all=True),
    ]
    return Boss(name="Boss", max_hp=1500, skills=skills)


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


def draw_hp_bar(surface, x, y, w, h, hp, max_hp, fg_color):
    hp_clamped = max(0, min(hp, max_hp))
    pygame.draw.rect(surface, COLOR_HP_BG, (x, y, w, h))
    if max_hp > 0:
        fill_w = int(w * (hp_clamped / max_hp))
        pygame.draw.rect(surface, fg_color, (x, y, fill_w, h))
    pygame.draw.rect(surface, COLOR_TEXT, (x, y, w, h), 1)


# ----------------------------------------------------------------------------
# MAIN GAME
# ----------------------------------------------------------------------------
class GameUI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("Limbus Ripoff - Prototype")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(FONT_NAME, 18)
        self.font_small = pygame.font.SysFont(FONT_NAME, 14)
        self.font_big = pygame.font.SysFont(FONT_NAME, 26, bold=True)

        self.rng = random.Random()
        self.units = make_units()
        self.boss = make_boss()
        self.battle = Battle(self.boss, self.units, rng=self.rng)

        self.manual_boss_mode = MANUAL_BOSS_MODE

        self.boss_slots = None
        self.player_actions = {}  # unit_name -> PlayerAction (may be incomplete/mid-build)
        self.armed = None  # (unit_name, position) currently armed, or None. position is 0 (bottom) or 1 (top).

        self.log_lines = []  # rolling log for on-screen display
        self.running = True

        self.clickables = []  # rebuilt every frame: list of Rect() for hit-testing

        # populated each draw() call so we can draw connecting lines and hit-test cancel buttons
        self.skill_circle_pos = {}  # (unit_name, position) -> (x, y) on screen
        self.boss_slot_rects = {}   # slot_index -> pygame.Rect
        self.unopposed_btn_rects = {}  # unit_name -> pygame.Rect

        self._start_new_planning_phase()

    # ---------------- turn / planning management ----------------

    def _start_new_planning_phase(self):
        self.boss_slots = self.battle.boss_choose_turn()
        self.player_actions = {}
        self.armed = None

    def _alive_unit_names(self):
        return [u.name for u in self.battle.alive_units()]

    def _all_actions_ready(self):
        alive = self._alive_unit_names()
        return all(name in self.player_actions for name in alive)

    def _claimed_slot_indices(self):
        return {
            a.clash_slot_index
            for a in self.player_actions.values()
            if a.clash_slot_index is not None
        }

    def _arm(self, unit_name, position):
        # arming a new skill for a unit clears any previous action for that unit
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
        log = self.battle.resolve_turn(self.boss_slots, actions)
        for line in log.dump().split("\n"):
            self.log_lines.append(line)
        self.log_lines = self.log_lines[-200:]  # cap memory
        self._start_new_planning_phase()

    def _cycle_boss_skill(self, slot_index):
        if not self.manual_boss_mode:
            return
        slot = self.boss_slots[slot_index]
        current_idx = self.boss.skills.index(slot.skill_def)
        new_def = self.boss.skills[(current_idx + 1) % len(self.boss.skills)]
        alive_names = self._alive_unit_names()
        if new_def.hits_all:
            targets = list(alive_names)
        else:
            # keep old target if still valid & not hits_all, else default to first alive
            old_target = slot.target_names[0] if slot.target_names else None
            targets = [old_target] if old_target in alive_names else [alive_names[0]]
        from engine import PlannedBossSkill
        self.boss_slots[slot_index] = PlannedBossSkill(new_def, targets)

    def _cycle_boss_target(self, slot_index):
        if not self.manual_boss_mode:
            return
        slot = self.boss_slots[slot_index]
        if slot.skill_def.hits_all:
            return  # AoE always targets all, not cyclable
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

        pygame.display.flip()

    def _draw_connections(self):
        """Draw a line from each committed unit's armed skill circle to its target."""
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
        draw_text(self.screen, self.font_big, f"Turn {self.battle.turn_number + 1}", (20, 15))

        # manual boss mode checkbox
        box_rect = pygame.Rect(20, 55, 20, 20)
        pygame.draw.rect(self.screen, COLOR_TEXT, box_rect, 2)
        if self.manual_boss_mode:
            pygame.draw.rect(self.screen, COLOR_ARMED, box_rect.inflate(-8, -8))
        draw_text(self.screen, self.font, "Manual Boss Targeting", (48, 57))
        self.clickables.append(Rect(box_rect.inflate(200, 10), ("toggle_manual", None)))

    def _draw_boss_area(self):
        cx, cy, r = SCREEN_W // 2, 110, 45
        pygame.draw.circle(self.screen, (120, 40, 140), (cx, cy), r)
        draw_text(self.screen, self.font, self.boss.name, (cx, cy - r - 15), center=True)
        draw_hp_bar(self.screen, cx - 100, cy + r + 8, 200, 14, self.boss.hp, self.boss.max_hp, COLOR_BOSS_HP_FG)
        draw_text(self.screen, self.font_small, f"{max(self.boss.hp,0)}/{self.boss.max_hp}",
                   (cx, cy + r + 15), center=True)

        boss_rect = pygame.Rect(cx - r, cy - r, r * 2, r * 2)
        self.clickables.append(Rect(boss_rect, ("boss_portrait", None)))

    def _draw_boss_slots(self):
        y = 200
        slot_w, slot_h = 300, 70
        gap = 30
        total_w = slot_w * 3 + gap * 2
        start_x = (SCREEN_W - total_w) // 2

        claimed = self._claimed_slot_indices()

        for i, slot in enumerate(self.boss_slots):
            x = start_x + i * (slot_w + gap)
            rect = pygame.Rect(x, y, slot_w, slot_h)
            bg = COLOR_SLOT_CLAIMED if i in claimed else COLOR_SLOT_BG
            pygame.draw.rect(self.screen, bg, rect, border_radius=6)
            pygame.draw.rect(self.screen, COLOR_TEXT, rect, 2, border_radius=6)

            skill_label = f"{slot.skill_def.name}  (roll {slot.skill_def.roll_lo}-{slot.skill_def.roll_hi})"
            target_label = "ALL" if slot.skill_def.hits_all else ", ".join(slot.target_names)

            skill_rect = draw_text(self.screen, self.font_small, skill_label, (x + 10, y + 10))
            target_rect = draw_text(self.screen, self.font_small, f"-> {target_label}", (x + 10, y + 32))
            draw_text(self.screen, self.font_small, f"dmg {slot.skill_def.base_damage}", (x + 10, y + 52))

            self.clickables.append(Rect(rect, ("clash_slot", i)))
            self.boss_slot_rects[i] = rect

            if self.manual_boss_mode:
                # small hint text + separate finer hit-areas for cycling skill/target
                self.clickables.append(Rect(skill_rect.inflate(10, 6), ("cycle_boss_skill", i)))
                self.clickables.append(Rect(target_rect.inflate(10, 6), ("cycle_boss_target", i)))

        if self.manual_boss_mode:
            draw_text(self.screen, self.font_small,
                      "(manual mode: click a skill name to cycle skill, target name to cycle target)",
                      (start_x, y + slot_h + 6), color=COLOR_DIM_TEXT)

    def _draw_party(self):
        y_portrait = 330
        r = 40
        spacing = SCREEN_W // (len(self.units) + 1)

        for idx, unit in enumerate(self.units):
            cx = spacing * (idx + 1)
            alive = unit.is_alive()
            color = (70, 150, 200) if alive else (60, 60, 60)
            pygame.draw.circle(self.screen, color, (cx, y_portrait), r)
            draw_text(self.screen, self.font, unit.name, (cx, y_portrait - r - 15), center=True)
            draw_hp_bar(self.screen, cx - 70, y_portrait + r + 8, 140, 12, unit.hp, unit.max_hp, COLOR_HP_FG)
            draw_text(self.screen, self.font_small, f"{max(unit.hp,0)}/{unit.max_hp}",
                       (cx, y_portrait + r + 24), center=True)

            if not alive:
                draw_text(self.screen, self.font_small, "DOWN", (cx, y_portrait), center=True, color=(255,80,80))
                continue

            # current action status + cancel button
            action = self.player_actions.get(unit.name)
            status_y = y_portrait + r + 42
            if action is not None:
                if action.clash_slot_index is not None:
                    status = f"-> Clash slot {action.clash_slot_index}"
                else:
                    status = "-> Unopposed"
                status_rect = draw_text(self.screen, self.font_small, status, (cx, status_y), center=True)
                cancel_rect = pygame.Rect(status_rect.right + 6, status_rect.top - 2, 18, 18)
                pygame.draw.rect(self.screen, COLOR_CANCEL, cancel_rect, border_radius=3)
                draw_text(self.screen, self.font_small, "x", cancel_rect.center, center=True)
                self.clickables.append(Rect(cancel_rect, ("cancel_action", unit.name)))
            else:
                draw_text(self.screen, self.font_small, "(no action yet)", (cx, status_y), center=True,
                          color=COLOR_DIM_TEXT)

            # skill stack (clickable bottom/top + dimmed next-up), drawn below status
            self._draw_unit_skills(unit, cx, status_y + 25)

            # explicit "attack unopposed" button, drawn below the skill stack
            btn_y = status_y + 25 + 150
            btn_rect = pygame.Rect(cx - 65, btn_y, 130, 28)
            already_acted = unit.name in self.player_actions
            can_click = (self.armed is not None and self.armed[0] == unit.name) and not already_acted
            btn_color = COLOR_UNOPPOSED_BTN if can_click else COLOR_BUTTON_OFF
            pygame.draw.rect(self.screen, btn_color, btn_rect, border_radius=5)
            draw_text(self.screen, self.font_small, "ATTACK UNOPPOSED", btn_rect.center, center=True)
            self.unopposed_btn_rects[unit.name] = btn_rect
            if can_click:
                self.clickables.append(Rect(btn_rect, ("attack_unopposed", unit.name)))

    def _draw_unit_skills(self, unit, cx, top_y):
        """
        Vertical stack, TOP of screen = topmost skill, BOTTOM of screen = bottom
        skill (matches the "bottom gets used/shifted first" queue semantics).
        Order drawn top-to-bottom: next_up[1], next_up[0], TOP(available[1]), BOTTOM(available[0]).
        """
        sr = 16
        row_h = 42
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

            if is_clickable:
                is_armed = self.armed == (unit.name, position)
                ring_color = COLOR_ARMED if is_armed else COLOR_TEXT
                pygame.draw.circle(self.screen, base_color, (cx, y), sr)
                pygame.draw.circle(self.screen, ring_color, (cx, y), sr, 3 if is_armed else 1)
                self.skill_circle_pos[(unit.name, position)] = (cx, y)
                if not already_acted:
                    rect = pygame.Rect(cx - sr, y - sr, sr * 2, sr * 2)
                    self.clickables.append(Rect(rect, ("arm_skill", (unit.name, position))))
            else:
                # dimmed, non-clickable next-up preview; still color-coded by skill type
                dim_color = tuple(int(c * 0.45) for c in base_color)
                pygame.draw.circle(self.screen, dim_color, (cx, y), sr - 4)

            label_text = f"{label}: {skill_def.name[:12]}" if is_clickable else f"({label})"
            draw_text(self.screen, self.font_small, label_text, (cx + sr + 8, y - 8),
                      color=COLOR_TEXT if is_clickable else COLOR_DIM_TEXT)

    def _draw_log_panel(self):
        panel_h = 95
        panel_y = SCREEN_H - panel_h - 15
        panel_rect = pygame.Rect(20, panel_y, SCREEN_W - 40, panel_h)
        pygame.draw.rect(self.screen, COLOR_PANEL, panel_rect)
        pygame.draw.rect(self.screen, COLOR_TEXT, panel_rect, 1)
        lines = self.log_lines[-5:]
        for i, line in enumerate(lines):
            draw_text(self.screen, self.font_small, line, (30, panel_y + 8 + i * 18), color=COLOR_DIM_TEXT)

    def _draw_start_button(self):
        ready = self._all_actions_ready() and self.boss.is_alive()
        rect = pygame.Rect(SCREEN_W - 180, 15, 160, 40)
        color = COLOR_BUTTON if ready else COLOR_BUTTON_OFF
        pygame.draw.rect(self.screen, color, rect, border_radius=6)
        draw_text(self.screen, self.font, "START TURN", rect.center, center=True)
        if ready:
            self.clickables.append(Rect(rect, ("start_turn", None)))

        if not self.boss.is_alive():
            draw_text(self.screen, self.font_big, "BOSS DEFEATED", (SCREEN_W // 2, 350), center=True, color=(255, 220, 80))

    # ---------------- event handling ----------------

    def handle_click(self, pos):
        for c in reversed(self.clickables):  # topmost-drawn first
            if c.collide(pos):
                kind, payload = c.payload
                if kind == "toggle_manual":
                    self.manual_boss_mode = not self.manual_boss_mode
                elif kind == "arm_skill":
                    unit_name, position = payload
                    self._arm(unit_name, position)
                elif kind == "clash_slot":
                    self._commit_action(clash_slot_index=payload)
                elif kind == "boss_portrait":
                    self._commit_action(clash_slot_index=None)
                elif kind == "attack_unopposed":
                    # payload is the unit_name; only clickable when that unit is armed
                    self._commit_action(clash_slot_index=None)
                elif kind == "cancel_action":
                    self._cancel_action(payload)
                elif kind == "start_turn":
                    self._start_turn()
                elif kind == "cycle_boss_skill":
                    self._cycle_boss_skill(payload)
                elif kind == "cycle_boss_target":
                    self._cycle_boss_target(payload)
                return  # only handle the first match

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
