"""
Single source of truth for tunable game balance data (HP, stagger thresholds,
base damage, roll ranges, and the boss's clash-bonus knobs) and the skill /
passive description text shown in tooltips.

Both game_ui.py (the real shipped game) and balance_sim.py (the tuning /
experimentation harness) build their units and boss from the same PARAMS
dict here via build_party_units() / build_boss(), instead of each keeping
their own hardcoded copy of the numbers. That used to be a real hazard: the
two lived independently and could quietly drift out of sync with each other.
Now there's exactly one place these numbers live.

Status effect MAGNITUDES (how much tremor/burn/etc. each skill applies, and
all the burst/conversion/stacking mechanics) are considered fixed design and
are NOT part of PARAMS - they're hardcoded directly in the effect_* functions
in engine.py, since the balance-tuning process was only ever meant to touch
HP, stagger thresholds, damage, and roll ranges (see balance_sim.py's
docstring for the reasoning behind that boundary).
"""

from engine import (
    PlayerUnit, SkillDef, Boss, BossSkillDef,
    effect_tremor_scorch_skill1, effect_tremor_scorch_skill2,
    effect_dark_flame_skill1, effect_dark_flame_skill2,
    effect_self_status_skill1, effect_self_status_skill2,
    effect_boss_tremor_slam, effect_boss_burn_wave, effect_boss_clash_baiter,
    effect_boss_scorch_point, effect_boss_amplitude_cascade,
    passive_roll_bonus_tremor_scorch, passive_roll_bonus_dark_flame,
    passive_roll_bonus_self_status, passive_damage_reduction_self_status,
)

# ----------------------------------------------------------------------------
# Display names for the shipped game. balance_sim.py uses its own plainer
# archetype names instead (passed to build_party_units/build_boss via `names`),
# since it's running throwaway experiments, not the real game.
# ----------------------------------------------------------------------------
TREMOR_SCORCH_NAME = "Thumb East Capo III"
DARK_FLAME_NAME = "Lobotomy EGO: Magic Bullet"
SELF_STATUS_NAME = "You Branch Adept"
BOSS_NAME = "Thumb East Capo II"

# ----------------------------------------------------------------------------
# THE tunable parameters. Arrived at via extensive simulation in balance_sim.py:
# skilled play (good clash decisions + passive awareness) wins ~78-85% of the
# time in ~8-9 turns; naive play (never clashes, never uses the stronger
# skill2) wins essentially never (~0-3%). Roll ranges are chosen so no clash
# matchup is ever a guaranteed win or loss for either side.
# ----------------------------------------------------------------------------
PARAMS = {
    "tremor_scorch": {
        "hp": 532, "stagger": [213],
        "skill1_dmg": 25, "skill1_roll": (13, 19),
        "skill2_dmg": 50, "skill2_roll": (9, 18),
    },
    "dark_flame": {
        "hp": 614, "stagger": [521, 338, 153],
        "skill1_dmg": 25, "skill1_roll": (13, 19),
        "skill2_dmg": 50, "skill2_roll": (9, 18),
    },
    "self_status": {
        "hp": 409, "stagger": [286, 143],
        "skill1_dmg": 25, "skill1_roll": (13, 19),
        "skill2_dmg": 50, "skill2_roll": (9, 18),
    },
    "boss": {
        "hp": 1450, "stagger": [1000, 500],
        "skills": [
            # (name, dmg, roll, hits_all, clash_roll_bonus, clash_damage_multiplier)
            ("Tremor Slam", 47, (9, 18), False, 0, 1.0),
            ("Burn Wave", 18, (3, 17), True, 0, 1.0),
            ("Clash Baiter", 27, (11, 15), False, 5, 10.0),
            ("Scorch Point", 32, (13, 18), False, 0, 1.0),
            ("Amplitude Cascade", 47, (14, 18), True, 0, 1.0),
        ],
    },
}


# ----------------------------------------------------------------------------
# Description templates. {lo}/{hi}/{dmg} are filled in from PARAMS at build
# time so the tooltip text can never drift out of sync with the actual
# numbers - only the fixed status-effect magnitudes are spelled out literally.
# ----------------------------------------------------------------------------
_TREMOR_SCORCH_SKILL1_DESC = (
    "Inflict 3 tremor potency and 2 tremor count. If tremor count "
    "is now more than 3, tremor burst once. Rolls {lo}-{hi}. Base damage {dmg}."
)
_TREMOR_SCORCH_SKILL2_DESC = (
    "Inflict 2 tremor potency, perform tremor burst twice, and trigger "
    "amplitude conversion of the tremor type to tremor scorch. "
    "Rolls {lo}-{hi}. Base damage {dmg}."
)
_TREMOR_SCORCH_PASSIVE = (
    "If opponent has +15 burn potency, roll +1.5 more (added to both "
    "lower and bigger bound). If opponent has +15 tremor potency, "
    "roll +1.5 more in the same way."
)

_DARK_FLAME_SKILL1_DESC = (
    "Inflict burn equal to current magic bullets, then inflict 1 dark "
    "flame on target. Gain 2 magic bullets (max 7). Rolls {lo}-{hi}. Base damage {dmg}."
)
_DARK_FLAME_SKILL2_DESC = (
    "Gain 1 magic bullet (max 7), then inflict dark flame equal to "
    "current magic bullets. Rolls {lo}-{hi}. Base damage {dmg}."
)
_DARK_FLAME_PASSIVE = (
    "If currently has 5+ magic bullets, roll +1.5 more (added to both "
    "lower and bigger bound). If opponent has +15 burn potency, "
    "roll +1.5 more (added to both lower and bigger bound)."
)

_SELF_STATUS_SKILL1_DESC = (
    "Inflict 16 tremor potency and 8 tremor count on opponent, while "
    "applying 3 tremor count and 5 tremor potency to self. "
    "Rolls {lo}-{hi}. Base damage {dmg}."
)
_SELF_STATUS_SKILL2_DESC = (
    "Inflict 10 burn potency and 5 burn count on self and target. Burn "
    "cannot cause HP to go below 1. Base damage is {dmg} + burn potency. Rolls {lo}-{hi}."
)
_SELF_STATUS_PASSIVE = (
    "If self has +10 tremor potency, roll +1.5 more and take 20% less "
    "damage. If self has +15 burn potency, roll +1.5 more and take "
    "20% less damage. Also, once per battle, if HP drops below zero "
    "(from any cause other than this unit's own self-burn), remove "
    "all burn and tremor on self and heal back to 80 HP."
)

_BOSS_TREMOR_SLAM_DESC = (
    "On hit, inflict 5 tremor potency and 3 tremor count, then trigger "
    "tremor burst. Damage {dmg}. Rolls {lo}-{hi}."
)
_BOSS_BURN_WAVE_DESC = (
    "Hits ALL party members, inflicting 10 burn potency and 3 burn count "
    "on each. Damage {dmg} to all members. Rolls {lo}-{hi}."
)
_BOSS_CLASH_BAITER_DESC = (
    "If this attack is clashed by another skill, gain +{bonus} to its roll "
    "and deal {mult:g}x damage if it wins the clash. Inflicts 3 tremor potency "
    "(and 3 tremor count, so it actually decays). "
    "Base damage {dmg}. Rolls {lo}-{hi} ({boosted_lo}-{boosted_hi} while being clashed)."
)
_BOSS_SCORCH_POINT_DESC = (
    "Deal 10 burn potency (and 3 burn count, so it actually decays) to "
    "target. Base damage {dmg}. Rolls {lo}-{hi}."
)
_BOSS_AMPLITUDE_CASCADE_DESC = (
    "Targets all enemies. Inflicts 1 tremor count and 1 tremor potency on "
    "all, then bursts, while also triggering amplitude conversion into "
    "tremor scorch. Base damage {dmg}. Rolls {lo}-{hi}."
)
_BOSS_PASSIVE = (
    "Max HP {hp}. Stagger thresholds at {t1} and {t2}. Picks 3 of its "
    "5 skills at random each turn, each with a random target, unless "
    "manually overridden."
)


def build_party_units(params=None, names=None):
    """
    Returns the 3 party PlayerUnit instances, built from `params` (defaults
    to the shipped PARAMS). Pass a `names` dict like
    {"tremor_scorch": "...", "dark_flame": "...", "self_status": "..."} to
    use different display names (balance_sim.py does this for its plainer
    archetype names instead of the flavor ones).
    """
    p = params if params is not None else PARAMS
    names = names or {
        "tremor_scorch": TREMOR_SCORCH_NAME,
        "dark_flame": DARK_FLAME_NAME,
        "self_status": SELF_STATUS_NAME,
    }

    ts = p["tremor_scorch"]
    tremor_scorch = PlayerUnit(
        name=names["tremor_scorch"], max_hp=ts["hp"],
        skill1=SkillDef(
            "Tremor Jab", *ts["skill1_roll"], ts["skill1_dmg"],
            description=_TREMOR_SCORCH_SKILL1_DESC.format(
                lo=ts["skill1_roll"][0], hi=ts["skill1_roll"][1], dmg=ts["skill1_dmg"]),
            effect=effect_tremor_scorch_skill1,
        ),
        skill2=SkillDef(
            "Tremor Burst Strike", *ts["skill2_roll"], ts["skill2_dmg"],
            description=_TREMOR_SCORCH_SKILL2_DESC.format(
                lo=ts["skill2_roll"][0], hi=ts["skill2_roll"][1], dmg=ts["skill2_dmg"]),
            effect=effect_tremor_scorch_skill2,
        ),
        passive_description=_TREMOR_SCORCH_PASSIVE,
        stagger_thresholds=list(ts["stagger"]),
        passive_roll_bonus_fn=passive_roll_bonus_tremor_scorch,
    )

    df = p["dark_flame"]
    dark_flame = PlayerUnit(
        name=names["dark_flame"], max_hp=df["hp"],
        skill1=SkillDef(
            "Flame Tag", *df["skill1_roll"], df["skill1_dmg"],
            description=_DARK_FLAME_SKILL1_DESC.format(
                lo=df["skill1_roll"][0], hi=df["skill1_roll"][1], dmg=df["skill1_dmg"]),
            effect=effect_dark_flame_skill1,
        ),
        skill2=SkillDef(
            "Dark Flame Surge", *df["skill2_roll"], df["skill2_dmg"],
            description=_DARK_FLAME_SKILL2_DESC.format(
                lo=df["skill2_roll"][0], hi=df["skill2_roll"][1], dmg=df["skill2_dmg"]),
            effect=effect_dark_flame_skill2,
        ),
        passive_description=_DARK_FLAME_PASSIVE,
        stagger_thresholds=list(df["stagger"]),
        max_magic_bullets=7,
        passive_roll_bonus_fn=passive_roll_bonus_dark_flame,
    )

    ss = p["self_status"]
    self_status = PlayerUnit(
        name=names["self_status"], max_hp=ss["hp"],
        skill1=SkillDef(
            "Shared Tremor", *ss["skill1_roll"], ss["skill1_dmg"],
            description=_SELF_STATUS_SKILL1_DESC.format(
                lo=ss["skill1_roll"][0], hi=ss["skill1_roll"][1], dmg=ss["skill1_dmg"]),
            effect=effect_self_status_skill1,
        ),
        skill2=SkillDef(
            "Shared Burn", *ss["skill2_roll"], ss["skill2_dmg"],
            description=_SELF_STATUS_SKILL2_DESC.format(
                lo=ss["skill2_roll"][0], hi=ss["skill2_roll"][1], dmg=ss["skill2_dmg"]),
            effect=effect_self_status_skill2,
        ),
        passive_description=_SELF_STATUS_PASSIVE,
        stagger_thresholds=list(ss["stagger"]),
        passive_roll_bonus_fn=passive_roll_bonus_self_status,
        passive_damage_reduction_fn=passive_damage_reduction_self_status,
    )

    return [tremor_scorch, dark_flame, self_status]


def build_boss(params=None, name=None):
    """Returns the Boss instance, built from `params` (defaults to PARAMS)."""
    p = params if params is not None else PARAMS
    name = name or BOSS_NAME
    b = p["boss"]

    skill_builders = [
        ("Tremor Slam", effect_boss_tremor_slam, _BOSS_TREMOR_SLAM_DESC),
        ("Burn Wave", effect_boss_burn_wave, _BOSS_BURN_WAVE_DESC),
        ("Clash Baiter", effect_boss_clash_baiter, _BOSS_CLASH_BAITER_DESC),
        ("Scorch Point", effect_boss_scorch_point, _BOSS_SCORCH_POINT_DESC),
        ("Amplitude Cascade", effect_boss_amplitude_cascade, _BOSS_AMPLITUDE_CASCADE_DESC),
    ]

    skills = []
    for (skill_name, dmg, roll, hits_all, bonus, mult), (_, effect_fn, desc_template) in \
            zip(b["skills"], skill_builders):
        lo, hi = roll
        description = desc_template.format(
            lo=lo, hi=hi, dmg=dmg, bonus=bonus, mult=mult,
            boosted_lo=lo + bonus, boosted_hi=hi + bonus,
        )
        skills.append(BossSkillDef(
            skill_name, lo, hi, dmg,
            hits_all=hits_all, effect=effect_fn, description=description,
            clash_roll_bonus=bonus, clash_damage_multiplier=mult,
        ))

    return Boss(
        name=name, max_hp=b["hp"], skills=skills,
        passive_description=_BOSS_PASSIVE.format(hp=b["hp"], t1=b["stagger"][0], t2=b["stagger"][1]),
        stagger_thresholds=list(b["stagger"]),
    )
