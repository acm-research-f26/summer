"""Deterministic FIGO derivation from geometry (Phase 4).

FIGO type is *defined* by geometry, so this is a lookup, not a prediction. The
whole reason the project is honest is that nothing here is learned: the type
falls out of three measured fractions and two contact booleans.

On type 8
---------
Type 8 is "other" -- cervical, parasitic, broad-ligament. It is **not derivable
from a UMD mask**. The label scheme is {background, wall, cavity, myoma,
nabothian cyst}; there is no cervix label and no landmark that separates the
cervix from the corpus, so a cervical fibroid is geometrically indistinguishable
from an intramural one low in the uterus. The plan's original gate ("all 9 types
are represented") is therefore not satisfiable honestly, and this module derives
**eight** types plus hybrids. Type 8 is reported as not-assessable rather than
guessed, and ``figo.include_type8`` controls whether a bucket is emitted at all.
"""

from __future__ import annotations

import math

TYPE_MEANINGS = {
    "0": "pedunculated intracavitary",
    "1": "submucosal, <50% intramural",
    "2": "submucosal, >=50% intramural",
    "3": "100% intramural, contacts endometrium",
    "4": "intramural, contacts neither surface",
    "5": "subserosal, >=50% intramural",
    "6": "subserosal, <50% intramural",
    "7": "subserosal pedunculated",
    "8": "other (cervical/parasitic) - NOT derivable from mask geometry",
}

SUBMUCOSAL = {"0", "1", "2"}
SUBSEROSAL = {"5", "6", "7"}


def _submucosal_type(p, frac_cav, cfg):
    """Types 0/1/2/3, separated by how much of the fibroid is *in* the cavity.

    Type 3 is the subtle one: it abuts the endometrium but does not bulge into
    the cavity. Keying it off ``p >= 95`` alone is wrong, because for a fibroid
    with no extraserosal part ``p >= 95`` and ``frac_cav <= 5`` are the same
    statement -- so a genuine type 2 with a 4% cavity bulge was being called
    type 3. The bulge is what the definition actually turns on, so it is tested
    directly and with a much tighter threshold.
    """
    ped = float(cfg["figo.pedunculated_max_intramural_pct"])
    thr = float(cfg["figo.intramural_threshold_pct"])
    if p <= ped and frac_cav >= 100.0 - ped:
        return "0"
    if frac_cav <= float(cfg["figo.type3_max_intracavitary_pct"]):
        return "3"
    return "1" if p < thr else "2"


def _subserosal_type(p, frac_ser, cfg):
    ped = float(cfg["figo.pedunculated_max_intramural_pct"])
    thr = float(cfg["figo.intramural_threshold_pct"])
    if p <= ped and frac_ser >= 100.0 - ped:
        return "7"
    return "6" if p < thr else "5"


def derive(record, cfg) -> str:
    """Map one fibroid's measurements to a FIGO type string.

    Hybrids (a fibroid reaching both the endometrium and the serosa) are written
    as two numbers, submucosal first, e.g. ``"2-5"``.
    """
    p = float(record["percent_intramural"])
    if math.isnan(p):
        return "NA"
    frac_cav = float(record["frac_intracavitary"])
    frac_ser = float(record["frac_extraserosal"])
    tc, ts = bool(record["touch_cavity"]), bool(record["touch_serosa"])

    if tc and ts:
        sep = cfg["figo.hybrid_separator"]
        return f"{_submucosal_type(p, frac_cav, cfg)}{sep}{_subserosal_type(p, frac_ser, cfg)}"
    if tc:
        return _submucosal_type(p, frac_cav, cfg)
    if ts:
        return _subserosal_type(p, frac_ser, cfg)
    return "4"


def components(figo_type: str, cfg) -> list[str]:
    """Split a hybrid like ``"2-5"`` into ``["2", "5"]``; singles pass through."""
    return figo_type.split(cfg["figo.hybrid_separator"]) if figo_type != "NA" else []


def is_submucosal(figo_type: str, cfg) -> bool:
    """True if any component is submucosal (0/1/2) -- the hysteroscopic candidates."""
    return bool(SUBMUCOSAL & set(components(figo_type, cfg)))


def is_subserosal(figo_type: str, cfg) -> bool:
    return bool(SUBSEROSAL & set(components(figo_type, cfg)))


def near_threshold(record, cfg) -> bool:
    """Within the configured band of the 50% line, where 1-vs-2 / 5-vs-6 is a coin flip."""
    p = float(record["percent_intramural"])
    if math.isnan(p):
        return False
    return abs(p - float(cfg["figo.intramural_threshold_pct"])) <= float(cfg["borderline.band_pct"])
