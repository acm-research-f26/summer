import random


class SkillQueue:
    """
    Manages a player unit's 8-slot skill queue.

    Index 0 = bottom-most (currently available, "bottom" of the pair)
    Index 1 = next available ("top" of the pair)
    Index 2,3 = next pair (visible, not pickable)
    Index 4-7 = hidden pairs

    Queue is stored as a flat list of strings: 'skill1' or 'skill2'.
    It's built out of pairs; each pair is a shuffled ('skill1','skill2').
    """

    def __init__(self, rng=None):
        self.rng = rng or random.Random()
        self.queue = []
        # start with 4 pairs = 8 slots
        for _ in range(4):
            self._add_pair()

    def _add_pair(self):
        pair = ['skill1', 'skill2']
        self.rng.shuffle(pair)
        self.queue.extend(pair)

    def _refill_if_needed(self):
        while len(self.queue) < 6:
            self._add_pair()

    @property
    def available(self):
        """The two skills the player can currently pick from (bottom pair remnants)."""
        return list(self.queue[0:2])

    @property
    def next_up(self):
        """Visible but not pickable yet."""
        return list(self.queue[2:4])

    def pick(self, skill_type, position=None):
        """
        Player picks a skill_type ('skill1' or 'skill2') from the available slots.
        Must be present in self.available.
        Applies the shift rule based on whether it was bottom (index0) or top (index1).

        `position` (0 or 1) disambiguates WHICH of the two available slots was picked,
        which matters when both available slots happen to be the same skill_type
        (e.g. available == ['skill1', 'skill1']) — in that case skill_type alone can't
        tell us whether the bottom-shift or top-shift rule should apply. If the two
        available slots differ, position is optional and can be left as None.

        Returns the skill_type picked (for convenience/chaining).
        """
        if skill_type not in self.available:
            raise ValueError(
                f"Cannot pick {skill_type}; available slots are {self.available}"
            )

        if position is not None:
            if position not in (0, 1):
                raise ValueError(f"position must be 0 or 1, got {position}")
            if self.queue[position] != skill_type:
                raise ValueError(
                    f"Slot {position} is {self.queue[position]!r}, not {skill_type!r}"
                )
            picked_bottom = (position == 0)
        else:
            # fall back to matching queue[0] first; ambiguous only when both slots
            # are the same type, in which case it doesn't matter which specific
            # instance we treat this as (they're identical), so default to bottom.
            picked_bottom = (self.queue[0] == skill_type)

        if picked_bottom:
            # picked the bottom slot -> remove just index 0, shift down by 1
            self.queue.pop(0)
        else:
            # picked the top slot (index1) -> remove index0 AND index1, shift down by 2
            self.queue.pop(0)
            self.queue.pop(0)  # after first pop, old index1 is now index0

        self._refill_if_needed()
        return skill_type

    def __repr__(self):
        return f"SkillQueue(available={self.available}, next_up={self.next_up}, len={len(self.queue)})"
