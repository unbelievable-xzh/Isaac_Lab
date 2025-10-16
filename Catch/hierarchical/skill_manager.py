from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from .skills import SkillConfig


@dataclass
class HierarchicalPolicyState:
    """Runtime state of the hierarchical controller."""

    active_skill: str
    steps_remaining: Optional[int]
    warmup_counter: int = 0
    skill_history: List[str] = field(default_factory=list)

    def reset(self, skill: str, steps_remaining: Optional[int]) -> None:
        self.active_skill = skill
        self.steps_remaining = steps_remaining
        self.warmup_counter = 0
        self.skill_history.clear()


class HierarchicalPolicyManager:
    """Helper class coordinating the high-level and low-level policies.

    The manager keeps track of the currently active skill and exposes small
    helper utilities that the training script can use to decide when the
    high-level policy needs to produce a new decision.
    """

    def __init__(self, skills: Dict[str, SkillConfig], default_skill: str, decision_interval: int) -> None:
        if default_skill not in skills:
            raise KeyError(f"Default skill '{default_skill}' is not defined")
        if decision_interval <= 0:
            raise ValueError("decision_interval must be strictly positive")

        self._skills = skills
        self._decision_interval = decision_interval
        self._state = HierarchicalPolicyState(
            active_skill=default_skill,
            steps_remaining=skills[default_skill].max_steps,
        )

    @property
    def active_skill(self) -> SkillConfig:
        return self._skills[self._state.active_skill]

    @property
    def state(self) -> HierarchicalPolicyState:
        return self._state

    @property
    def skills(self) -> Dict[str, SkillConfig]:
        return self._skills

    @property
    def decision_interval(self) -> int:
        return self._decision_interval

    def reset(self, *, default_skill: Optional[str] = None) -> None:
        """Reset the manager to the initial state."""

        skill_name = default_skill or self._state.active_skill
        skill_cfg = self._skills[skill_name]
        self._state.reset(skill_name, skill_cfg.max_steps)

    def step(self, high_level_action: Optional[int] = None) -> SkillConfig:
        """Update the active skill and return its configuration.

        Parameters
        ----------
        high_level_action:
            Discrete index returned by the high-level policy. If ``None`` the
            current skill remains active until its timer runs out.
        """

        if high_level_action is not None:
            self._activate_skill_by_index(high_level_action)
            return self.active_skill

        if self._state.steps_remaining is not None:
            self._state.steps_remaining -= 1
            if self._state.steps_remaining <= 0:
                # Force the high-level controller to produce a new action by
                # setting ``steps_remaining`` to zero which callers can check.
                self._state.steps_remaining = 0
        return self.active_skill

    def needs_high_level_action(self) -> bool:
        """Return True if a new high-level decision should be requested."""

        skill = self.active_skill
        if self._state.steps_remaining == 0:
            return True
        return (self._state.warmup_counter % self._decision_interval) == 0

    def apply_skill_transition(self) -> None:
        """Advance counters after the low-level policy step finished."""

        self._state.warmup_counter += 1
        skill = self.active_skill
        if skill.max_steps is not None and self._state.steps_remaining is not None and self._state.steps_remaining > 0:
            self._state.steps_remaining = max(0, self._state.steps_remaining - 1)

    def _activate_skill_by_index(self, action_index: int) -> None:
        skills = list(self._skills.values())
        if action_index < 0 or action_index >= len(skills):
            raise IndexError(f"High level action index {action_index} is out of bounds")
        selected = skills[action_index]
        self._state.skill_history.append(selected.name)
        self._state.active_skill = selected.name
        self._state.warmup_counter = 0
        self._state.steps_remaining = selected.max_steps

    def available_skill_names(self) -> Iterable[str]:
        return self._skills.keys()
