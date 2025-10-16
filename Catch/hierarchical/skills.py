from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SkillConfig:
    """Configuration describing a single low-level skill policy.

    A skill is defined by the observation group it consumes, the action head it
    controls and the reward/termination terms that should be enabled while the
    skill is active.  The actual policy network can freely choose how to use
    the observation and reward definitions provided here.
    """

    name: str
    observation_group: str
    action_name: str
    reward_terms: List[str] = field(default_factory=list)
    termination_terms: List[str] = field(default_factory=list)
    warmup_steps: int = 0
    max_steps: Optional[int] = None

    def clone(self, name: Optional[str] = None) -> "SkillConfig":
        """Return a shallow copy optionally overriding the name."""

        return SkillConfig(
            name=name or self.name,
            observation_group=self.observation_group,
            action_name=self.action_name,
            reward_terms=list(self.reward_terms),
            termination_terms=list(self.termination_terms),
            warmup_steps=self.warmup_steps,
            max_steps=self.max_steps,
        )


class SkillLibrary:
    """Factory helpers that expose commonly used skill definitions."""

    @staticmethod
    def default_catch_skills() -> Dict[str, SkillConfig]:
        """Return a dictionary with the default Catch task skills.

        The returned dictionary is intended to provide a starting point for
        hierarchical policies.  Each skill focuses on a subset of the full
        objective which allows the associated policy head to specialise.
        """

        return {
            "approach": SkillConfig(
                name="approach",
                observation_group="high_level",
                action_name="arm_action",
                reward_terms=["approach_ee_object", "orientation_correct"],
                termination_terms=[],
                warmup_steps=0,
                max_steps=120,
            ),
            "adjust": SkillConfig(
                name="adjust",
                observation_group="skill_alignment",
                action_name="arm_action",
                reward_terms=["approach_gripper_handle", "align_grasp_around_handle"],
                termination_terms=[],
                warmup_steps=5,
                max_steps=60,
            ),
            "grasp": SkillConfig(
                name="grasp",
                observation_group="grasping",
                action_name="gripper_action",
                reward_terms=["grasp_object"],
                termination_terms=["object_dropping"],
                warmup_steps=0,
                max_steps=40,
            ),
            "transport": SkillConfig(
                name="transport",
                observation_group="transport",
                action_name="arm_action",
                reward_terms=["object_goal_tracking_fine_grained"],
                termination_terms=["object_dropping"],
                warmup_steps=0,
                max_steps=None,
            ),
        }

    @staticmethod
    def merge_skills(base: Dict[str, SkillConfig], overrides: Optional[Dict[str, SkillConfig]] = None) -> Dict[str, SkillConfig]:
        """Merge ``overrides`` into ``base`` returning a new dictionary."""

        merged = {name: cfg.clone() for name, cfg in base.items()}
        if overrides:
            for name, cfg in overrides.items():
                merged[name] = cfg.clone(name)
        return merged
