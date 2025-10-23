# Hierarchical RL Migration Guide for Catch Task

This document outlines how to refactor the current single-policy PPO setup into a hierarchical reinforcement learning (HRL) approach with multiple sub-policies collaborating on the Catch manipulation task.

## 1. Motivation and High-Level Structure

The existing environment (`CatchEnvCfg`) couples all stages of the task—approach, grasp, and object stabilization—into a single MDP with dense shaping rewards. Moving to HRL allows you to:

- Separate high-level decision making (selecting sub-goals / skill primitives) from low-level motor control.
- Reuse lower-level controllers across tasks or scene variations.
- Improve exploration by constraining each sub-policy to a simpler objective.

A practical architecture is a **two-level hierarchy**:

1. **Manager (high level)** operating at a lower frequency that selects sub-goals or activates a discrete skill primitive ("approach", "align", "grasp", "stabilize").
2. **Skill policies (low level)** that execute continuous control actions conditioned on the current sub-goal for a fixed horizon.

## 2. Environment Refactoring

1. **Expose skill-specific observations and rewards**
   - Split the current observation terms in `ObservationsCfg.PolicyCfg` into subsets that are meaningful for each skill. For example, the approach skill needs end-effector to object pose (`gripper_object_rel_pose`) whereas the stabilize skill focuses on object pose and velocity.
   - Create new observation groups (e.g., `ApproachObsCfg`, `GraspObsCfg`) and register them in the manager config so that each sub-policy receives only relevant signals.【F:Catch/catch_env_cfg.py†L63-L117】

2. **Modularize reward terms**
   - Partition the reward definitions in `RewardsCfg` into reusable functions for each skill. For instance, keep `approach_ee_object` and `orientation_correct` for the approach skill, and `grasp_object`, `align_grasp_around_handle` for the grasp skill.【F:Catch/catch_env_cfg.py†L118-L159】
   - Implement skill-specific reward weights and termination conditions to encourage completion of sub-goals before switching.

3. **Add sub-goal interfaces**
   - Extend `CommandsCfg` so the manager can issue symbolic goals (e.g., desired end-effector pose or gripper state).【F:Catch/catch_env_cfg.py†L87-L103】
   - Provide API hooks in the task (e.g., `set_sub_goal(skill_id, parameters)`) to update the environment state that each low-level controller reads.

4. **Temporal abstraction**
   - Introduce a macro-action horizon `k` so that the manager acts every `k` environment steps (`decimation` × `k` simulation steps). This can be done by adding a counter in the environment loop that only queries the high-level policy at the desired frequency.

## 3. Agent Configuration Changes

1. **Define multiple policies**
   - Replace `CatchCubePPORunnerCfg` with a wrapper runner that instantiates one PPO agent per skill plus a manager policy (which can also be PPO or a discrete policy gradient).【F:Catch/config/CatchRobot/agents/rsl_rl_ppo_cfg.py†L1-L33】
   - Factor the rollout storage so that each skill collects its own trajectories (e.g., separate replay buffers or `rollout_storage` objects).

2. **Observation/action routing**
   - Modify the action manager in `mdp` to route the low-level policy output to the existing `arm_action`/`gripper_action` when its skill is active. When a different skill is active, freeze or blend the previous policy outputs.
   - For the manager, define a categorical action space whose dimension equals the number of skills (and optionally continuous goal parameters).

3. **Training loop adjustments**
   - Implement an outer loop that alternates between collecting experience for the manager and the active skill policies. Each environment step should log `(state, skill_id, reward, done)` for the manager and `(state_skill, action, reward_skill, done_skill)` for the low-level agent.
   - Update the PPO configuration (discount factors, GAE lambda) separately for manager vs. skill policies to reflect their different temporal scales.

## 4. Suggested Implementation Steps

1. **Skill API in `mdp`**
   - Add enumerations for skill IDs and helper functions (e.g., `is_grasp_complete`, `is_approach_complete`) to determine switching conditions.
   - Wrap the existing reward functions so they can be toggled per-skill.

2. **High-level policy skeleton**
   - Create a new module `Catch/config/CatchRobot/agents/hrl_manager_cfg.py` with PPO (or option-critic) hyperparameters for the manager policy. Share the same value network as skills if needed, but keep separate optimizers.

3. **Skill policy configs**
   - Duplicate and simplify the existing PPO config for each skill, adjusting the observation dimension and reward scale accordingly.

4. **Logging & curriculum**
   - Extend `CurriculumCfg` with skill-specific schedules (e.g., progressively decrease the `approach` reward weight or increase success threshold for `grasp`).【F:Catch/catch_env_cfg.py†L160-L170】
   - Track success metrics per skill to monitor which parts of the hierarchy require retraining.

## 5. Optional Enhancements

- Use **option termination functions** instead of fixed horizons so skills can relinquish control early when their goal is met.
- Pretrain skills with demonstration data or imitation learning to bootstrap the HRL system.
- Incorporate **state machines** or **behavior trees** as an intermediate scaffold before fully learning the manager policy.

## 6. References

- Sutton, R. S., Precup, D., & Singh, S. (1999). Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning.
- Nachum, O., Gu, S., Lee, H., & Levine, S. (2018). Data-efficient hierarchical reinforcement learning.

