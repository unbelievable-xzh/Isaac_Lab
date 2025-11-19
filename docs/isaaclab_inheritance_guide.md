# 继承 Isaac Lab 官方库的常见流程

下列步骤总结了在自己的工程中复用 Isaac Lab 官方库函数、任务定义与训练脚本的一般做法。请根据实际安装路径与项目结构调整具体命令。

## 1. 准备官方库依赖

1. 克隆官方仓库或安装发行包：
   ```bash
   git clone https://github.com/NVIDIA-Omniverse/IsaacLab.git ~/externals/isaaclab
   # 或者在已经下载的源码目录下执行
   pip install -e ~/externals/isaaclab[rlgames]
   ```
2. 将 Isaac Lab 的 Python 包加入环境变量，以便可以直接 `import omni.isaac.lab` 等模块：
   ```bash
   export ISAACLAB_PATH=~/externals/isaaclab
   export PYTHONPATH=$ISAACLAB_PATH/source/extensions:$ISAACLAB_PATH/source/python:$PYTHONPATH
   ```

上述命令确保你可以继承官方提供的任务、资源管理器和强化学习工具链。

## 2. 继承任务或环境定义

1. 在自己的仓库中新建一个 Python 模块（例如 `my_lab/tasks/my_robot_task.py`），并继承官方的任务基类：
   ```python
   from omni.isaac.lab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv
   from omni.isaac.lab_tasks.manager_based.base import ManagerBasedRLEnvCfg as BaseEnvCfg
   from omni.isaac.lab_tasks.manager_based.locomotion.anymal import AnymalEnvCfg

   class MyRobotEnvCfg(AnymalEnvCfg):
       def __post_init__(self):
           super().__post_init__()
           self.scene.num_envs = 4096

   class MyRobotEnv(ManagerBasedRLEnv):
       cfg: MyRobotEnvCfg
   ```
2. 通过继承配置类（`EnvCfg`、`CommandCfg`、`RewardsCfg` 等）或任务类（如 `ManagerBasedRLEnv`）定制观测、奖励、命令分布。继承时可以在 `__post_init__` 中修改默认组件，或覆盖类属性来引入自己的机器人描述文件、传感器等。
3. 如果只需要复用部分函数，可直接从 `omni.isaac.lab.utils`, `omni.isaac.lab.sim`, `omni.isaac.lab.assets` 模块中导入并组合。

## 3. 继承训练配置

1. Isaac Lab 的训练使用 Hydra + RL-Games 配置系统。复制官方仓库中 `source/extensions/omni.isaac.lab_tasks/config/rlgames/<task>.yaml` 到你的项目，例如 `configs/rlgames/my_robot.yaml`。
2. 在复制的 YAML 中覆盖需要修改的字段（如 `params.config.name`, `params.network` 或自定义的 `task.env.numEnvs` 等）。
3. 使用官方的训练入口脚本（如 `omni.isaac.lab.app` 或 `scripts/train.py`）：
   ```bash
   ./isaaclab.sh -p source/extensions/omni.isaac.lab.app/omni/isaac/lab/app.py \
       --config-name=my_robot \
       task.env=MyRobotEnvCfg \
       task.env.scene.num_envs=4096
   ```
   通过命令行参数或 Hydra overrides 注入你继承的配置与任务类。

## 4. 项目结构建议

- 将自定义任务与训练脚本放在独立 Python 包中，并在 `setup.cfg` 或 `pyproject.toml` 中声明 `entry_points`，使 Isaac Lab 能够发现自定义任务。
- 在仓库中保留一个 `requirements.txt` 或 `environment.yml`，写明 Isaac Lab 版本与额外依赖。
- 若需要继承官方的 USD 资产或运动学描述，建议将其复制到 `assets/` 目录并在配置里更新路径。

通过以上步骤，你可以无缝继承 Isaac Lab 官方库提供的类与函数，同时保持自定义项目与官方更新解耦。
