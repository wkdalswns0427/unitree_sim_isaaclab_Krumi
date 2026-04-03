# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

import gymnasium as gym

from . import agents
from . import fist_control_env_cfg

gym.register(
    id="Isaac-Fist-Control-H12",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": fist_control_env_cfg.FistControlH12EnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FistControlH12PPORunnerCfg",
    },
    disable_env_checker=True,
)