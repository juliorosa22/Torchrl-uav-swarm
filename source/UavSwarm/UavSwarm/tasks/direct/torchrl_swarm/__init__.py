# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="FullTask-TorchRL-UAVSwarm-Direct-v0",
    entry_point=f"{__name__}.torchrl_swarm_env:FullTaskUAVSwarmEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.torchrl_swarm_env_cfg:FullTaskUAVSwarmEnvCfg",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)

gym.register(
    id="Baseline-TorchRL-UAVSwarm-Direct-v0",
    entry_point=f"{__name__}.torchrl_swarm_env:BaselineUAVSwarmEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.torchrl_swarm_env_cfg:BaselineUAVSwarmEnvCfg",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)

gym.register(
    id="Formation-TorchRL-UAVSwarm-Direct-v0",
    entry_point=f"{__name__}.torchrl_swarm_env:FormationUAVSwarmEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.torchrl_swarm_env_cfg:FormationUAVSwarmEnvCfg",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)

gym.register(
    id="SingleGoal-TorchRL-UAVSwarm-Direct-v0",
    entry_point=f"{__name__}.torchrl_swarm_env:SingleGoalUAVSwarmEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.torchrl_swarm_env_cfg:SingleGoalUAVSwarmEnvCfg",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)
