#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Configuration for the DIY Gorge Chase PPO agent.
"""


class Config:
    HERO_SELF_FEATURES = 10
    MONSTER_FEATURES = 16
    TREASURE_FEATURES = 10
    BUFF_FEATURES = 10
    LOCAL_ROUTE_FEATURES = 24
    PROGRESS_FEATURES = 4

    FEATURES = [
        HERO_SELF_FEATURES,
        MONSTER_FEATURES,
        TREASURE_FEATURES,
        BUFF_FEATURES,
        LOCAL_ROUTE_FEATURES,
        PROGRESS_FEATURES,
    ]
    FEATURE_SPLIT_SHAPE = FEATURES
    FEATURE_LEN = sum(FEATURE_SPLIT_SHAPE)
    DIM_OF_OBSERVATION = FEATURE_LEN
    FEATURE_VECTOR_SHAPE = (FEATURE_LEN,)

    ACTION_NUM = 16
    ACTION_SHAPE = (ACTION_NUM,)
    VALUE_NUM = 1
    VALUE_SHAPE = (VALUE_NUM,)

    MAP_SIZE = 128.0
    VIEW_SIZE = 21
    VIEW_RADIUS = 10
    LOCAL_ROUTE_MAX_LEN = VIEW_RADIUS
    MOVE_DIRS = (
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    )

    MAX_EPISODE_STEP = 1000.0
    MONSTER_DIST_MAX = (MAP_SIZE**2 + MAP_SIZE**2) ** 0.5
    MONSTER_SPEED_MAX = 2.0
    FLASH_COOLDOWN_MAX = 2000.0
    BUFF_DURATION = 50.0
    RECENT_FLASH_WINDOW = 20
    FLASH_RANGE_STRAIGHT = 10
    FLASH_RANGE_DIAGONAL = 8

    SURVIVE_REWARD = 0.01
    DANGER_DELTA_REWARD_SCALE = 0.05
    TREASURE_REWARD = 1.25
    BUFF_REWARD = 0.55
    GOOD_FLASH_REWARD = 0.45
    BAD_FLASH_PENALTY = -0.18
    STUCK_PENALTY = 0.03
    STUCK_THRESHOLD = 3
    THREAT_DISTANCE = 6.0
    THREAT_PENALTY_NEAR = -0.12
    THREAT_PENALTY_DANGER = -0.8
    FLASH_ESCAPE_THRESHOLD = 1.5
    FAIL_PENALTY = -10.0
    COMPLETE_BONUS = 9.0

    DEFAULT_MONSTER_INTERVAL = 300
    DEFAULT_MONSTER_SPEEDUP = 500

    GAMMA = 0.99
    LAMDA = 0.95
    INIT_LEARNING_RATE_START = 3.0e-4
    BETA_START = 0.001
    CLIP_PARAM = 0.2
    VF_COEF = 1.0
    GRAD_CLIP_RANGE = 0.5

    MODEL_HIDDEN_DIMS = (256, 128, 64)

    SAVE_MODEL_INTERVAL_SEC = 1800
    MONITOR_REPORT_INTERVAL_SEC = 60
    TRAINING_METRICS_INTERVAL_SEC = 60

    TRAIN_HEURISTIC_WEIGHT = 0.98
    EXPLOIT_HEURISTIC_WEIGHT = 1.0
    DANGER_HEURISTIC_WEIGHT = 1.0
    TRAIN_GREEDY_PROB = 0.97
    HEURISTIC_TEMPERATURE = 0.45
    HEURISTIC_DANGER_DIST = 8.0
    HEURISTIC_CRITICAL_DIST = 4.0
    HEURISTIC_TREASURE_SAFE_DIST = 18.0
    HEURISTIC_FLASH_SAFE_DIST = 7.0
    HEURISTIC_ESCAPE_GAIN_WEIGHT = 3.4
    HEURISTIC_DISTANCE_WEIGHT = 0.04
    HEURISTIC_CORRIDOR_WEIGHT = 1.6
    HEURISTIC_MOVE_STABILITY_WEIGHT = 0.25
    HEURISTIC_TREASURE_PROGRESS_WEIGHT = 0.14
    HEURISTIC_BUFF_PROGRESS_WEIGHT = 0.06
    HEURISTIC_TARGET_REACH_BONUS = 1.8
    HEURISTIC_FLASH_BASE_PENALTY = 1.4
    HEURISTIC_FLASH_ESCAPE_WEIGHT = 0.55
    HEURISTIC_DEAD_END_PENALTY = 2.2
    HEURISTIC_REVERSE_PENALTY = 0.5
    HEURISTIC_STUCK_ACTION_PENALTY = 0.7

    CURRICULUM_STAGES = (
        {
            "name": "stage_a",
            "max_episode": 200,
            "env_conf": {
                "map": [1, 2, 3, 4],
                "map_random": True,
                "monster_interval": 500,
                "monster_speedup": 800,
                "max_step": 600,
            },
        },
        {
            "name": "stage_b",
            "max_episode": 600,
            "env_conf": {
                "map": [1, 2, 3, 4, 5, 6, 7],
                "map_random": True,
                "monster_interval": 300,
                "monster_speedup": 500,
                "max_step": 800,
            },
        },
        {
            "name": "stage_c",
            "max_episode": None,
            "env_conf": {
                "map": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "map_random": True,
                "monster_interval": -1,
                "monster_speedup": -1,
                "max_step": 1000,
            },
        },
    )
