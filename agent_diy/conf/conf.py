#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright 漏 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Configuration for the DIY PPO agent.
"""


class Config:
    # Phase 2 multi-scale map feature dimensions (234D total).
    FEATURES = [
        8,   # hero self
        16,  # monsters: 2 * 8
        10,  # treasures: 2 * 5
        10,  # buffs: 2 * 5
        170, # local map: coarse 7 * 7 + center 11 * 11 from local vision
        16,  # legal movement + flash actions
        4,   # progress and environment rhythm
    ]
    FEATURE_SPLIT_SHAPE = FEATURES
    FEATURE_LEN = sum(FEATURE_SPLIT_SHAPE)
    DIM_OF_OBSERVATION = FEATURE_LEN

    # Compatibility aliases for the original DIY template.
    USE_CNN = False
    VIEW_SIZE = 0
    FEATURE_VECTOR_SHAPE = (DIM_OF_OBSERVATION,)
    FEATURE_IMAGE_SHAPE = (4, VIEW_SIZE + 1, VIEW_SIZE + 1)

    # Phase 2 expands the action space to 16 actions.
    ACTION_NUM = 16
    ACTION_SHAPE = (ACTION_NUM,)

    VALUE_NUM = 1
    VALUE_SHAPE = (VALUE_NUM,)

    # PPO hyperparameters.
    GAMMA = 0.99
    LAMDA = 0.95
    INIT_LEARNING_RATE_START = 0.0003
    START_LR = INIT_LEARNING_RATE_START
    BETA_START = 0.001
    CLIP_PARAM = 0.2
    VF_COEF = 1.0
    VALUE_LOSS_COEFF = VF_COEF
    ENTROPY_LOSS_COEFF = BETA_START
    GRAD_CLIP_RANGE = 0.5
