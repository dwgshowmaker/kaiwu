#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Configuration for the DIY PPO baseline.
DIY PPO 基线配置。
"""


class Config:

    # Feature dimensions / 特征维度（共40维）
    FEATURES = [
        4,
        5,
        5,
        16,
        8,
        2,
    ]
    FEATURE_SPLIT_SHAPE = FEATURES
    FEATURE_LEN = sum(FEATURE_SPLIT_SHAPE)
    DIM_OF_OBSERVATION = FEATURE_LEN

    # Compatibility aliases for the original DIY template.
    # 保留模板字段，方便后续扩展或框架侧读取。
    USE_CNN = False
    VIEW_SIZE = 0
    FEATURE_VECTOR_SHAPE = (DIM_OF_OBSERVATION,)
    FEATURE_IMAGE_SHAPE = (4, VIEW_SIZE + 1, VIEW_SIZE + 1)

    # Action space / 动作空间：Phase 0 先保持8个移动方向
    ACTION_NUM = 8
    ACTION_SHAPE = (ACTION_NUM,)

    # Value head / 价值头：单头生存奖励
    VALUE_NUM = 1
    VALUE_SHAPE = (VALUE_NUM,)

    # PPO hyperparameters / PPO 超参数
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
