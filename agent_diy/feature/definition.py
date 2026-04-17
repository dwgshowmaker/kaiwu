#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Data definitions and GAE helpers for the DIY Gorge Chase PPO agent.
"""

import numpy as np
from common_python.utils.common_func import create_cls

from agent_diy.conf.conf import Config


ObsData = create_cls("ObsData", feature=None, legal_action=None, action_bias=None)

ActData = create_cls("ActData", action=None, d_action=None, prob=None, value=None)

SampleData = create_cls(
    "SampleData",
    obs=Config.DIM_OF_OBSERVATION,
    legal_action=Config.ACTION_NUM,
    action_bias=Config.ACTION_NUM,
    act=1,
    reward=Config.VALUE_NUM,
    reward_sum=Config.VALUE_NUM,
    done=1,
    value=Config.VALUE_NUM,
    next_value=Config.VALUE_NUM,
    advantage=Config.VALUE_NUM,
    prob=Config.ACTION_NUM,
)


def sample_process(list_sample_data):
    for i in range(len(list_sample_data) - 1):
        list_sample_data[i].next_value = list_sample_data[i + 1].value
    _calc_gae(list_sample_data)
    return list_sample_data


def _calc_gae(list_sample_data):
    gae = np.zeros((Config.VALUE_NUM,), dtype=np.float32)
    gamma = Config.GAMMA
    lamda = Config.LAMDA

    for sample in reversed(list_sample_data):
        done = _scalar(sample.done)
        not_done = 1.0 - done
        value = _to_np(sample.value)
        next_value = _to_np(sample.next_value)
        reward = _to_np(sample.reward)

        delta = reward + gamma * next_value * not_done - value
        gae = delta + gamma * lamda * not_done * gae
        sample.advantage = gae.astype(np.float32)
        sample.reward_sum = (gae + value).astype(np.float32)


def _to_np(value):
    if isinstance(value, np.ndarray):
        return value.astype(np.float32)
    return np.array(value, dtype=np.float32).reshape(-1)[: Config.VALUE_NUM]


def _scalar(value):
    if isinstance(value, np.ndarray):
        return float(value.reshape(-1)[0])
    if isinstance(value, (list, tuple)):
        return float(value[0])
    return float(value)
