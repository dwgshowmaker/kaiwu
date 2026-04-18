#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Data definitions and sample post-processing for the DIY Gorge Chase agent.
"""

import numpy as np

from common_python.utils.common_func import create_cls

from agent_diy.conf.conf import Config


ObsData = create_cls("ObsData", feature=None, legal_action=None)
ActData = create_cls("ActData", action=None, d_action=None, prob=None, value=None)
SampleData = create_cls(
    "SampleData",
    obs=Config.DIM_OF_OBSERVATION,
    legal_action=Config.ACTION_NUM,
    act=1,
    reward=Config.VALUE_NUM,
    reward_sum=Config.VALUE_NUM,
    done=1,
    value=Config.VALUE_NUM,
    next_value=Config.VALUE_NUM,
    advantage=Config.VALUE_NUM,
    prob=Config.ACTION_NUM,
)


def reward_shaping(frame_no, score, terminated, truncated, remain_info, _remain_info, obs, _obs):
    del frame_no, score, terminated, truncated, remain_info, obs, _obs
    reward = _remain_info.get("reward", [0.0])
    return np.asarray(reward, dtype=np.float32)


def sample_process(list_sample_data):
    if not list_sample_data:
        return list_sample_data

    for index in range(len(list_sample_data) - 1):
        list_sample_data[index].next_value = list_sample_data[index + 1].value

    _calc_gae(list_sample_data)
    return list_sample_data


def _calc_gae(list_sample_data):
    gae = np.zeros(Config.VALUE_NUM, dtype=np.float32)
    gamma = Config.GAMMA
    lamda = Config.LAMDA

    for sample in reversed(list_sample_data):
        done = np.asarray(sample.done, dtype=np.float32)
        reward = np.asarray(sample.reward, dtype=np.float32)
        value = np.asarray(sample.value, dtype=np.float32)
        next_value = np.asarray(sample.next_value, dtype=np.float32)

        delta = reward + gamma * next_value * (1.0 - done) - value
        gae = delta + gamma * lamda * (1.0 - done) * gae
        sample.advantage = np.asarray(gae, dtype=np.float32)
        sample.reward_sum = np.asarray(value + sample.advantage, dtype=np.float32)
