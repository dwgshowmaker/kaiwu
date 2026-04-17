#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Neural network model for the DIY Gorge Chase PPO agent.
"""

import torch.nn as nn

from agent_diy.conf.conf import Config


def make_fc_layer(in_features, out_features, gain=1.0):
    fc = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(fc.weight.data, gain=gain)
    nn.init.zeros_(fc.bias.data)
    return fc


class Model(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.model_name = "gorge_chase_diy"
        self.device = device

        input_dim = Config.DIM_OF_OBSERVATION
        hidden_dims = [256, 128, 64]

        layers = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    make_fc_layer(last_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                ]
            )
            last_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)

        self.actor_head = make_fc_layer(last_dim, Config.ACTION_NUM, gain=0.01)
        self.critic_head = make_fc_layer(last_dim, Config.VALUE_NUM, gain=1.0)

    def forward(self, obs, inference=False):
        hidden = self.backbone(obs)
        logits = self.actor_head(hidden)
        value = self.critic_head(hidden)
        return logits, value

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
