#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Neural network model for the DIY Gorge Chase agent.
"""

import math

import torch.nn as nn

from agent_diy.conf.conf import Config


def make_fc_layer(in_features, out_features, std=math.sqrt(2.0)):
    layer = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(layer.weight, gain=std)
    nn.init.zeros_(layer.bias)
    return layer


class Model(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.device = device
        self.model_name = "gorge_chase_diy_ppo"

        last_dim = Config.DIM_OF_OBSERVATION
        backbone_layers = []
        for hidden_dim in Config.MODEL_HIDDEN_DIMS:
            backbone_layers.extend(
                [
                    make_fc_layer(last_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                ]
            )
            last_dim = hidden_dim
        self.backbone = nn.Sequential(*backbone_layers)
        self.actor_head = make_fc_layer(last_dim, Config.ACTION_NUM, std=0.01)
        self.critic_head = make_fc_layer(last_dim, Config.VALUE_NUM, std=1.0)

    def forward(self, obs, inference=False):
        hidden = self.backbone(obs)
        logits = self.actor_head(hidden)
        value = self.critic_head(hidden)
        return logits, value

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
