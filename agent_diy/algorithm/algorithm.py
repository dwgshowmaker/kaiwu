#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
PPO algorithm for the DIY Gorge Chase agent.
"""

import os
import time

import numpy as np
import torch

from agent_diy.conf.conf import Config


class Algorithm:
    def __init__(self, model, optimizer, scheduler=None, device=None, logger=None, monitor=None):
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.parameters = [p for group in self.optimizer.param_groups for p in group["params"]]
        self.logger = logger
        self.monitor = monitor

        self.label_size = Config.ACTION_NUM
        self.var_beta = Config.BETA_START
        self.vf_coef = Config.VF_COEF
        self.clip_param = Config.CLIP_PARAM

        self.last_report_monitor_time = 0.0
        self.train_step = 0

    def learn(self, list_sample_data):
        if not list_sample_data:
            return

        obs = self._batch_tensor([sample.obs for sample in list_sample_data])
        legal_action = self._batch_tensor([sample.legal_action for sample in list_sample_data])
        act = self._batch_tensor([sample.act for sample in list_sample_data]).view(-1, 1)
        old_prob = self._batch_tensor([sample.prob for sample in list_sample_data])
        reward = self._batch_tensor([sample.reward for sample in list_sample_data])
        advantage = self._batch_tensor([sample.advantage for sample in list_sample_data])
        old_value = self._batch_tensor([sample.value for sample in list_sample_data])
        reward_sum = self._batch_tensor([sample.reward_sum for sample in list_sample_data])

        self.model.set_train_mode()
        self.optimizer.zero_grad()

        logits, value_pred = self.model(obs)
        total_loss, info_list = self._compute_loss(
            logits=logits,
            value_pred=value_pred,
            legal_action=legal_action,
            old_action=act,
            old_prob=old_prob,
            advantage=advantage,
            old_value=old_value,
            reward_sum=reward_sum,
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters, Config.GRAD_CLIP_RANGE)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        self.train_step += 1
        self._report(total_loss, info_list, reward)

    def _compute_loss(
        self,
        logits,
        value_pred,
        legal_action,
        old_action,
        old_prob,
        advantage,
        old_value,
        reward_sum,
    ):
        prob_dist = self._masked_softmax(logits, legal_action)

        one_hot = torch.nn.functional.one_hot(old_action[:, 0].long(), self.label_size).float()
        new_prob = (one_hot * prob_dist).sum(dim=1, keepdim=True)
        old_action_prob = (one_hot * old_prob).sum(dim=1, keepdim=True).clamp(1.0e-9)
        ratio = new_prob / old_action_prob

        adv = advantage.view(-1, 1)
        policy_loss1 = -ratio * adv
        policy_loss2 = -ratio.clamp(1 - self.clip_param, 1 + self.clip_param) * adv
        policy_loss = torch.maximum(policy_loss1, policy_loss2).mean()

        value_clip = old_value + (value_pred - old_value).clamp(-self.clip_param, self.clip_param)
        value_loss = 0.5 * torch.maximum(
            torch.square(reward_sum - value_pred),
            torch.square(reward_sum - value_clip),
        ).mean()

        entropy_loss = (-prob_dist * torch.log(prob_dist.clamp(1.0e-9, 1.0))).sum(dim=1).mean()
        total_loss = self.vf_coef * value_loss + policy_loss - self.var_beta * entropy_loss
        return total_loss, [value_loss, policy_loss, entropy_loss]

    def _masked_softmax(self, logits, legal_action):
        legal_action = legal_action.clamp(0.0, 1.0)
        label_max, _ = torch.max(logits * legal_action, dim=1, keepdim=True)
        label = logits - label_max
        label = label * legal_action
        label = label + 1.0e5 * (legal_action - 1.0)
        return torch.nn.functional.softmax(label, dim=1)

    def _batch_tensor(self, values):
        tensor_values = []
        for value in values:
            if isinstance(value, torch.Tensor):
                tensor_values.append(value.detach().cpu().numpy())
            else:
                tensor_values.append(np.asarray(value, dtype=np.float32))
        batch = np.asarray(tensor_values, dtype=np.float32)
        return torch.as_tensor(batch, dtype=torch.float32, device=self.device)

    def _report(self, total_loss, info_list, reward):
        now = time.time()
        if now - self.last_report_monitor_time < Config.MONITOR_REPORT_INTERVAL_SEC:
            return

        results = {
            "total_loss": round(total_loss.item(), 4),
            "value_loss": round(info_list[0].item(), 4),
            "policy_loss": round(info_list[1].item(), 4),
            "entropy_loss": round(info_list[2].item(), 4),
            "reward": round(reward.mean().item(), 4),
        }

        if self.logger is not None:
            self.logger.info(
                f"[train] total_loss:{results['total_loss']} "
                f"policy_loss:{results['policy_loss']} "
                f"value_loss:{results['value_loss']} "
                f"entropy:{results['entropy_loss']} "
                f"reward:{results['reward']}"
            )
        if self.monitor is not None:
            self.monitor.put_data({os.getpid(): results})
        self.last_report_monitor_time = now
