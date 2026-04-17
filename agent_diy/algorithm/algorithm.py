#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
PPO algorithm implementation for the DIY Gorge Chase agent.
"""

import os
import time

import torch

from agent_diy.conf.conf import Config


class Algorithm:
    def __init__(self, model, optimizer, device=None, logger=None, monitor=None):
        self.device = device
        self.model = model
        self.optimizer = optimizer
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
        obs = self._stack(list_sample_data, "obs")
        legal_action = self._stack(list_sample_data, "legal_action")
        act = self._stack(list_sample_data, "act").view(-1, 1)
        old_prob = self._stack(list_sample_data, "prob")
        reward = self._stack(list_sample_data, "reward")
        advantage = self._stack(list_sample_data, "advantage")
        old_value = self._stack(list_sample_data, "value")
        reward_sum = self._stack(list_sample_data, "reward_sum")

        advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)

        self.model.set_train_mode()
        self.optimizer.zero_grad()

        logits, value_pred = self.model(obs)
        total_loss, info = self._compute_loss(
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
        self.train_step += 1

        now = time.time()
        if now - self.last_report_monitor_time >= Config.MONITOR_REPORT_INTERVAL_SEC:
            monitor_data = {
                "total_loss": round(total_loss.item(), 4),
                "value_loss": round(info["value_loss"].item(), 4),
                "policy_loss": round(info["policy_loss"].item(), 4),
                "entropy_loss": round(info["entropy_loss"].item(), 4),
                "reward": round(reward.mean().item(), 4),
            }
            if self.logger is not None:
                self.logger.info(
                    "[train] total_loss:%s policy_loss:%s value_loss:%s entropy:%s reward:%s"
                    % (
                        monitor_data["total_loss"],
                        monitor_data["policy_loss"],
                        monitor_data["value_loss"],
                        monitor_data["entropy_loss"],
                        monitor_data["reward"],
                    )
                )
            if self.monitor is not None:
                self.monitor.put_data({os.getpid(): monitor_data})
            self.last_report_monitor_time = now

        return {
            "total_loss": float(total_loss.item()),
            "policy_loss": float(info["policy_loss"].item()),
            "value_loss": float(info["value_loss"].item()),
            "entropy_loss": float(info["entropy_loss"].item()),
        }

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
        old_action_prob = (one_hot * old_prob).sum(dim=1, keepdim=True).clamp(1e-9)

        ratio = new_prob / old_action_prob
        adv = advantage.view(-1, 1)
        surr1 = ratio * adv
        surr2 = ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param) * adv
        policy_loss = -torch.minimum(surr1, surr2).mean()

        value_clip = old_value + (value_pred - old_value).clamp(-self.clip_param, self.clip_param)
        value_loss = 0.5 * torch.maximum(
            torch.square(reward_sum - value_pred),
            torch.square(reward_sum - value_clip),
        ).mean()

        entropy_loss = (-prob_dist * torch.log(prob_dist.clamp(1e-9, 1.0))).sum(dim=1).mean()
        total_loss = self.vf_coef * value_loss + policy_loss - self.var_beta * entropy_loss

        return total_loss, {
            "value_loss": value_loss,
            "policy_loss": policy_loss,
            "entropy_loss": entropy_loss,
        }

    @staticmethod
    def _masked_softmax(logits, legal_action):
        safe_legal = torch.where(
            legal_action.sum(dim=1, keepdim=True) > 0,
            legal_action,
            torch.cat(
                [
                    torch.ones((legal_action.shape[0], 8), device=legal_action.device),
                    torch.zeros((legal_action.shape[0], legal_action.shape[1] - 8), device=legal_action.device),
                ],
                dim=1,
            ),
        )
        label_max, _ = torch.max(logits * safe_legal, dim=1, keepdim=True)
        label = (logits - label_max) * safe_legal + 1e5 * (safe_legal - 1.0)
        return torch.nn.functional.softmax(label, dim=1)

    def _stack(self, list_sample_data, attr_name):
        tensors = []
        for sample in list_sample_data:
            value = getattr(sample, attr_name)
            if isinstance(value, torch.Tensor):
                tensor = value.float()
            else:
                tensor = torch.as_tensor(value, dtype=torch.float32)
            tensors.append(tensor)
        return torch.stack(tensors).to(self.device)
