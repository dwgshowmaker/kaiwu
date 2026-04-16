#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Agent class for the DIY PPO baseline.
DIY PPO 基线 Agent 主类。
"""

import torch

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

import numpy as np
from kaiwudrl.interface.agent import BaseAgent

from agent_diy.algorithm.algorithm import Algorithm
from agent_diy.conf.conf import Config
from agent_diy.feature.definition import ActData, ObsData
from agent_diy.feature.preprocessor import Preprocessor
from agent_diy.model.model import Model


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        torch.manual_seed(0)
        self.device = device
        self.model = Model(device).to(self.device)
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=Config.INIT_LEARNING_RATE_START,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        self.algorithm = Algorithm(self.model, self.optimizer, self.device, logger, monitor)
        self.preprocessor = Preprocessor()
        self.last_action = -1
        self.logger = logger
        self.monitor = monitor
        super().__init__(agent_type, device, logger, monitor)

    def reset(self, env_obs=None):
        """Reset per-episode state."""
        self.preprocessor.reset()
        self.last_action = -1

    def predict(self, list_obs_data):
        """Stochastic inference for training."""
        feature = list_obs_data[0].feature
        legal_action = list_obs_data[0].legal_action

        logits, value, prob = self._run_model(feature, legal_action)
        prob = self._normalize_probs(prob)
        prob, safe_prior_used = self._apply_safety_prior(prob, list_obs_data[0])
        prob = self._normalize_probs(prob)

        action = self._legal_sample(prob, use_max=False)
        d_action = self._select_greedy_action(prob, list_obs_data[0])
        safe_action = int(getattr(list_obs_data[0], "safe_action", -1))

        return [
            ActData(
                action=[action],
                d_action=[d_action],
                prob=list(prob),
                value=value,
                safe_prior_used=int(safe_prior_used),
                safe_action_used=int(safe_prior_used and action == safe_action),
            )
        ]

    def exploit(self, env_obs):
        """Greedy inference for evaluation."""
        obs_data, _ = self.observation_process(env_obs)
        act_data = self.predict([obs_data])
        return self.action_process(act_data[0], is_stochastic=False)

    def learn(self, list_sample_data):
        """Train the model."""
        return self.algorithm.learn(list_sample_data)

    def save_model(self, path=None, id="1"):
        """Save model checkpoint."""
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        state_dict_cpu = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
        torch.save(state_dict_cpu, model_file_path)
        if self.logger:
            self.logger.info(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        """Load model checkpoint."""
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        try:
            state_dict = torch.load(model_file_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            if self.logger:
                self.logger.info(f"load model {model_file_path} successfully")
        except FileNotFoundError:
            if self.logger:
                self.logger.warning(f"skip loading missing model {model_file_path}")
        except RuntimeError as exc:
            if self.logger:
                self.logger.warning(
                    f"skip incompatible model {model_file_path}, keep current weights: {exc}"
                )

    def observation_process(self, env_obs):
        """Convert raw env_obs to ObsData and remain_info."""
        feature, legal_action, reward, metrics = self.preprocessor.feature_process(env_obs, self.last_action)
        obs_data = ObsData(
            feature=list(feature),
            legal_action=legal_action,
            safe_action=metrics.get("safe_action", -1),
            danger_level=metrics.get("danger_level", 0.0),
            safe_action_score=metrics.get("safe_action_score", 0.0),
            safe_path_len=metrics.get("safe_path_len", 0.0),
            safe_action_margin=metrics.get("safe_action_margin", 0.0),
            safe_is_flash=metrics.get("safe_is_flash", 0.0),
            safe_trap_risk=metrics.get("safe_trap_risk", 0.0),
        )
        remain_info = {"reward": reward}
        remain_info.update(metrics)
        return obs_data, remain_info

    def action_process(self, act_data, is_stochastic=True):
        """Unpack ActData to int action and update last_action."""
        action = act_data.action if is_stochastic else act_data.d_action
        self.last_action = int(action[0])
        return int(action[0])

    def _run_model(self, feature, legal_action):
        """Run model inference, return logits, value and action prob."""
        self.model.set_eval_mode()
        obs_tensor = torch.tensor(np.array([feature]), dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits, value = self.model(obs_tensor, inference=True)

        logits_np = logits.cpu().numpy()[0]
        value_np = value.cpu().numpy()[0]

        legal_action_np = np.array(legal_action, dtype=np.float32)
        prob = self._legal_soft_max(logits_np, legal_action_np)
        prob = self._normalize_probs(prob)

        return logits_np, value_np, prob

    def _legal_soft_max(self, input_hidden, legal_action):
        """Softmax with legal action masking."""
        _w, _e = 1e20, 1e-5
        tmp = input_hidden - _w * (1.0 - legal_action)
        tmp_max = np.max(tmp, keepdims=True)
        tmp = np.clip(tmp - tmp_max, -_w, 1)
        tmp = (np.exp(tmp) + _e) * legal_action
        return tmp / (np.sum(tmp, keepdims=True) * 1.00001)

    def _legal_sample(self, probs, use_max=False):
        """Sample action from probability distribution."""
        probs = self._normalize_probs(probs)
        if use_max:
            return int(np.argmax(probs))

        random_value = float(np.random.random())
        cdf = np.cumsum(probs, dtype=np.float64)
        cdf[-1] = 1.0
        return int(np.searchsorted(cdf, random_value, side="right"))

    def _select_greedy_action(self, probs, obs_data):
        safe_context = self._get_safe_action_context(obs_data, len(probs))
        if safe_context is not None and self._calc_safe_prior_weight(safe_context) >= 0.24:
            return safe_context["safe_action"]
        return self._legal_sample(probs, use_max=True)

    def _apply_safety_prior(self, probs, obs_data):
        """Blend model policy with a danger-only escape prior for early PPO stability."""
        safe_context = self._get_safe_action_context(obs_data, len(probs))
        if safe_context is None:
            return probs, False

        prior_weight = self._calc_safe_prior_weight(safe_context)
        if prior_weight <= 1e-6:
            return probs, False

        safe_action = safe_context["safe_action"]
        prior = np.zeros_like(probs, dtype=np.float32)
        prior[safe_action] = 1.0
        mixed = (1.0 - prior_weight) * np.array(probs, dtype=np.float32) + prior_weight * prior
        mixed = self._normalize_probs(mixed)
        if mixed.size == 0:
            return probs, False
        return mixed, True

    def _get_safe_action_context(self, obs_data, prob_size):
        safe_action = int(getattr(obs_data, "safe_action", -1))
        if not (0 <= safe_action < prob_size):
            return None

        return {
            "safe_action": safe_action,
            "danger_level": float(getattr(obs_data, "danger_level", 0.0)),
            "safe_action_score": float(getattr(obs_data, "safe_action_score", 0.0)),
            "safe_path_len": float(getattr(obs_data, "safe_path_len", 0.0)),
            "safe_action_margin": float(getattr(obs_data, "safe_action_margin", 0.0)),
            "safe_is_flash": bool(getattr(obs_data, "safe_is_flash", 0.0)),
            "safe_trap_risk": float(getattr(obs_data, "safe_trap_risk", 0.0)),
        }

    def _calc_safe_prior_weight(self, safe_context):
        danger_level = safe_context["danger_level"]
        if danger_level < 0.25:
            return 0.0

        if safe_context["safe_is_flash"]:
            safe_margin = safe_context["safe_action_margin"]
            safe_path_len = safe_context["safe_path_len"]
            safe_trap_risk = safe_context["safe_trap_risk"]
            if danger_level < 0.35 or safe_path_len < 1.0 or safe_trap_risk >= 0.85:
                return 0.0
            if safe_margin < -0.1 and danger_level < 0.78:
                return 0.0

            prior_weight = 0.12 + 0.28 * danger_level + 0.05 * np.clip(safe_margin, 0.0, 2.0)
            if danger_level >= 0.82:
                prior_weight += 0.04
            prior_weight *= max(0.45, 1.0 - 0.5 * np.clip(safe_trap_risk, 0.0, 1.0))
            if safe_path_len <= 1.0:
                prior_weight *= 0.85
            return min(0.48, float(prior_weight))

        prior_weight = min(0.58, 0.14 + 0.44 * danger_level)
        if safe_context["safe_action_margin"] > 0.75:
            prior_weight = min(0.62, prior_weight + 0.04)
        return float(prior_weight)

    def _normalize_probs(self, probs):
        """Clamp and normalize probabilities to a numerically safe distribution."""
        probs = np.asarray(probs, dtype=np.float64).reshape(-1)
        if probs.size == 0:
            return probs

        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs = np.clip(probs, 0.0, None)

        total = float(np.sum(probs, dtype=np.float64))
        if total <= 1e-12:
            probs.fill(1.0 / probs.size)
            return probs

        probs /= total
        pivot = int(np.argmax(probs))
        others_sum = float(np.sum(np.delete(probs, pivot), dtype=np.float64))
        probs[pivot] = max(0.0, 1.0 - others_sum)

        total = float(np.sum(probs, dtype=np.float64))
        if total <= 1e-12:
            probs.fill(1.0 / probs.size)
        else:
            probs /= total

        return probs
