#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Agent implementation for the DIY Gorge Chase PPO agent.
"""

import os
from pathlib import Path
import re

import numpy as np
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from kaiwudrl.interface.agent import BaseAgent

from agent_diy.algorithm.algorithm import Algorithm
from agent_diy.conf.conf import Config
from agent_diy.feature.definition import ActData, ObsData
from agent_diy.feature.preprocessor import Preprocessor
from agent_diy.model.model import Model


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        torch.manual_seed(0)
        np.random.seed(0)
        self.device = device or torch.device("cpu")
        self.model = Model(self.device).to(self.device)
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=Config.INIT_LEARNING_RATE_START,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        self.algorithm = Algorithm(self.model, self.optimizer, device=self.device, logger=logger, monitor=monitor)
        self.preprocessor = Preprocessor()
        self.last_action = -1
        self._loaded_model_signature = None
        self.logger = logger
        self.monitor = monitor
        super().__init__(agent_type, device, logger, monitor)

    def reset(self, env_obs=None):
        self.preprocessor.reset()
        self.preprocessor.set_runtime_mode("eval")
        self.last_action = -1

    def set_curriculum_context(self, stage_name=None, episode_idx=0):
        self.preprocessor.set_curriculum_context(stage_name=stage_name, episode_idx=episode_idx)

    def set_runtime_mode(self, mode):
        self.preprocessor.set_runtime_mode(mode)

    def observation_process(self, env_obs, preprocessor=None, extra_info=None):
        feature, legal_action, action_bias, reward, info = self.preprocessor.feature_process(env_obs, self.last_action)
        obs_data = ObsData(
            feature=list(feature),
            legal_action=list(legal_action),
            action_bias=list(action_bias),
        )
        remain_info = {"reward": reward, "info": info}
        return obs_data, remain_info

    def predict(self, list_obs_data):
        if not list_obs_data:
            return []

        features = np.array([obs.feature for obs in list_obs_data], dtype=np.float32)
        legal_actions = np.array([obs.legal_action for obs in list_obs_data], dtype=np.float32)
        action_biases = np.array([obs.action_bias for obs in list_obs_data], dtype=np.float32)

        logits_batch, value_batch = self._run_model(features)
        act_data_list = []
        for idx, obs_data in enumerate(list_obs_data):
            legal_action = legal_actions[idx]
            prob = self._legal_soft_max(logits_batch[idx] + action_biases[idx], legal_action)
            action = self._legal_sample(prob, use_max=False)
            d_action = self._legal_sample(prob, use_max=True)
            act_data_list.append(
                ActData(
                    action=[action],
                    d_action=[d_action],
                    prob=list(prob),
                    value=value_batch[idx],
                )
            )
        return act_data_list

    def exploit(self, env_obs):
        self.preprocessor.set_runtime_mode("eval")
        obs_data, _ = self.observation_process(env_obs)
        act_data = self.predict([obs_data])[0]
        return self.action_process(act_data, is_stochastic=False)

    def learn(self, list_sample_data):
        return self.algorithm.learn(list_sample_data)

    def save_model(self, path=None, id="1"):
        target_dir = self._resolve_ckpt_dir(path)
        target_dir.mkdir(parents=True, exist_ok=True)
        model_file_path = target_dir / f"model.ckpt-{str(id)}.pkl"
        state_dict_cpu = {name: tensor.detach().clone().cpu() for name, tensor in self.model.state_dict().items()}
        torch.save(state_dict_cpu, model_file_path)
        if self.logger is not None:
            self.logger.info(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        model_file_path = self._resolve_model_file(path=path, id=id)
        if model_file_path is None:
            if self.logger is not None:
                self.logger.info(f"skip load_model because no checkpoint was found for id={id}")
            return False

        stat = model_file_path.stat()
        signature = (str(model_file_path), int(stat.st_mtime_ns), int(stat.st_size))
        if signature == self._loaded_model_signature:
            return True

        self.model.load_state_dict(torch.load(model_file_path, map_location=self.device))
        self._loaded_model_signature = signature
        if self.logger is not None:
            self.logger.info(f"load model {model_file_path} successfully")
        return True

    def action_process(self, act_data, is_stochastic=True):
        action = act_data.action if is_stochastic else act_data.d_action
        self.last_action = int(action[0])
        return int(action[0])

    def get_episode_summary(self):
        return self.preprocessor.get_episode_summary()

    def _run_model(self, feature_batch):
        self.model.set_eval_mode()
        obs_tensor = torch.as_tensor(feature_batch, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits, value = self.model(obs_tensor, inference=True)
        return logits.cpu().numpy(), value.cpu().numpy()

    @staticmethod
    def _legal_soft_max(logits, legal_action):
        legal_action = np.asarray(legal_action, dtype=np.float32)
        if legal_action.shape[0] != Config.ACTION_NUM:
            fixed = np.zeros((Config.ACTION_NUM,), dtype=np.float32)
            limit = min(Config.ACTION_NUM, legal_action.shape[0])
            fixed[:limit] = legal_action[:limit]
            legal_action = fixed
        if legal_action.sum() <= 0:
            legal_action = np.array([1] * 8 + [0] * (Config.ACTION_NUM - 8), dtype=np.float32)

        huge = 1e20
        eps = 1e-5
        masked = logits - huge * (1.0 - legal_action)
        masked = np.clip(masked - np.max(masked, keepdims=True), -huge, 1.0)
        prob = (np.exp(masked) + eps) * legal_action
        return prob / (prob.sum(keepdims=True) * 1.00001)

    @staticmethod
    def _legal_sample(probs, use_max=False):
        if use_max:
            return int(np.argmax(probs))
        return int(np.argmax(np.random.multinomial(1, probs, size=1)))

    def _resolve_ckpt_dir(self, path, create=True):
        if path is not None:
            return Path(path)

        candidates = [
            Path.cwd() / "ckpt",
            Path(__file__).resolve().parents[1] / "ckpt",
        ]
        for candidate in candidates:
            if candidate.exists() or create:
                return candidate
        return None

    def _resolve_model_file(self, path=None, id="latest"):
        candidates = self._candidate_model_locations(path)
        all_files = []
        for candidate in candidates:
            all_files.extend(self._collect_model_files(candidate))

        if not all_files:
            return None

        target_id = str(id)
        if target_id != "latest":
            exact_name = f"model.ckpt-{target_id}.pkl"
            for file_path in sorted(all_files, key=self._file_sort_key, reverse=True):
                if file_path.name == exact_name:
                    return file_path
            return None

        return sorted(all_files, key=self._file_sort_key, reverse=True)[0]

    def _candidate_model_locations(self, path=None):
        candidates = []
        project_name = self.model.model_name

        def add_candidate(candidate):
            if candidate is None:
                return
            try:
                candidate = Path(candidate)
            except Exception:
                return
            if candidate not in candidates:
                candidates.append(candidate)

        if path is not None:
            add_candidate(path)
            add_candidate(Path(path) / "models")
            add_candidate(Path(path) / "models_new")

        for env_name in ("KAIWU_MODEL_PATH", "KAIWU_CKPT_PATH", "MODEL_PATH", "CKPT_PATH"):
            env_value = os.getenv(env_name)
            if env_value:
                add_candidate(env_value)
                add_candidate(Path(env_value) / "models")
                add_candidate(Path(env_value) / "models_new")

        add_candidate(Path.cwd() / "ckpt")
        add_candidate(Path(__file__).resolve().parents[1] / "ckpt")
        add_candidate(Path("/data/ckpt") / project_name)
        add_candidate(Path("/data/ckpt") / project_name / "models")
        add_candidate(Path("/data/ckpt") / project_name / "models_new")
        return candidates

    def _collect_model_files(self, candidate):
        files = []
        if not candidate.exists():
            return files
        if candidate.is_file():
            if self._is_model_file(candidate):
                files.append(candidate)
            return files
        try:
            for file_path in candidate.rglob("model.ckpt-*.pkl"):
                if file_path.is_file():
                    files.append(file_path)
        except Exception:
            return files
        return files

    @staticmethod
    def _is_model_file(path_obj):
        return path_obj.is_file() and bool(re.match(r"model\.ckpt-\w+\.pkl$", path_obj.name))

    @staticmethod
    def _file_sort_key(path_obj):
        name = path_obj.name
        match = re.match(r"model\.ckpt-(\d+)\.pkl$", name)
        numeric_id = int(match.group(1)) if match else -1
        stat = path_obj.stat()
        return (int(stat.st_mtime_ns), numeric_id)
