#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Agent implementation for the DIY Gorge Chase solution.
"""

import glob
import os

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
        self.device = device
        self.logger = logger
        self.monitor = monitor
        self.model = Model(device=device).to(self.device)
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=Config.INIT_LEARNING_RATE_START,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        self.algorithm = Algorithm(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=None,
            device=self.device,
            logger=logger,
            monitor=monitor,
        )
        self.preprocessor = Preprocessor()
        self.last_action = -1
        super().__init__(agent_type, device, logger, monitor)

    def reset(self, env_obs=None):
        self.preprocessor.reset()
        self.last_action = -1

    def observation_process(self, env_obs):
        feature, legal_action, reward, info = self.preprocessor.feature_process(env_obs, self.last_action)
        obs_data = ObsData(feature=list(feature), legal_action=list(legal_action))
        remain_info = {"reward": reward}
        remain_info.update(info)
        return obs_data, remain_info

    def predict(self, list_obs_data):
        try:
            obs_data = list_obs_data[0]
            feature = obs_data.feature
            legal_action = obs_data.legal_action

            _, value, model_prob = self._run_model(feature, legal_action)
            prob = self._compose_action_probs(
                obs_data=obs_data,
                model_prob=model_prob,
                use_exploit=False,
            )
            d_action = self._legal_sample(prob, use_max=True)
            action = self._select_train_action(prob, d_action)

            return [
                ActData(
                    action=[action],
                    d_action=[d_action],
                    prob=list(prob),
                    value=value,
                )
            ]
        except Exception as exc:
            self._log("error", f"predict fallback because of exception: {exc}")
            return [self._build_fallback_act_data(list_obs_data[0])]

    def exploit(self, env_obs):
        try:
            obs_data, _ = self.observation_process(env_obs)
            _, value, model_prob = self._run_model(obs_data.feature, obs_data.legal_action)
            prob = self._compose_action_probs(
                obs_data=obs_data,
                model_prob=model_prob,
                use_exploit=True,
            )
            greedy_action = self._legal_sample(prob, use_max=True)
            act_data = ActData(
                action=[greedy_action],
                d_action=[greedy_action],
                prob=list(prob),
                value=value,
            )
        except Exception as exc:
            self._log("error", f"exploit fallback because of exception: {exc}")
            obs_data, _ = self.observation_process(env_obs)
            act_data = self._build_fallback_act_data(obs_data)
        return self.action_process(act_data, is_stochastic=False)

    def learn(self, list_sample_data):
        return self.algorithm.learn(list_sample_data)

    def save_model(self, path=None, id="1"):
        save_dir = self._resolve_save_dir(path)
        os.makedirs(save_dir, exist_ok=True)

        model_file_path = os.path.join(save_dir, f"model.ckpt-{str(id)}.pkl")
        state_dict_cpu = {k: v.detach().clone().cpu() for k, v in self.model.state_dict().items()}
        torch.save(state_dict_cpu, model_file_path)
        self._log("info", f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        model_file_path = self._resolve_load_file(path, id)
        if not model_file_path:
            self._log("info", f"skip loading model because checkpoint {id} was not found")
            return

        state_dict = torch.load(model_file_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self._log("info", f"load model {model_file_path} successfully")

    def action_process(self, act_data, is_stochastic=True):
        action = act_data.action if is_stochastic else act_data.d_action
        action_value = int(action[0] if isinstance(action, (list, tuple, np.ndarray)) else action)
        self.last_action = action_value
        return action_value

    def _run_model(self, feature, legal_action):
        self.model.set_eval_mode()
        obs_tensor = torch.tensor(np.asarray([feature], dtype=np.float32), dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits, value = self.model(obs_tensor, inference=True)

        logits_np = logits.detach().cpu().numpy()[0]
        value_np = value.detach().cpu().numpy()[0]
        legal_action_np = self._sanitize_legal_action(legal_action)
        prob = self._legal_soft_max(logits_np, legal_action_np)
        return logits_np, value_np, prob

    def _legal_soft_max(self, input_hidden, legal_action):
        legal = self._sanitize_legal_action(legal_action).astype(np.float64)

        masked_logits = input_hidden - (1.0 - legal) * 1.0e9
        masked_logits = masked_logits - np.max(masked_logits)
        exp_logits = np.exp(np.clip(masked_logits, -50.0, 50.0)) * legal
        return self._normalize_probs(exp_logits, legal)

    def _legal_sample(self, probs, use_max=False):
        probs = self._normalize_probs(probs)
        if use_max:
            return int(np.argmax(probs))

        total = float(np.sum(probs, dtype=np.float64))
        if total <= 0.0:
            return 0
        random_value = float(np.random.random()) * total
        cumulative = np.cumsum(probs, dtype=np.float64)
        index = int(np.searchsorted(cumulative, random_value, side="right"))
        if index >= probs.shape[0]:
            index = probs.shape[0] - 1
        return index

    def _sanitize_legal_action(self, legal_action):
        legal = np.asarray(legal_action, dtype=np.float32).reshape(-1)
        if legal.size < Config.ACTION_NUM:
            pad = np.zeros(Config.ACTION_NUM - legal.size, dtype=np.float32)
            legal = np.concatenate([legal, pad], axis=0)
        legal = legal[: Config.ACTION_NUM]
        legal = np.where(np.isfinite(legal), legal, 0.0)
        legal = np.clip(legal, 0.0, 1.0)
        if float(legal.sum()) <= 0.0:
            legal[:8] = 1.0
        return legal

    def _normalize_probs(self, probs, legal_action=None):
        prob_array = np.asarray(probs, dtype=np.float64).reshape(-1)
        if prob_array.size < Config.ACTION_NUM:
            pad = np.zeros(Config.ACTION_NUM - prob_array.size, dtype=np.float64)
            prob_array = np.concatenate([prob_array, pad], axis=0)
        prob_array = prob_array[: Config.ACTION_NUM]
        prob_array = np.where(np.isfinite(prob_array), prob_array, 0.0)
        prob_array = np.clip(prob_array, 0.0, None)

        if legal_action is not None:
            legal = self._sanitize_legal_action(legal_action).astype(np.float64)
            prob_array = prob_array * legal
        else:
            legal = None

        total = float(np.sum(prob_array, dtype=np.float64))
        if total <= 0.0:
            if legal is None:
                prob_array = np.zeros(Config.ACTION_NUM, dtype=np.float64)
                prob_array[0] = 1.0
            else:
                legal_sum = float(np.sum(legal, dtype=np.float64))
                prob_array = legal / max(legal_sum, 1.0)
        else:
            prob_array = prob_array / total

        prob_array = np.clip(prob_array, 0.0, 1.0)
        total = float(np.sum(prob_array, dtype=np.float64))
        if total <= 0.0:
            prob_array = np.zeros(Config.ACTION_NUM, dtype=np.float64)
            prob_array[0] = 1.0
        else:
            prob_array = prob_array / total

        prob_array[-1] = max(0.0, 1.0 - float(np.sum(prob_array[:-1], dtype=np.float64)))
        total = float(np.sum(prob_array, dtype=np.float64))
        if total <= 0.0:
            prob_array = np.zeros(Config.ACTION_NUM, dtype=np.float64)
            prob_array[0] = 1.0
        else:
            prob_array = prob_array / total

        return prob_array.astype(np.float32)

    def _build_fallback_act_data(self, obs_data):
        legal_action = self._sanitize_legal_action(obs_data.legal_action)
        fallback_action = int(np.argmax(legal_action))
        fallback_prob = self._normalize_probs(legal_action, legal_action)
        fallback_value = np.zeros(Config.VALUE_NUM, dtype=np.float32)
        return ActData(
            action=[fallback_action],
            d_action=[fallback_action],
            prob=list(fallback_prob),
            value=fallback_value,
        )

    def _compose_action_probs(self, obs_data, model_prob, use_exploit=False):
        legal_action = self._sanitize_legal_action(obs_data.legal_action)
        normalized_model_prob = self._normalize_probs(model_prob, legal_action)
        heuristic_prob = self._build_heuristic_probs(legal_action)
        if heuristic_prob is None:
            return normalized_model_prob

        heuristic_weight = Config.EXPLOIT_HEURISTIC_WEIGHT if use_exploit else Config.TRAIN_HEURISTIC_WEIGHT
        context = getattr(self.preprocessor, "policy_context", None)
        if context is not None and float(context.get("min_monster_dist", Config.MONSTER_DIST_MAX)) <= Config.HEURISTIC_DANGER_DIST:
            heuristic_weight = max(heuristic_weight, Config.DANGER_HEURISTIC_WEIGHT)

        blended_prob = (1.0 - heuristic_weight) * normalized_model_prob + heuristic_weight * heuristic_prob
        return self._normalize_probs(blended_prob, legal_action)

    def _select_train_action(self, prob, greedy_action):
        context = getattr(self.preprocessor, "policy_context", None)
        if context is not None and float(context.get("min_monster_dist", Config.MONSTER_DIST_MAX)) <= Config.HEURISTIC_DANGER_DIST:
            return greedy_action
        if float(np.random.random()) <= Config.TRAIN_GREEDY_PROB:
            return greedy_action
        return self._legal_sample(prob, use_max=False)

    def _build_heuristic_probs(self, legal_action):
        context = getattr(self.preprocessor, "policy_context", None)
        if not context:
            return None

        scores = np.full(Config.ACTION_NUM, -1.0e9, dtype=np.float64)
        legal = self._sanitize_legal_action(legal_action)
        for action in range(Config.ACTION_NUM):
            if legal[action] <= 0.0:
                continue
            scores[action] = self._score_action(context, action)

        valid_scores = scores[legal > 0.0]
        if valid_scores.size == 0:
            return self._normalize_probs(legal, legal)

        max_score = float(np.max(valid_scores))
        scaled_scores = np.exp(np.clip((scores - max_score) / Config.HEURISTIC_TEMPERATURE, -30.0, 30.0)) * legal
        return self._normalize_probs(scaled_scores, legal)

    def _score_action(self, context, action):
        dir_idx = int(action % 8)
        is_flash = action >= 8
        route = context["routes"][dir_idx] if dir_idx < len(context["routes"]) else {
            "step_1": 0.0,
            "step_2": 0.0,
            "corridor": 0.0,
            "passable_len": 0,
        }

        travel = self._estimate_action_travel(context, dir_idx, is_flash, route)
        dx, dz = Config.MOVE_DIRS[dir_idx]
        hero_x, hero_z = context["hero_pos"]
        new_pos = (hero_x + dx * travel, hero_z + dz * travel)

        current_min_dist = float(context.get("min_monster_dist", Config.MONSTER_DIST_MAX))
        new_min_dist = current_min_dist
        if context["monster_positions"]:
            new_min_dist = min(
                self._distance(new_pos, monster_pos) for monster_pos in context["monster_positions"]
            )
        escape_gain = new_min_dist - current_min_dist

        score = 0.0
        score += float(route.get("corridor", 0.0)) * Config.HEURISTIC_CORRIDOR_WEIGHT
        score += float(route.get("step_1", 0.0)) * 0.35
        score += float(route.get("step_2", 0.0)) * 0.25
        score += float(new_min_dist) * Config.HEURISTIC_DISTANCE_WEIGHT

        danger = self._clip01((Config.HEURISTIC_DANGER_DIST - current_min_dist) / Config.HEURISTIC_DANGER_DIST)
        critical = self._clip01((Config.HEURISTIC_CRITICAL_DIST - current_min_dist) / Config.HEURISTIC_CRITICAL_DIST)
        score += escape_gain * Config.HEURISTIC_ESCAPE_GAIN_WEIGHT * (1.0 + 2.5 * danger)

        if current_min_dist <= Config.HEURISTIC_CRITICAL_DIST and not is_flash:
            score -= 2.0
        if new_min_dist <= 3.5:
            score -= 3.0
        if float(route.get("corridor", 0.0)) <= 0.1 and current_min_dist <= Config.HEURISTIC_DANGER_DIST:
            score -= Config.HEURISTIC_DEAD_END_PENALTY
        if travel <= 0:
            score -= 2.5

        score += self._score_target_progress(context, new_pos, current_min_dist, "treasures")
        score += self._score_target_progress(context, new_pos, current_min_dist, "buffs")

        last_action = context.get("last_action", -1)
        if last_action is not None and last_action >= 0:
            last_dir = int(last_action % 8)
            reverse_dir = (last_dir + 4) % 8
            if dir_idx == reverse_dir:
                score -= Config.HEURISTIC_REVERSE_PENALTY
            if dir_idx == last_dir:
                score += Config.HEURISTIC_MOVE_STABILITY_WEIGHT * float(route.get("corridor", 0.0))
                if context.get("stuck_count", 0) > 0:
                    score -= Config.HEURISTIC_STUCK_ACTION_PENALTY

        if is_flash:
            score -= Config.HEURISTIC_FLASH_BASE_PENALTY
            if current_min_dist > Config.HEURISTIC_FLASH_SAFE_DIST:
                score -= 3.0
            score += escape_gain * Config.HEURISTIC_FLASH_ESCAPE_WEIGHT * (1.0 + 3.0 * danger)
            if travel >= 4:
                score += 0.2
            if travel <= 1:
                score -= 1.5
        else:
            score += 0.2
            if current_min_dist <= Config.HEURISTIC_CRITICAL_DIST and escape_gain > 0:
                score += 0.8

        score += critical * (0.6 if not is_flash else 0.9)
        return float(score)

    def _estimate_action_travel(self, context, dir_idx, is_flash, route):
        passable_len = int(route.get("passable_len", 0))
        if is_flash:
            flash_range = Config.FLASH_RANGE_DIAGONAL if dir_idx % 2 == 1 else Config.FLASH_RANGE_STRAIGHT
            return max(0, min(passable_len, flash_range))
        move_speed = int(context.get("move_speed", 1))
        if move_speed >= 2 and passable_len >= 2:
            return 2
        if passable_len >= 1:
            return 1
        return 0

    def _score_target_progress(self, context, new_pos, current_min_dist, target_key):
        targets = context.get(target_key, [])
        if not targets:
            return 0.0

        if target_key == "treasures":
            if current_min_dist < Config.HEURISTIC_TREASURE_SAFE_DIST:
                weight = 0.35
            else:
                weight = 1.0
            if int(context.get("treasure_count", 0)) <= 0 and int(context.get("step_no", 0)) >= 500:
                weight += 0.4
            if current_min_dist >= 28.0:
                weight += 0.35
            return self._best_target_progress_score(
                new_pos=new_pos,
                targets=targets,
                progress_weight=Config.HEURISTIC_TREASURE_PROGRESS_WEIGHT,
                base_weight=weight,
                reach_bonus=Config.HEURISTIC_TARGET_REACH_BONUS,
                safety_gate=0.35,
            )

        if context.get("buff_remaining", 0.0) > 0.0:
            return 0.0
        weight = 0.9 if current_min_dist < Config.HEURISTIC_DANGER_DIST else 0.55
        return self._best_target_progress_score(
            new_pos=new_pos,
            targets=targets,
            progress_weight=Config.HEURISTIC_BUFF_PROGRESS_WEIGHT,
            base_weight=weight,
            reach_bonus=0.65,
            safety_gate=0.3,
        )

    def _best_target_progress_score(self, new_pos, targets, progress_weight, base_weight, reach_bonus, safety_gate):
        best_score = 0.0
        for target in targets[:4]:
            before_dist = float(target["dist"])
            after_dist = self._distance(new_pos, target["pos"])
            progress = before_dist - after_dist
            value = float(target.get("value", 0.0))
            score = progress * progress_weight * value * base_weight
            if after_dist <= 1.5 and value >= safety_gate:
                score += reach_bonus
            if score > best_score:
                best_score = score
        return best_score

    def _distance(self, pos_a, pos_b):
        return float(
            np.hypot(
                float(pos_a[0]) - float(pos_b[0]),
                float(pos_a[1]) - float(pos_b[1]),
            )
        )

    def _clip01(self, value):
        return float(np.clip(value, 0.0, 1.0))

    def _resolve_save_dir(self, path):
        if path:
            return path
        return os.path.join(os.getcwd(), "ckpt")

    def _resolve_load_file(self, path, id):
        candidate_dirs = []
        if path:
            candidate_dirs.append(path)
        candidate_dirs.append(os.path.join(os.getcwd(), "ckpt"))
        candidate_dirs.append(os.path.join(os.getcwd(), "agent_diy", "ckpt"))

        seen = set()
        unique_dirs = []
        for candidate_dir in candidate_dirs:
            if not candidate_dir:
                continue
            norm_dir = os.path.abspath(candidate_dir)
            if norm_dir not in seen:
                seen.add(norm_dir)
                unique_dirs.append(norm_dir)

        for candidate_dir in unique_dirs:
            exact_file = os.path.join(candidate_dir, f"model.ckpt-{str(id)}.pkl")
            if os.path.exists(exact_file):
                return exact_file

        if str(id) != "latest":
            return None

        latest_candidate = None
        latest_numeric_id = -1
        latest_mtime = -1.0
        for candidate_dir in unique_dirs:
            for file_path in glob.glob(os.path.join(candidate_dir, "model.ckpt-*.pkl")):
                file_name = os.path.basename(file_path)
                model_id = file_name[len("model.ckpt-") : -len(".pkl")]
                if model_id.isdigit():
                    numeric_id = int(model_id)
                    if numeric_id > latest_numeric_id:
                        latest_numeric_id = numeric_id
                        latest_candidate = file_path
                elif latest_candidate is None:
                    mtime = os.path.getmtime(file_path)
                    if mtime > latest_mtime:
                        latest_mtime = mtime
                        latest_candidate = file_path
        return latest_candidate

    def _log(self, level, message):
        if self.logger is None:
            return
        log_fn = getattr(self.logger, level, None)
        if callable(log_fn):
            log_fn(message)
