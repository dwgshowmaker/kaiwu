#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Training workflow for the DIY Gorge Chase PPO agent.
"""

from collections import deque
import copy
import os
import time

import numpy as np

from agent_diy.conf.conf import Config
from agent_diy.feature.definition import SampleData, sample_process
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery
from tools.metrics_utils import get_training_metrics
from tools.train_env_conf_validate import read_usr_conf


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    env = envs[0]
    agent = agents[0]
    last_save_model_time = time.time()

    base_usr_conf = read_usr_conf("agent_diy/conf/train_env_conf.toml", logger)
    if base_usr_conf is None:
        if logger is not None:
            logger.error("usr_conf is None, please check agent_diy/conf/train_env_conf.toml")
        return

    runner = EpisodeRunner(env=env, agent=agent, base_usr_conf=base_usr_conf, logger=logger, monitor=monitor)

    while True:
        for episode_data in runner.run_episodes():
            agent.send_sample_data(episode_data)
            episode_data.clear()

            now = time.time()
            if now - last_save_model_time >= Config.MODEL_SAVE_INTERVAL_SEC:
                agent.save_model()
                last_save_model_time = now


class EpisodeRunner:
    def __init__(self, env, agent, base_usr_conf, logger, monitor):
        self.env = env
        self.agent = agent
        self.base_usr_conf = base_usr_conf
        self.logger = logger
        self.monitor = monitor
        self.episode_cnt = 0
        self.last_report_monitor_time = 0.0
        self.last_get_training_metrics_time = 0.0
        self.recent_steps = deque(maxlen=Config.CURRICULUM_WINDOW_SIZE)
        self.recent_scores = deque(maxlen=Config.CURRICULUM_WINDOW_SIZE)
        self.recent_treasures = deque(maxlen=Config.CURRICULUM_WINDOW_SIZE)

    def run_episodes(self):
        while True:
            now = time.time()
            if now - self.last_get_training_metrics_time >= Config.MONITOR_REPORT_INTERVAL_SEC:
                training_metrics = get_training_metrics()
                self.last_get_training_metrics_time = now
                if training_metrics is not None and self.logger is not None:
                    self.logger.info(f"training_metrics is {training_metrics}")

            self.episode_cnt += 1
            usr_conf, stage_name = self._build_episode_usr_conf(self.episode_cnt)
            env_obs = self.env.reset(usr_conf)
            if handle_disaster_recovery(env_obs, self.logger):
                continue

            self.agent.reset(env_obs)
            self.agent.set_curriculum_context(stage_name=stage_name, episode_idx=self.episode_cnt)
            self.agent.set_runtime_mode("train")
            if Config.LOAD_LATEST_ON_EPISODE_START:
                self.agent.load_model(id="latest")

            obs_data, _ = self.agent.observation_process(env_obs)
            collector = []
            total_reward = 0.0
            step = 0
            done = False

            if self.logger is not None:
                self.logger.info(f"Episode {self.episode_cnt} start, curriculum stage: {stage_name}")

            while not done:
                act_data = self.agent.predict([obs_data])[0]
                act = self.agent.action_process(act_data)

                _, env_obs = self.env.step(act)
                if handle_disaster_recovery(env_obs, self.logger):
                    break

                terminated = bool(env_obs.get("terminated", False))
                truncated = bool(env_obs.get("truncated", False))
                done = terminated or truncated
                step += 1

                next_obs_data, next_remain_info = self.agent.observation_process(env_obs)
                reward = np.array(next_remain_info.get("reward", [0.0]), dtype=np.float32).reshape(-1)[:1]
                total_reward += float(reward[0])

                frame = SampleData(
                    obs=np.array(obs_data.feature, dtype=np.float32),
                    legal_action=np.array(obs_data.legal_action, dtype=np.float32),
                    action_bias=np.array(obs_data.action_bias, dtype=np.float32),
                    act=np.array([act_data.action[0]], dtype=np.float32),
                    reward=reward,
                    done=np.array([float(done)], dtype=np.float32),
                    reward_sum=np.zeros((1,), dtype=np.float32),
                    value=np.array(act_data.value, dtype=np.float32).reshape(-1)[:1],
                    next_value=np.zeros((1,), dtype=np.float32),
                    advantage=np.zeros((1,), dtype=np.float32),
                    prob=np.array(act_data.prob, dtype=np.float32),
                )
                collector.append(frame)

                if done:
                    final_reward, result_str, episode_monitor = self._finalize_episode(
                        env_obs=env_obs,
                        terminated=terminated,
                        truncated=truncated,
                        total_reward=total_reward,
                        step=step,
                    )
                    if collector:
                        collector[-1].reward = collector[-1].reward + np.array([final_reward], dtype=np.float32)

                    if self.logger is not None:
                        self.logger.info(
                            "[GAMEOVER] episode:%s steps:%s result:%s score:%.1f treasure:%s buff:%s flash:%s total_reward:%.3f"
                            % (
                                self.episode_cnt,
                                step,
                                result_str,
                                episode_monitor["total_score"],
                                episode_monitor["treasure_count"],
                                episode_monitor["buff_count"],
                                episode_monitor["flash_count"],
                                total_reward + final_reward,
                            )
                        )

                    if self.monitor is not None and (
                        time.time() - self.last_report_monitor_time >= Config.MONITOR_REPORT_INTERVAL_SEC
                    ):
                        self.monitor.put_data({os.getpid(): episode_monitor})
                        self.last_report_monitor_time = time.time()

                    if collector:
                        yield sample_process(collector)
                    break

                obs_data = next_obs_data

    def _build_episode_usr_conf(self, episode_cnt):
        usr_conf = copy.deepcopy(self.base_usr_conf)
        env_conf = usr_conf.setdefault("env_conf", {}) if isinstance(usr_conf, dict) else {}
        stage_name = self._select_curriculum_stage(episode_cnt)

        if stage_name == "stage_a_survive":
            env_conf.update(
                {
                    "map": [1, 2, 3, 4],
                    "map_random": True,
                    "treasure_count": 10,
                    "buff_count": 2,
                    "monster_interval": 600,
                    "monster_speedup": 1000,
                    "max_step": 800,
                }
            )
        elif stage_name == "stage_b_standard":
            env_conf.update(
                {
                    "map": [1, 2, 3, 4, 5, 6, 7],
                    "map_random": True,
                    "treasure_count": 10,
                    "buff_count": 2,
                    "monster_interval": 400,
                    "monster_speedup": 650,
                    "max_step": 900,
                }
            )
        else:
            env_conf.update(
                {
                    "map": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    "map_random": True,
                    "treasure_count": 10,
                    "buff_count": 2,
                    "monster_interval": -1,
                    "monster_speedup": -1,
                    "max_step": 1000,
                }
            )
        return usr_conf, stage_name

    def _finalize_episode(self, env_obs, terminated, truncated, total_reward, step):
        env_info = env_obs.get("observation", {}).get("env_info", {})
        summary = self.agent.get_episode_summary()

        total_score = _safe_float(env_info.get("total_score", summary.get("total_score", 0.0)), 0.0)
        treasure_count = _safe_int(
            env_info.get("treasures_collected", summary.get("treasure_count", 0)),
            0,
        )
        buff_count = _safe_int(env_info.get("collected_buff", summary.get("buff_count", 0)), 0)
        flash_count = _safe_int(env_info.get("flash_count", summary.get("flash_count", 0)), 0)

        if terminated:
            final_reward = Config.FAIL_PENALTY
            result_str = "FAIL"
            fail_rate = 1.0
            result = 0.0
        else:
            final_reward = Config.COMPLETE_BONUS
            result_str = "COMPLETE" if truncated else "DONE"
            fail_rate = 0.0
            result = 1.0

        episode_monitor = {
            "reward": round(total_reward + final_reward, 4),
            "episode_steps": step,
            "total_score": round(total_score, 4),
            "treasure_count": treasure_count,
            "buff_count": buff_count,
            "flash_count": flash_count,
            "fail_rate": fail_rate,
            "result": result,
            "avg_min_monster_dist": round(_safe_float(summary.get("avg_min_monster_dist", 0.0), 0.0), 4),
            "stuck_count": _safe_int(summary.get("stuck_count", 0), 0),
        }
        self._update_curriculum_history(episode_monitor)
        return final_reward, result_str, episode_monitor

    def _select_curriculum_stage(self, episode_cnt):
        avg_steps, avg_score, avg_treasure = self._curriculum_stats()

        if (
            episode_cnt < Config.STAGE_A_MIN_EPISODES
            or avg_steps < Config.STAGE_A_TARGET_STEPS
            or avg_score < Config.STAGE_A_TARGET_SCORE
            or avg_treasure < Config.STAGE_A_TARGET_TREASURE
        ):
            return "stage_a_survive"

        if (
            episode_cnt < Config.STAGE_B_MIN_EPISODES
            or avg_steps < Config.STAGE_B_TARGET_STEPS
            or avg_score < Config.STAGE_B_TARGET_SCORE
            or avg_treasure < Config.STAGE_B_TARGET_TREASURE
        ):
            return "stage_b_standard"

        return "stage_c_generalize"

    def _curriculum_stats(self):
        avg_steps = float(np.mean(self.recent_steps)) if self.recent_steps else 0.0
        avg_score = float(np.mean(self.recent_scores)) if self.recent_scores else 0.0
        avg_treasure = float(np.mean(self.recent_treasures)) if self.recent_treasures else 0.0
        return avg_steps, avg_score, avg_treasure

    def _update_curriculum_history(self, episode_monitor):
        self.recent_steps.append(float(episode_monitor.get("episode_steps", 0.0)))
        self.recent_scores.append(float(episode_monitor.get("total_score", 0.0)))
        self.recent_treasures.append(float(episode_monitor.get("treasure_count", 0.0)))


def _safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default
