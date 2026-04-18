#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Training workflow for the DIY Gorge Chase agent.
"""

import copy
import os
import time

import numpy as np

from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery
from tools.metrics_utils import get_training_metrics
from tools.train_env_conf_validate import read_usr_conf

from agent_diy.conf.conf import Config
from agent_diy.feature.definition import SampleData, sample_process


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    del args, kwargs
    env = envs[0]
    agent = agents[0]
    last_save_model_time = time.time()

    usr_conf = read_usr_conf("agent_diy/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        if logger is not None:
            logger.error("usr_conf is None, please check agent_diy/conf/train_env_conf.toml")
        return

    episode_runner = EpisodeRunner(env=env, agent=agent, usr_conf=usr_conf, logger=logger, monitor=monitor)

    while True:
        for g_data in episode_runner.run_episodes():
            if hasattr(agent, "send_sample_data"):
                agent.send_sample_data(g_data)
            else:
                agent.learn(g_data)
            g_data.clear()

            now = time.time()
            if now - last_save_model_time >= Config.SAVE_MODEL_INTERVAL_SEC:
                agent.save_model()
                last_save_model_time = now


class EpisodeRunner:
    def __init__(self, env, agent, usr_conf, logger=None, monitor=None):
        self.env = env
        self.agent = agent
        self.usr_conf = usr_conf
        self.logger = logger
        self.monitor = monitor
        self.episode_cnt = 0
        self.last_report_monitor_time = 0.0
        self.last_get_training_metrics_time = 0.0

    def run_episodes(self):
        while True:
            self._report_training_metrics()

            next_episode = self.episode_cnt + 1
            episode_usr_conf, stage_name = self._build_episode_usr_conf(next_episode)

            env_obs = self.env.reset(usr_conf=episode_usr_conf)
            if handle_disaster_recovery(env_obs, self.logger):
                continue

            self.agent.reset(env_obs)
            self.agent.load_model(id="latest")

            obs_data, _ = self.agent.observation_process(env_obs)
            collector = []
            self.episode_cnt = next_episode
            step = 0
            total_reward = 0.0
            aborted = False

            if self.logger is not None:
                self.logger.info(f"Episode {self.episode_cnt} start stage:{stage_name}")

            while True:
                act_data = self.agent.predict(list_obs_data=[obs_data])[0]
                act = self.agent.action_process(act_data)

                _, env_obs = self.env.step(act)
                if handle_disaster_recovery(env_obs, self.logger):
                    aborted = True
                    break

                terminated = bool(env_obs["terminated"])
                truncated = bool(env_obs["truncated"])
                done = terminated or truncated
                step += 1

                next_obs_data, next_remain_info = self.agent.observation_process(env_obs)
                reward = np.asarray(next_remain_info.get("reward", [0.0]), dtype=np.float32).reshape(Config.VALUE_NUM)
                total_reward += float(reward[0])

                final_reward = np.zeros(Config.VALUE_NUM, dtype=np.float32)
                result_str = "RUNNING"
                if done:
                    final_reward, result_str = self.agent.preprocessor.terminal_reward(env_obs)
                    total_reward += float(final_reward[0])

                frame = SampleData(
                    obs=np.asarray(obs_data.feature, dtype=np.float32),
                    legal_action=np.asarray(obs_data.legal_action, dtype=np.float32),
                    act=np.asarray([act_data.action[0]], dtype=np.float32),
                    reward=reward,
                    reward_sum=np.zeros(Config.VALUE_NUM, dtype=np.float32),
                    done=np.asarray([float(done)], dtype=np.float32),
                    value=np.asarray(act_data.value, dtype=np.float32).reshape(Config.VALUE_NUM),
                    next_value=np.zeros(Config.VALUE_NUM, dtype=np.float32),
                    advantage=np.zeros(Config.VALUE_NUM, dtype=np.float32),
                    prob=np.asarray(act_data.prob, dtype=np.float32).reshape(Config.ACTION_NUM),
                )
                collector.append(frame)

                if done:
                    if collector:
                        collector[-1].reward = collector[-1].reward + final_reward

                    episode_metrics = self.agent.preprocessor.get_episode_metrics(env_obs)
                    episode_metrics.update(
                        {
                            "reward": round(total_reward, 4),
                            "episode_cnt": self.episode_cnt,
                            "episode_steps": step,
                            "result": 0.0 if terminated else 1.0,
                            "fail_rate": 1.0 if terminated else 0.0,
                        }
                    )

                    if self.logger is not None:
                        self.logger.info(
                            f"[GAMEOVER] episode:{self.episode_cnt} stage:{stage_name} "
                            f"steps:{episode_metrics['episode_steps']} result:{result_str} "
                            f"score:{episode_metrics['total_score']:.1f} "
                            f"treasure:{episode_metrics['treasure_count']} "
                            f"buff:{episode_metrics['buff_count']} "
                            f"flash:{episode_metrics['flash_count']} "
                            f"avg_min_monster_dist:{episode_metrics['avg_min_monster_dist']} "
                            f"stuck:{episode_metrics['stuck_count']} "
                            f"reward:{episode_metrics['reward']}"
                        )

                    self._report_monitor(episode_metrics)

                    if collector:
                        collector = sample_process(collector)
                        yield collector
                    break

                obs_data = next_obs_data

            if aborted:
                continue

    def _build_episode_usr_conf(self, episode_index):
        stage = Config.CURRICULUM_STAGES[-1]
        for candidate in Config.CURRICULUM_STAGES:
            max_episode = candidate.get("max_episode")
            if max_episode is None or episode_index <= max_episode:
                stage = candidate
                break

        usr_conf = copy.deepcopy(self.usr_conf)
        target_conf = usr_conf.get("env_conf", usr_conf) if isinstance(usr_conf, dict) else usr_conf
        for key, value in stage["env_conf"].items():
            target_conf[key] = copy.deepcopy(value)
        return usr_conf, stage["name"]

    def _report_training_metrics(self):
        now = time.time()
        if now - self.last_get_training_metrics_time < Config.TRAINING_METRICS_INTERVAL_SEC:
            return
        training_metrics = get_training_metrics()
        self.last_get_training_metrics_time = now
        if training_metrics is not None and self.logger is not None:
            self.logger.info(f"training_metrics is {training_metrics}")

    def _report_monitor(self, monitor_data):
        now = time.time()
        if self.monitor is None or now - self.last_report_monitor_time < Config.MONITOR_REPORT_INTERVAL_SEC:
            return

        report = {
            "reward": monitor_data["reward"],
            "episode_steps": monitor_data["episode_steps"],
            "total_score": round(float(monitor_data["total_score"]), 4),
            "treasure_count": float(monitor_data["treasure_count"]),
            "buff_count": float(monitor_data["buff_count"]),
            "flash_count": float(monitor_data["flash_count"]),
            "result": float(monitor_data["result"]),
            "fail_rate": float(monitor_data["fail_rate"]),
            "avg_min_monster_dist": float(monitor_data["avg_min_monster_dist"]),
            "stuck_count": float(monitor_data["stuck_count"]),
        }
        self.monitor.put_data({os.getpid(): report})
        self.last_report_monitor_time = now
