#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Training workflow for the DIY PPO baseline.
DIY PPO 基线训练工作流。
"""


import os
import time
import numpy as np

from agent_diy.feature.definition import SampleData, sample_process
from tools.metrics_utils import get_training_metrics
from tools.train_env_conf_validate import read_usr_conf
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    last_save_model_time = time.time()
    env = envs[0]
    agent = agents[0]

    usr_conf = read_usr_conf("agent_diy/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        logger.error("usr_conf is None, please check agent_diy/conf/train_env_conf.toml")
        return

    episode_runner = EpisodeRunner(
        env=env,
        agent=agent,
        usr_conf=usr_conf,
        logger=logger,
        monitor=monitor,
    )

    while True:
        for g_data in episode_runner.run_episodes():
            agent.send_sample_data(g_data)
            g_data.clear()

            now = time.time()
            if now - last_save_model_time >= 1800:
                agent.save_model()
                last_save_model_time = now


class EpisodeRunner:
    def __init__(self, env, agent, usr_conf, logger, monitor):
        self.env = env
        self.agent = agent
        self.usr_conf = usr_conf
        self.logger = logger
        self.monitor = monitor
        self.episode_cnt = 0
        self.last_report_monitor_time = 0
        self.last_get_training_metrics_time = 0

    def run_episodes(self):
        """Run episodes and yield collected training samples."""
        while True:
            now = time.time()
            if now - self.last_get_training_metrics_time >= 60:
                training_metrics = get_training_metrics()
                self.last_get_training_metrics_time = now
                if training_metrics is not None:
                    self.logger.info(f"training_metrics is {training_metrics}")

            env_obs = self.env.reset(self.usr_conf)

            if handle_disaster_recovery(env_obs, self.logger):
                continue

            self.agent.reset(env_obs)
            self.agent.load_model(id="latest")

            obs_data, remain_info = self.agent.observation_process(env_obs)

            collector = []
            self.episode_cnt += 1
            done = False
            step = 0
            total_reward = 0.0
            min_dist_sum = 0.0
            min_dist_count = 0
            safe_prior_count = 0
            safe_action_count = 0
            safe_flash_prior_count = 0
            safe_flash_step_count = 0
            safe_action_margin_sum = 0.0
            safe_trap_risk_sum = 0.0
            state_eval_count = 0
            state_potential_sum = 0.0
            safety_potential_sum = 0.0
            resource_potential_sum = 0.0
            flash_potential_sum = 0.0

            self.logger.info(f"Episode {self.episode_cnt} start")

            while not done:
                safe_is_flash = int(getattr(obs_data, "safe_is_flash", 0) or 0)
                safe_flash_step_count += safe_is_flash
                safe_action_margin_sum += float(getattr(obs_data, "safe_action_margin", 0.0) or 0.0)
                safe_trap_risk_sum += float(getattr(obs_data, "safe_trap_risk", 0.0) or 0.0)
                state_eval_count += 1

                act_data_list = self.agent.predict(list_obs_data=[obs_data])
                if not act_data_list:
                    self.logger.error(
                        f"predict returned empty result at episode {self.episode_cnt}, restart episode"
                    )
                    break

                act_data = act_data_list[0]
                safe_prior_count += int(getattr(act_data, "safe_prior_used", 0) or 0)
                safe_action_count += int(getattr(act_data, "safe_action_used", 0) or 0)
                safe_flash_prior_count += safe_is_flash * int(getattr(act_data, "safe_prior_used", 0) or 0)
                act = self.agent.action_process(act_data)

                env_reward, env_obs = self.env.step(act)

                if handle_disaster_recovery(env_obs, self.logger):
                    break

                terminated = env_obs["terminated"]
                truncated = env_obs["truncated"]
                step += 1
                done = terminated or truncated

                _obs_data, _remain_info = self.agent.observation_process(env_obs)

                reward = np.array(_remain_info.get("reward", [0.0]), dtype=np.float32)
                total_reward += float(reward[0])
                state_potential_sum += float(_remain_info.get("state_potential", 0.0) or 0.0)
                safety_potential_sum += float(_remain_info.get("safety_potential", 0.0) or 0.0)
                resource_potential_sum += float(_remain_info.get("resource_potential", 0.0) or 0.0)
                flash_potential_sum += float(_remain_info.get("flash_potential", 0.0) or 0.0)
                if "min_monster_dist" in _remain_info:
                    min_dist_sum += float(_remain_info["min_monster_dist"])
                    min_dist_count += 1

                final_reward = np.zeros(1, dtype=np.float32)
                if done:
                    env_info = env_obs["observation"]["env_info"]
                    total_score = env_info.get("total_score", 0)

                    if terminated:
                        final_reward[0] = -10.0
                        result_str = "FAIL"
                    else:
                        final_reward[0] = 10.0
                        result_str = "WIN"

                    self.logger.info(
                        f"[GAMEOVER] episode:{self.episode_cnt} steps:{step} "
                        f"result:{result_str} sim_score:{total_score:.1f} "
                        f"total_reward:{total_reward:.3f} "
                        f"treasure:{env_info.get('treasures_collected', 0)} "
                        f"buff:{env_info.get('collected_buff', 0)} "
                        f"flash:{env_info.get('flash_count', 0)} "
                        f"stuck:{_remain_info.get('stuck_count', 0)} "
                        f"blocked:{_remain_info.get('blocked_count', 0)} "
                        f"danger:{float(_remain_info.get('danger_level', 0.0)):.3f} "
                        f"danger_steps:{_remain_info.get('danger_steps', 0)} "
                        f"near_death:{_remain_info.get('near_death_count', 0)} "
                        f"good_flash:{_remain_info.get('good_flash_count', 0)} "
                        f"bad_flash:{_remain_info.get('bad_flash_count', 0)} "
                        f"flash_trap:{_remain_info.get('flash_trap_count', 0)} "
                        f"flash_hold:{_remain_info.get('flash_hold_count', 0)} "
                        f"flash_gain:{float(_remain_info.get('flash_escape_gain', 0.0)):.3f} "
                        f"late_game:{_remain_info.get('late_game_steps', 0)} "
                        f"prep_speedup:{_remain_info.get('speedup_prep_steps', 0)} "
                        f"post_speedup:{_remain_info.get('post_speedup_steps', 0)} "
                        f"post_nobuff:{_remain_info.get('post_speedup_buffless_steps', 0)} "
                        f"post_unready:{_remain_info.get('post_speedup_unready_steps', 0)} "
                        f"prep_flash_ready:{_remain_info.get('prep_flash_ready_steps', 0)} "
                        f"prep_buff_active:{_remain_info.get('prep_buff_active_steps', 0)} "
                        f"buff_ready:{_remain_info.get('buff_ready_at_speedup', 0)} "
                        f"flash_ready:{_remain_info.get('flash_ready_at_speedup', 0)} "
                        f"loop:{_remain_info.get('loop_count', 0)} "
                        f"state_pot:{float(_remain_info.get('state_potential', 0.0)):.3f} "
                        f"safety_pot:{float(_remain_info.get('safety_potential', 0.0)):.3f} "
                        f"resource_pot:{float(_remain_info.get('resource_potential', 0.0)):.3f} "
                        f"flash_pot:{float(_remain_info.get('flash_potential', 0.0)):.3f} "
                        f"state_pot_avg:{state_potential_sum / max(1, state_eval_count):.3f} "
                        f"safety_pot_avg:{safety_potential_sum / max(1, state_eval_count):.3f} "
                        f"resource_pot_avg:{resource_potential_sum / max(1, state_eval_count):.3f} "
                        f"flash_pot_avg:{flash_potential_sum / max(1, state_eval_count):.3f} "
                        f"safe_prior:{safe_prior_count} "
                        f"safe_action:{safe_action_count} "
                        f"safe_flash_steps:{safe_flash_step_count} "
                        f"safe_flash_prior:{safe_flash_prior_count} "
                        f"safe_margin_avg:{safe_action_margin_sum / max(1, state_eval_count):.3f} "
                        f"safe_trap_avg:{safe_trap_risk_sum / max(1, state_eval_count):.3f}"
                    )

                frame = SampleData(
                    obs=np.array(obs_data.feature, dtype=np.float32),
                    legal_action=np.array(obs_data.legal_action, dtype=np.float32),
                    act=np.array([act_data.action[0]], dtype=np.float32),
                    reward=reward,
                    done=np.array([float(done)], dtype=np.float32),
                    reward_sum=np.zeros(1, dtype=np.float32),
                    value=np.array(act_data.value, dtype=np.float32).flatten()[:1],
                    next_value=np.zeros(1, dtype=np.float32),
                    advantage=np.zeros(1, dtype=np.float32),
                    prob=np.array(act_data.prob, dtype=np.float32),
                )
                collector.append(frame)

                if done:
                    if collector:
                        collector[-1].reward = collector[-1].reward + final_reward

                    now = time.time()
                    if now - self.last_report_monitor_time >= 60 and self.monitor:
                        avg_min_monster_dist = min_dist_sum / max(1, min_dist_count)
                        monitor_data = {
                            "reward": round(total_reward + float(final_reward[0]), 4),
                            "episode_steps": step,
                            "episode_cnt": self.episode_cnt,
                            "total_score": round(float(env_info.get("total_score", 0)), 4),
                            "treasure_count": int(env_info.get("treasures_collected", 0)),
                            "buff_count": int(env_info.get("collected_buff", 0)),
                            "flash_count": int(env_info.get("flash_count", 0)),
                            "fail_rate": 1.0 if terminated else 0.0,
                            "avg_min_monster_dist": round(avg_min_monster_dist, 4),
                            "stuck_count": int(_remain_info.get("stuck_count", 0)),
                            "loop_count": int(_remain_info.get("loop_count", 0)),
                            "blocked_count": int(_remain_info.get("blocked_count", 0)),
                            "danger_level": round(float(_remain_info.get("danger_level", 0.0)), 4),
                            "danger_steps": int(_remain_info.get("danger_steps", 0)),
                            "near_death_count": int(_remain_info.get("near_death_count", 0)),
                            "good_flash_count": int(_remain_info.get("good_flash_count", 0)),
                            "bad_flash_count": int(_remain_info.get("bad_flash_count", 0)),
                            "flash_trap_count": int(_remain_info.get("flash_trap_count", 0)),
                            "flash_hold_count": int(_remain_info.get("flash_hold_count", 0)),
                            "flash_escape_gain": round(float(_remain_info.get("flash_escape_gain", 0.0)), 4),
                            "late_game_steps": int(_remain_info.get("late_game_steps", 0)),
                            "speedup_prep_steps": int(_remain_info.get("speedup_prep_steps", 0)),
                            "post_speedup_steps": int(_remain_info.get("post_speedup_steps", 0)),
                            "post_speedup_buffless_steps": int(
                                _remain_info.get("post_speedup_buffless_steps", 0)
                            ),
                            "post_speedup_unready_steps": int(
                                _remain_info.get("post_speedup_unready_steps", 0)
                            ),
                            "prep_flash_ready_steps": int(_remain_info.get("prep_flash_ready_steps", 0)),
                            "prep_buff_active_steps": int(_remain_info.get("prep_buff_active_steps", 0)),
                            "buff_ready_at_speedup": int(_remain_info.get("buff_ready_at_speedup", 0)),
                            "flash_ready_at_speedup": int(_remain_info.get("flash_ready_at_speedup", 0)),
                            "state_potential": round(state_potential_sum / max(1, state_eval_count), 4),
                            "safety_potential": round(safety_potential_sum / max(1, state_eval_count), 4),
                            "resource_potential": round(resource_potential_sum / max(1, state_eval_count), 4),
                            "flash_potential": round(flash_potential_sum / max(1, state_eval_count), 4),
                            "safe_prior_count": safe_prior_count,
                            "safe_action_count": safe_action_count,
                            "safe_flash_step_count": safe_flash_step_count,
                            "safe_flash_prior_count": safe_flash_prior_count,
                            "safe_action_margin": round(safe_action_margin_sum / max(1, state_eval_count), 4),
                            "safe_trap_risk": round(safe_trap_risk_sum / max(1, state_eval_count), 4),
                        }
                        self.monitor.put_data({os.getpid(): monitor_data})
                        self.last_report_monitor_time = now

                    if collector:
                        collector = sample_process(collector)
                        yield collector
                    break

                obs_data = _obs_data
                remain_info = _remain_info
