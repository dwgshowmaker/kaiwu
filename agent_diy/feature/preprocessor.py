#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Feature engineering and reward shaping for the DIY Gorge Chase agent.
"""

import math
from collections import deque

import numpy as np

from agent_diy.conf.conf import Config


TREASURE_SUBTYPE = 1
BUFF_SUBTYPE = 2
RELATIVE_DIRECTION_TO_VEC = {
    0: (0.0, 0.0),
    1: (1.0, 0.0),
    2: (1.0, -1.0),
    3: (0.0, -1.0),
    4: (-1.0, -1.0),
    5: (-1.0, 0.0),
    6: (-1.0, 1.0),
    7: (0.0, 1.0),
    8: (1.0, 1.0),
}
DISTANCE_BUCKET_CENTER = {
    0: 15.0,
    1: 45.0,
    2: 75.0,
    3: 105.0,
    4: 135.0,
    5: 165.0,
}


def _safe_get(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _safe_get_many(obj, keys, default=None):
    for key in keys:
        value = _safe_get(obj, key, None)
        if value is not None:
            return value
    return default


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _clip01(value):
    return float(np.clip(value, 0.0, 1.0))


def _norm_signed(value, scale):
    if scale <= 0:
        return 0.0
    return float(np.clip(float(value) / float(scale), -1.0, 1.0))


def _distance(pos_a, pos_b):
    if pos_a is None or pos_b is None:
        return Config.MONSTER_DIST_MAX
    return float(math.hypot(float(pos_a[0]) - float(pos_b[0]), float(pos_a[1]) - float(pos_b[1])))


def _extract_pos(obj):
    if obj is None:
        return None
    pos = _safe_get(obj, "pos", None)
    if pos is not None:
        return (
            float(_safe_get(pos, "x", 0.0)),
            float(_safe_get(pos, "z", 0.0)),
        )
    x = _safe_get(obj, "x", None)
    z = _safe_get(obj, "z", None)
    if x is None or z is None:
        return None
    return float(x), float(z)


def _direction_and_dist_from_monster(monster):
    relative_direction = int(_safe_get(monster, "hero_relative_direction", 0) or 0)
    bucket = int(_safe_get(monster, "hero_l2_distance", 5) or 5)
    direction = RELATIVE_DIRECTION_TO_VEC.get(relative_direction, (0.0, 0.0))
    distance = DISTANCE_BUCKET_CENTER.get(bucket, DISTANCE_BUCKET_CENTER[5])
    return direction, distance


class Preprocessor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.step_no = 0
        self.max_step = int(Config.MAX_EPISODE_STEP)
        self.last_pos = None
        self.last_min_monster_dist = Config.MONSTER_DIST_MAX
        self.last_treasure_count = 0
        self.last_buff_count = 0
        self.last_total_score = 0.0
        self.last_flash_step = None
        self.last_buff_pick_step = None
        self.recent_flash_flags = deque(maxlen=Config.RECENT_FLASH_WINDOW)
        self.stuck_count = 0
        self.total_flash_count = 0
        self.min_monster_dist_sum = 0.0
        self.min_monster_dist_count = 0
        self.episode_reward = 0.0
        self.policy_context = None

    def feature_process(self, env_obs, last_action):
        observation = _safe_get(env_obs, "observation", {}) or {}
        frame_state = _safe_get(observation, "frame_state", {}) or {}
        env_info = _safe_get(observation, "env_info", {}) or {}
        map_info = _safe_get(observation, "map_info", None)

        legal_action = self._parse_legal_action(observation)
        step_no = int(
            _safe_get_many(
                observation,
                ["step_no"],
                _safe_get_many(env_info, ["step_no", "finished_steps"], 0),
            )
            or 0
        )
        self.step_no = step_no
        self.max_step = int(_safe_get(env_info, "max_step", self.max_step) or self.max_step)

        if last_action is not None and last_action >= 0:
            flashed = 1.0 if last_action >= 8 else 0.0
            self.recent_flash_flags.append(flashed)
            if flashed > 0:
                self.last_flash_step = step_no
                self.total_flash_count += 1

        hero = self._extract_hero(frame_state, env_info)
        hero_pos = _extract_pos(hero) or _extract_pos(env_info) or (0.0, 0.0)

        treasure_count = int(
            _safe_get_many(hero, ["treasure_collected_count"], _safe_get_many(env_info, ["treasures_collected"], 0)) or 0
        )
        buff_count = int(_safe_get(env_info, "collected_buff", self.last_buff_count) or self.last_buff_count)
        total_score = float(_safe_get(env_info, "total_score", self.last_total_score) or self.last_total_score)

        if buff_count > self.last_buff_count:
            self.last_buff_pick_step = step_no

        buff_remaining = self._estimate_buff_remaining(step_no)
        has_buff = float(buff_remaining > 0.0)

        monster_features, monster_positions, monster_distances, min_monster_dist = self._build_monster_features(
            frame_state, hero_pos, env_info
        )
        treasure_candidates = self._collect_targets(
            frame_state=frame_state,
            hero_pos=hero_pos,
            monster_positions=monster_positions,
            min_monster_dist=min_monster_dist,
            has_buff=has_buff,
            step_no=step_no,
            target_type=TREASURE_SUBTYPE,
        )
        buff_candidates = self._collect_targets(
            frame_state=frame_state,
            hero_pos=hero_pos,
            monster_positions=monster_positions,
            min_monster_dist=min_monster_dist,
            has_buff=has_buff,
            step_no=step_no,
            target_type=BUFF_SUBTYPE,
        )
        treasure_features = self._build_target_features(treasure_candidates)
        buff_features = self._build_target_features(buff_candidates)
        local_route_features, route_summary = self._build_local_route_features(map_info)
        hero_features = self._build_hero_features(
            hero_pos=hero_pos,
            env_info=env_info,
            legal_action=legal_action,
            step_no=step_no,
            last_action=last_action,
        )
        progress_features = self._build_progress_features(env_info, step_no, treasure_count, buff_count)

        feature = np.concatenate(
            [
                hero_features,
                monster_features,
                treasure_features,
                buff_features,
                local_route_features,
                progress_features,
            ]
        ).astype(np.float32)

        reward, reward_components = self._build_reward(
            hero_pos=hero_pos,
            min_monster_dist=min_monster_dist,
            treasure_count=treasure_count,
            buff_count=buff_count,
            total_score=total_score,
            last_action=last_action,
        )

        self.last_pos = hero_pos
        self.last_min_monster_dist = min_monster_dist
        self.last_treasure_count = treasure_count
        self.last_buff_count = buff_count
        self.last_total_score = total_score
        self.min_monster_dist_sum += min_monster_dist
        self.min_monster_dist_count += 1
        self.episode_reward += float(reward[0])
        self.policy_context = {
            "hero_pos": hero_pos,
            "step_no": step_no,
            "max_step": self.max_step,
            "legal_action": np.asarray(legal_action, dtype=np.float32).copy(),
            "map_info": None if map_info is None else np.asarray(map_info, dtype=np.int8),
            "monster_positions": [tuple(position) for position in monster_positions],
            "monster_distances": [float(dist) for dist in monster_distances],
            "min_monster_dist": float(min_monster_dist),
            "treasures": treasure_candidates,
            "buffs": buff_candidates,
            "routes": route_summary,
            "flash_ready": bool(float(np.sum(legal_action[8:])) > 0.0),
            "flash_remaining": float(self._estimate_flash_remaining(step_no, float(_safe_get_many(env_info, ["flash_cooldown", "talent_cooldown"], Config.FLASH_COOLDOWN_MAX) or Config.FLASH_COOLDOWN_MAX))),
            "buff_remaining": float(buff_remaining),
            "move_speed": 2 if buff_remaining > 0.0 else 1,
            "treasure_count": treasure_count,
            "buff_count": buff_count,
            "stuck_count": self.stuck_count,
            "last_action": last_action,
        }

        info = {
            "reward_components": reward_components,
            "step_no": step_no,
            "min_monster_dist": min_monster_dist,
            "treasure_count": treasure_count,
            "buff_count": buff_count,
            "flash_count": self.total_flash_count,
            "stuck_count": self.stuck_count,
            "total_score": total_score,
        }
        return feature, legal_action, reward, info

    def terminal_reward(self, env_obs):
        terminated = bool(_safe_get(env_obs, "terminated", False))
        truncated = bool(_safe_get(env_obs, "truncated", False))
        if terminated:
            return np.asarray([Config.FAIL_PENALTY], dtype=np.float32), "FAIL"

        if truncated:
            observation = _safe_get(env_obs, "observation", {}) or {}
            env_info = _safe_get(observation, "env_info", {}) or {}
            extra_info = _safe_get(env_obs, "extra_info", {}) or {}
            step_no = int(
                _safe_get_many(
                    observation,
                    ["step_no"],
                    _safe_get_many(env_info, ["finished_steps", "step_no"], self.step_no),
                )
                or self.step_no
            )
            result_code = _safe_get(extra_info, "result_code", None)
            max_step = int(_safe_get(env_info, "max_step", self.max_step) or self.max_step)
            if step_no >= max_step or (result_code is not None and int(result_code) == 0):
                return np.asarray([Config.COMPLETE_BONUS], dtype=np.float32), "COMPLETE"
        return np.zeros(1, dtype=np.float32), "TRUNCATED"

    def get_episode_metrics(self, env_obs):
        observation = _safe_get(env_obs, "observation", {}) or {}
        env_info = _safe_get(observation, "env_info", {}) or {}
        frame_state = _safe_get(observation, "frame_state", {}) or {}
        hero = self._extract_hero(frame_state, env_info)

        treasure_count = int(
            _safe_get_many(hero, ["treasure_collected_count"], _safe_get_many(env_info, ["treasures_collected"], 0)) or 0
        )
        buff_count = int(_safe_get(env_info, "collected_buff", self.last_buff_count) or self.last_buff_count)
        total_score = float(_safe_get(env_info, "total_score", self.last_total_score) or self.last_total_score)
        flash_count = int(_safe_get(env_info, "flash_count", self.total_flash_count) or self.total_flash_count)
        step_no = int(
            _safe_get_many(
                observation,
                ["step_no"],
                _safe_get_many(env_info, ["finished_steps", "step_no"], self.step_no),
            )
            or self.step_no
        )

        avg_min_monster_dist = self.min_monster_dist_sum / max(self.min_monster_dist_count, 1)
        return {
            "episode_steps": step_no,
            "total_score": total_score,
            "treasure_count": treasure_count,
            "buff_count": buff_count,
            "flash_count": flash_count,
            "avg_min_monster_dist": round(avg_min_monster_dist, 4),
            "stuck_count": self.stuck_count,
        }

    def _extract_hero(self, frame_state, env_info):
        heroes = _as_list(_safe_get(frame_state, "heroes", None))
        if heroes:
            return heroes[0]
        hero_like = _safe_get(frame_state, "hero", None)
        if hero_like is not None:
            return hero_like
        return env_info

    def _parse_legal_action(self, observation):
        raw_legal_action = _safe_get_many(observation, ["legal_action", "legal_act"], None)
        legal_action = np.zeros(Config.ACTION_NUM, dtype=np.float32)

        if raw_legal_action is None:
            legal_action[:8] = 1.0
            return legal_action

        if isinstance(raw_legal_action, np.ndarray):
            raw_legal_action = raw_legal_action.tolist()

        if isinstance(raw_legal_action, (list, tuple)) and raw_legal_action:
            first_item = raw_legal_action[0]
            if isinstance(first_item, (bool, np.bool_, int, np.integer, float, np.floating)):
                raw_array = np.asarray(raw_legal_action, dtype=np.float32).reshape(-1)
                legal_action[: min(Config.ACTION_NUM, raw_array.size)] = raw_array[: Config.ACTION_NUM]
            else:
                valid_actions = {int(action) for action in raw_legal_action if 0 <= int(action) < Config.ACTION_NUM}
                for action in valid_actions:
                    legal_action[action] = 1.0

        if legal_action.sum() <= 0.0:
            legal_action[:8] = 1.0
        return legal_action

    def _build_hero_features(self, hero_pos, env_info, legal_action, step_no, last_action):
        flash_total_cd = float(
            _safe_get_many(env_info, ["flash_cooldown", "talent_cooldown"], Config.FLASH_COOLDOWN_MAX)
            or Config.FLASH_COOLDOWN_MAX
        )
        flash_remaining = self._estimate_flash_remaining(step_no, flash_total_cd)
        buff_remaining = self._estimate_buff_remaining(step_no)
        flash_ready = 1.0 if float(legal_action[8:].sum()) > 0.0 else float(flash_remaining <= 0.0)
        recent_flash_ratio = (
            float(sum(self.recent_flash_flags)) / float(len(self.recent_flash_flags))
            if self.recent_flash_flags
            else 0.0
        )

        return np.asarray(
            [
                _clip01(hero_pos[0] / Config.MAP_SIZE),
                _clip01(hero_pos[1] / Config.MAP_SIZE),
                _clip01(flash_remaining / max(flash_total_cd, 1.0)),
                flash_ready,
                _clip01(buff_remaining / Config.BUFF_DURATION),
                float(buff_remaining > 0.0),
                _clip01(step_no / max(float(self.max_step), 1.0)),
                _clip01((self.max_step - step_no) / max(float(self.max_step), 1.0)),
                float(last_action is not None and last_action >= 8),
                _clip01(recent_flash_ratio),
            ],
            dtype=np.float32,
        )

    def _build_monster_features(self, frame_state, hero_pos, env_info):
        monsters = _as_list(_safe_get(frame_state, "monsters", None))
        monsters = sorted(monsters, key=lambda monster: int(_safe_get(monster, "monster_id", 0) or 0))

        features = []
        monster_positions = []
        monster_distances = []
        min_monster_dist = Config.MONSTER_DIST_MAX

        for index in range(2):
            if index >= len(monsters):
                features.append(np.zeros(8, dtype=np.float32))
                continue

            monster = monsters[index]
            monster_pos = _extract_pos(monster)
            if monster_pos is not None:
                rel_x = monster_pos[0] - hero_pos[0]
                rel_z = monster_pos[1] - hero_pos[1]
                dist = _distance(hero_pos, monster_pos)
            else:
                direction, dist = _direction_and_dist_from_monster(monster)
                rel_x = direction[0] * dist
                rel_z = direction[1] * dist
                monster_pos = (hero_pos[0] + rel_x, hero_pos[1] + rel_z)

            speed = float(_safe_get(monster, "speed", _safe_get(env_info, "monster_speed", 1.0)) or 1.0)
            min_monster_dist = min(min_monster_dist, dist)
            monster_positions.append(monster_pos)
            monster_distances.append(dist)

            features.append(
                np.asarray(
                    [
                        1.0,
                        _norm_signed(rel_x, Config.MAP_SIZE),
                        _norm_signed(rel_z, Config.MAP_SIZE),
                        _clip01(dist / Config.MONSTER_DIST_MAX),
                        float(np.sign(rel_x)),
                        float(np.sign(rel_z)),
                        _clip01(speed / Config.MONSTER_SPEED_MAX),
                        float(dist <= Config.THREAT_DISTANCE),
                    ],
                    dtype=np.float32,
                )
            )

        return np.concatenate(features).astype(np.float32), monster_positions, monster_distances, min_monster_dist

    def _collect_targets(self, frame_state, hero_pos, monster_positions, min_monster_dist, has_buff, step_no, target_type):
        organs = _as_list(_safe_get(frame_state, "organs", None))
        candidates = []
        step_norm = _clip01(step_no / max(float(self.max_step), 1.0))

        for organ in organs:
            if int(_safe_get(organ, "sub_type", 0) or 0) != target_type:
                continue
            if int(_safe_get(organ, "status", 1) or 1) != 1:
                continue

            organ_pos = _extract_pos(organ)
            if organ_pos is None:
                continue

            rel_x = organ_pos[0] - hero_pos[0]
            rel_z = organ_pos[1] - hero_pos[1]
            dist = _distance(hero_pos, organ_pos)
            nearest_monster_dist = (
                min(_distance(organ_pos, monster_pos) for monster_pos in monster_positions)
                if monster_positions
                else Config.MONSTER_DIST_MAX
            )

            if target_type == TREASURE_SUBTYPE:
                safety = _clip01((nearest_monster_dist - dist + 10.0) / 25.0)
                priority = _clip01(0.6 * (1.0 - min(dist / 40.0, 1.0)) + 0.4 * safety)
                value = priority
                rank = -priority + dist / 200.0
            else:
                danger = _clip01((Config.THREAT_DISTANCE + 2.0 - min_monster_dist) / (Config.THREAT_DISTANCE + 2.0))
                worth = _clip01(0.45 * (1.0 - has_buff) + 0.35 * danger + 0.20 * (1.0 - step_norm))
                value = worth
                rank = -worth + dist / 200.0

            candidates.append(
                {
                    "rank": rank,
                    "pos": (float(organ_pos[0]), float(organ_pos[1])),
                    "rel_x": float(rel_x),
                    "rel_z": float(rel_z),
                    "dist": float(dist),
                    "value": float(value),
                    "nearest_monster_dist": float(nearest_monster_dist),
                }
            )

        candidates.sort(key=lambda item: (item["rank"], item["dist"]))
        return candidates

    def _build_target_features(self, candidates):
        selected = []
        for candidate in candidates[:2]:
            selected.append(
                np.asarray(
                    [
                        1.0,
                        _norm_signed(candidate["rel_x"], Config.MAP_SIZE),
                        _norm_signed(candidate["rel_z"], Config.MAP_SIZE),
                        _clip01(candidate["dist"] / Config.MONSTER_DIST_MAX),
                        candidate["value"],
                    ],
                    dtype=np.float32,
                )
            )
        while len(selected) < 2:
            selected.append(np.zeros(5, dtype=np.float32))
        return np.concatenate(selected).astype(np.float32)

    def _build_local_route_features(self, map_info):
        if map_info is None:
            return np.zeros(24, dtype=np.float32), self._empty_route_summary()

        map_array = np.asarray(map_info, dtype=np.int8)
        if map_array.ndim != 2 or map_array.size == 0:
            return np.zeros(24, dtype=np.float32), self._empty_route_summary()

        center_row = map_array.shape[0] // 2
        center_col = map_array.shape[1] // 2
        features = []
        route_summary = []

        for delta_x, delta_z in Config.MOVE_DIRS:
            corridor_len = 0
            for step in range(1, Config.LOCAL_ROUTE_MAX_LEN + 1):
                if self._is_direction_passable(map_array, center_row, center_col, delta_x, delta_z, step):
                    corridor_len = step
                else:
                    break
            features.extend(
                [
                    float(corridor_len >= 1),
                    float(corridor_len >= 2),
                    _clip01(corridor_len / Config.LOCAL_ROUTE_MAX_LEN),
                ]
            )
            route_summary.append(
                {
                    "step_1": float(corridor_len >= 1),
                    "step_2": float(corridor_len >= 2),
                    "corridor": _clip01(corridor_len / Config.LOCAL_ROUTE_MAX_LEN),
                    "passable_len": int(corridor_len),
                }
            )

        return np.asarray(features, dtype=np.float32), route_summary

    def _is_direction_passable(self, map_array, center_row, center_col, delta_x, delta_z, step):
        row = center_row + delta_z * step
        col = center_col + delta_x * step
        if not self._cell_passable(map_array, row, col):
            return False

        if delta_x != 0 and delta_z != 0:
            prev_row = center_row + delta_z * (step - 1)
            prev_col = center_col + delta_x * (step - 1)
            horizontal_ok = self._cell_passable(map_array, prev_row, prev_col + delta_x)
            vertical_ok = self._cell_passable(map_array, prev_row + delta_z, prev_col)
            if not (horizontal_ok or vertical_ok):
                return False
        return True

    def _cell_passable(self, map_array, row, col):
        if row < 0 or row >= map_array.shape[0] or col < 0 or col >= map_array.shape[1]:
            return False
        return bool(map_array[row, col] != 0)

    def _build_progress_features(self, env_info, step_no, treasure_count, buff_count):
        monster_interval = int(_safe_get(env_info, "monster_interval", Config.DEFAULT_MONSTER_INTERVAL) or Config.DEFAULT_MONSTER_INTERVAL)
        monster_speedup = int(_safe_get(env_info, "monster_speedup", Config.DEFAULT_MONSTER_SPEEDUP) or Config.DEFAULT_MONSTER_SPEEDUP)
        if monster_interval < 0:
            monster_interval = Config.DEFAULT_MONSTER_INTERVAL
        if monster_speedup < 0:
            monster_speedup = Config.DEFAULT_MONSTER_SPEEDUP

        total_treasure = int(_safe_get(env_info, "total_treasure", 10) or 10)
        total_buff = int(_safe_get(env_info, "total_buff", 2) or 2)

        return np.asarray(
            [
                _clip01(max(monster_interval - step_no, 0) / max(float(self.max_step), 1.0)),
                _clip01(max(monster_speedup - step_no, 0) / max(float(self.max_step), 1.0)),
                _clip01(treasure_count / max(float(total_treasure), 1.0)),
                _clip01(buff_count / max(float(total_buff), 1.0)),
            ],
            dtype=np.float32,
        )

    def _build_reward(self, hero_pos, min_monster_dist, treasure_count, buff_count, total_score, last_action):
        if self.last_pos is None:
            return np.zeros(1, dtype=np.float32), {}

        reward = 0.0
        reward_components = {}

        reward += Config.SURVIVE_REWARD
        reward_components["survive_reward"] = Config.SURVIVE_REWARD

        min_dist_delta = float(min_monster_dist - self.last_min_monster_dist)
        danger_escape_reward = Config.DANGER_DELTA_REWARD_SCALE * float(np.clip(min_dist_delta, -3.0, 3.0))
        reward += danger_escape_reward
        reward_components["danger_escape_reward"] = danger_escape_reward

        treasure_delta = max(0, treasure_count - self.last_treasure_count)
        treasure_reward = treasure_delta * Config.TREASURE_REWARD
        reward += treasure_reward
        reward_components["treasure_reward"] = treasure_reward

        buff_delta = max(0, buff_count - self.last_buff_count)
        danger_scale = 1.0 + 0.5 * float(self.last_min_monster_dist <= Config.THREAT_DISTANCE)
        buff_reward = buff_delta * Config.BUFF_REWARD * danger_scale
        reward += buff_reward
        reward_components["buff_reward"] = buff_reward

        movement_dist = _distance(hero_pos, self.last_pos)
        if movement_dist < 0.5 and treasure_delta == 0 and buff_delta == 0 and abs(total_score - self.last_total_score) < 1.0e-6:
            self.stuck_count += 1
        else:
            self.stuck_count = max(0, self.stuck_count - 1)

        stuck_penalty = 0.0
        if self.stuck_count >= Config.STUCK_THRESHOLD:
            stuck_penalty = -min(Config.STUCK_PENALTY * self.stuck_count, 0.12)
            reward += stuck_penalty
        reward_components["stuck_penalty"] = stuck_penalty

        threat_penalty = 0.0
        if min_monster_dist <= 3.0:
            threat_penalty = Config.THREAT_PENALTY_DANGER
        elif min_monster_dist <= 5.0:
            threat_penalty = Config.THREAT_PENALTY_NEAR * (6.0 - min_monster_dist)
        reward += threat_penalty
        reward_components["threat_penalty"] = threat_penalty

        flash_reward = 0.0
        if last_action is not None and last_action >= 8:
            if treasure_delta > 0 or buff_delta > 0 or min_dist_delta >= Config.FLASH_ESCAPE_THRESHOLD:
                flash_reward = Config.GOOD_FLASH_REWARD + 0.15 * treasure_delta + 0.10 * buff_delta
            else:
                flash_reward = Config.BAD_FLASH_PENALTY
            reward += flash_reward
        reward_components["flash_reward"] = flash_reward

        return np.asarray([reward], dtype=np.float32), reward_components

    def _estimate_flash_remaining(self, step_no, flash_total_cd):
        if self.last_flash_step is None:
            return 0.0
        elapsed = max(0, step_no - self.last_flash_step)
        return max(0.0, float(flash_total_cd) - float(elapsed))

    def _estimate_buff_remaining(self, step_no):
        if self.last_buff_pick_step is None:
            return 0.0
        elapsed = max(0, step_no - self.last_buff_pick_step)
        return max(0.0, Config.BUFF_DURATION - float(elapsed))

    def _empty_route_summary(self):
        return [
            {
                "step_1": 0.0,
                "step_2": 0.0,
                "corridor": 0.0,
                "passable_len": 0,
            }
            for _ in range(8)
        ]
