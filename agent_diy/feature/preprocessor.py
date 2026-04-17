#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Feature preprocessor, action priors, and reward shaping for the DIY Gorge Chase agent.
"""

from collections import deque
import math

import numpy as np

from agent_diy.conf.conf import Config


MOVE_DIRS = [
    (1, 0),    # E
    (1, -1),   # NE
    (0, -1),   # N
    (-1, -1),  # NW
    (-1, 0),   # W
    (-1, 1),   # SW
    (0, 1),    # S
    (1, 1),    # SE
]

FLASH_DISTANCE = [10, 8, 10, 8, 10, 8, 10, 8]

DIR_CODE_TO_DELTA = {
    0: (0, 0),
    1: (1, 0),
    2: (1, -1),
    3: (0, -1),
    4: (-1, -1),
    5: (-1, 0),
    6: (-1, 1),
    7: (0, 1),
    8: (1, 1),
}

DIST_BUCKET_MID = [15.0, 45.0, 75.0, 105.0, 135.0, 165.0]


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


def _clip_norm(value, v_max, v_min=0.0):
    value = float(np.clip(value, v_min, v_max))
    denom = v_max - v_min
    if denom <= 1e-6:
        return 0.0
    return (value - v_min) / denom


def _signed_norm(value, max_abs):
    if max_abs <= 1e-6:
        return 0.0
    return float(np.clip(float(value) / float(max_abs), -1.0, 1.0))


def _l2(pos_a, pos_b):
    return math.hypot(float(pos_a[0]) - float(pos_b[0]), float(pos_a[1]) - float(pos_b[1]))


class Preprocessor:
    def __init__(self):
        self.runtime_mode = "eval"
        self.curriculum_stage = None
        self.curriculum_episode_idx = 0
        self.reset()

    def reset(self):
        self.step_no = 0
        self.max_step = Config.DEFAULT_MAX_STEP
        self.monster_interval = 300
        self.monster_speedup = 500

        self.last_pos = None
        self.last_min_monster_dist = Config.MAX_MONSTER_DIST
        self.last_treasure_count = 0
        self.last_buff_count = 0
        self.last_flash_count = 0
        self.last_total_score = 0.0
        self.last_target_treasure_key = None
        self.last_target_treasure_dist = None
        self.last_target_buff_key = None
        self.last_target_buff_dist = None

        self.stuck_count = 0
        self.stuck_streak = 0
        self.total_min_monster_dist = 0.0
        self.frame_count = 0
        self.visit_count_map = np.zeros((Config.MAP_SIZE, Config.MAP_SIZE), dtype=np.float32)

        self.known_treasures = {}
        self.known_buffs = {}
        self.recent_positions = deque(maxlen=Config.RECENT_POSITION_WINDOW)
        self.recent_actions = deque(maxlen=Config.RECENT_ACTION_WINDOW)
        self.latest_info = {}

    def set_runtime_mode(self, mode):
        self.runtime_mode = mode or "eval"

    def set_curriculum_context(self, stage_name=None, episode_idx=0):
        self.curriculum_stage = stage_name
        self.curriculum_episode_idx = episode_idx

    def feature_process(self, env_obs, last_action):
        observation = env_obs.get("observation", {})
        frame_state = observation.get("frame_state", {})
        env_info = observation.get("env_info", {})
        map_info = observation.get("map_info", [])

        self.step_no = _safe_int(observation.get("step_no", env_info.get("step_no", 0)), 0)
        self.max_step = _safe_int(env_info.get("max_step", self.max_step), self.max_step)
        self.monster_interval = _safe_int(env_info.get("monster_interval", self.monster_interval), self.monster_interval)
        self.monster_speedup = _safe_int(env_info.get("monster_speedup", self.monster_speedup), self.monster_speedup)

        hero = self._extract_hero(frame_state)
        hero_pos = self._extract_position(hero, fallback=self._extract_position(env_info))
        visit_before = self._update_visit_memory(hero_pos)

        self.recent_positions.append(hero_pos)
        if last_action is not None and last_action >= 0:
            self.recent_actions.append(int(last_action))

        legal_action = self._parse_legal_action(
            observation.get("legal_action", observation.get("legal_act", [1] * Config.ACTION_NUM))
        )

        treasure_count = _safe_int(
            hero.get("treasure_collected_count", env_info.get("treasures_collected", self.last_treasure_count)),
            self.last_treasure_count,
        )
        buff_count = _safe_int(env_info.get("collected_buff", self.last_buff_count), self.last_buff_count)
        flash_count = _safe_int(env_info.get("flash_count", self.last_flash_count), self.last_flash_count)
        total_score = _safe_float(env_info.get("total_score", self.last_total_score), self.last_total_score)

        flash_cd = _safe_int(hero.get("flash_cooldown", hero.get("talent_cooldown", 0)), 0)
        flash_cd_max = max(_safe_int(env_info.get("flash_cooldown", Config.MAX_FLASH_CD), Config.MAX_FLASH_CD), 1)
        buff_remain = _safe_int(hero.get("buff_remaining_time", hero.get("buff_remain", 0)), 0)
        has_buff = 1.0 if buff_remain > 0 else 0.0
        hero_speed = 2 if has_buff > 0 else 1

        monsters = self._extract_monsters(frame_state)
        monster_feature, monster_states, min_monster_dist, current_threat = self._build_monster_feature(monsters, hero_pos)

        organs = self._extract_organs(frame_state)
        self._update_known_items(organs, hero_pos)
        self._consume_nearby_items(self.known_treasures, hero_pos, treasure_count - self.last_treasure_count)
        self._consume_nearby_items(self.known_buffs, hero_pos, buff_count - self.last_buff_count)

        target_treasure = self._select_target_item(
            source=self.known_treasures,
            hero_pos=hero_pos,
            monster_states=monster_states,
            current_threat=current_threat,
            is_buff=False,
            has_buff=has_buff,
        )
        target_buff = self._select_target_item(
            source=self.known_buffs,
            hero_pos=hero_pos,
            monster_states=monster_states,
            current_threat=current_threat,
            is_buff=True,
            has_buff=has_buff,
        )

        hero_feature = self._build_hero_feature(
            hero_pos=hero_pos,
            flash_cd=flash_cd,
            flash_cd_max=flash_cd_max,
            buff_remain=buff_remain,
            has_buff=has_buff,
            last_action=last_action,
        )
        treasure_feature = self._build_item_feature(
            source=self.known_treasures,
            hero_pos=hero_pos,
            monster_states=monster_states,
            current_threat=current_threat,
            is_buff=False,
            has_buff=has_buff,
        )
        buff_feature = self._build_item_feature(
            source=self.known_buffs,
            hero_pos=hero_pos,
            monster_states=monster_states,
            current_threat=current_threat,
            is_buff=True,
            has_buff=has_buff,
        )

        local_route_feature, action_routes = self._build_local_route_feature(
            map_info=map_info,
            hero_pos=hero_pos,
            hero_speed=hero_speed,
        )
        action_bias = self._build_action_bias(
            legal_action=legal_action,
            action_routes=action_routes,
            hero_pos=hero_pos,
            monster_states=monster_states,
            min_monster_dist=min_monster_dist,
            current_threat=current_threat,
            target_treasure=target_treasure,
            target_buff=target_buff,
            has_buff=has_buff,
            flash_cd=flash_cd,
        )
        legal_action = self._apply_action_constraints(
            legal_action=legal_action,
            action_routes=action_routes,
            action_bias=action_bias,
            current_threat=current_threat,
            flash_cd=flash_cd,
            target_treasure=target_treasure,
            target_buff=target_buff,
        )

        progress_feature = self._build_progress_feature(env_info, treasure_count, buff_count)

        feature = np.concatenate(
            [
                hero_feature,
                monster_feature,
                treasure_feature,
                buff_feature,
                local_route_feature,
                progress_feature,
            ]
        ).astype(np.float32)

        if feature.shape[0] != Config.DIM_OF_OBSERVATION:
            raise ValueError(
                f"Feature length mismatch: got {feature.shape[0]}, expect {Config.DIM_OF_OBSERVATION}"
            )

        reward, reward_info = self._compute_reward(
            hero_pos=hero_pos,
            min_monster_dist=min_monster_dist,
            current_threat=current_threat,
            treasure_count=treasure_count,
            buff_count=buff_count,
            flash_count=flash_count,
            total_score=total_score,
            last_action=last_action,
            target_treasure=target_treasure,
            target_buff=target_buff,
            has_buff=has_buff,
            visit_before=visit_before,
        )

        self.total_min_monster_dist += min_monster_dist
        self.frame_count += 1

        self.last_pos = hero_pos
        self.last_min_monster_dist = min_monster_dist
        self.last_treasure_count = treasure_count
        self.last_buff_count = buff_count
        self.last_flash_count = flash_count
        self.last_total_score = total_score
        self.last_target_treasure_key = target_treasure["key"] if target_treasure else None
        self.last_target_treasure_dist = target_treasure["dist"] if target_treasure else None
        self.last_target_buff_key = target_buff["key"] if target_buff else None
        self.last_target_buff_dist = target_buff["dist"] if target_buff else None

        self.latest_info = {
            "reward": reward,
            "total_score": total_score,
            "treasure_count": treasure_count,
            "buff_count": buff_count,
            "flash_count": flash_count,
            "stuck_count": self.stuck_count,
            "avg_min_monster_dist": self.total_min_monster_dist / max(self.frame_count, 1),
        }
        self.latest_info.update(reward_info)

        return feature, legal_action.astype(np.float32), action_bias.astype(np.float32), [reward], self.latest_info

    def get_episode_summary(self):
        return {
            "stuck_count": self.stuck_count,
            "avg_min_monster_dist": self.total_min_monster_dist / max(self.frame_count, 1),
            "treasure_count": self.last_treasure_count,
            "buff_count": self.last_buff_count,
            "flash_count": self.last_flash_count,
            "total_score": self.last_total_score,
        }

    def _extract_hero(self, frame_state):
        heroes = frame_state.get("heroes", {})
        if isinstance(heroes, list):
            return heroes[0] if heroes else {}
        return heroes if isinstance(heroes, dict) else {}

    def _extract_monsters(self, frame_state):
        monsters = frame_state.get("monsters", [])
        if isinstance(monsters, dict):
            monsters = [monsters]
        return monsters if isinstance(monsters, list) else []

    def _extract_organs(self, frame_state):
        organs = frame_state.get("organs", [])
        if isinstance(organs, dict):
            organs = [organs]
        return organs if isinstance(organs, list) else []

    def _extract_position(self, source, fallback=(0, 0)):
        if not isinstance(source, dict):
            return fallback
        pos = source.get("pos", source)
        if not isinstance(pos, dict):
            return fallback
        if "x" not in pos or "z" not in pos:
            return fallback
        base_x = fallback[0] if isinstance(fallback, tuple) else 0
        base_z = fallback[1] if isinstance(fallback, tuple) else 0
        x = int(np.clip(_safe_int(pos.get("x", base_x), base_x), 0, Config.MAP_SIZE - 1))
        z = int(np.clip(_safe_int(pos.get("z", base_z), base_z), 0, Config.MAP_SIZE - 1))
        return (x, z)

    def _estimate_position_from_relative(self, entity, hero_pos):
        bucket = _safe_int(entity.get("hero_l2_distance", 5), 5)
        bucket = int(np.clip(bucket, 0, len(DIST_BUCKET_MID) - 1))
        dist = DIST_BUCKET_MID[bucket]
        dx, dz = DIR_CODE_TO_DELTA.get(_safe_int(entity.get("hero_relative_direction", 0), 0), (0, 0))
        return (
            int(np.clip(hero_pos[0] + dx * dist, 0, Config.MAP_SIZE - 1)),
            int(np.clip(hero_pos[1] + dz * dist, 0, Config.MAP_SIZE - 1)),
        )

    def _parse_legal_action(self, raw):
        legal_action = np.zeros((Config.ACTION_NUM,), dtype=np.float32)
        if isinstance(raw, (list, tuple, np.ndarray)) and len(raw) > 0:
            first = raw[0]
            if isinstance(first, (bool, np.bool_, int, np.integer, float, np.floating)) and len(raw) >= Config.ACTION_NUM:
                for i in range(Config.ACTION_NUM):
                    legal_action[i] = 1.0 if bool(raw[i]) else 0.0
            else:
                for idx in raw:
                    idx = _safe_int(idx, -1)
                    if 0 <= idx < Config.ACTION_NUM:
                        legal_action[idx] = 1.0
        if legal_action.sum() <= 0:
            legal_action[:8] = 1.0
        return legal_action

    def _update_visit_memory(self, hero_pos):
        x, z = hero_pos
        before = float(self.visit_count_map[z, x])
        self.visit_count_map[z, x] += 1.0
        return before

    def _update_known_items(self, organs, hero_pos):
        for organ in organs:
            sub_type = _safe_int(organ.get("sub_type", 0), 0)
            if sub_type not in (1, 2):
                continue
            pos = self._extract_position(organ, fallback=None)
            if pos is None:
                pos = self._estimate_position_from_relative(organ, hero_pos)
            key = str(organ.get("config_id", f"{sub_type}:{pos[0]}:{pos[1]}"))
            status = _safe_int(organ.get("status", 1), 1)
            source = self.known_treasures if sub_type == 1 else self.known_buffs
            if status == 1:
                source[key] = {"key": key, "pos": pos, "step_seen": self.step_no}
            else:
                source.pop(key, None)

        self._prune_known_items(self.known_treasures)
        self._prune_known_items(self.known_buffs)

    def _prune_known_items(self, source):
        stale_keys = []
        for key, value in source.items():
            if self.step_no - _safe_int(value.get("step_seen", self.step_no), self.step_no) > Config.ITEM_MEMORY_TIMEOUT:
                stale_keys.append(key)
        for key in stale_keys:
            source.pop(key, None)

    def _consume_nearby_items(self, source, hero_pos, delta_count):
        if delta_count <= 0 or not source:
            return
        candidates = sorted(source.items(), key=lambda kv: _l2(hero_pos, kv[1]["pos"]))
        remove_num = min(delta_count, len(candidates))
        for idx in range(remove_num):
            source.pop(candidates[idx][0], None)

    def _build_hero_feature(self, hero_pos, flash_cd, flash_cd_max, buff_remain, has_buff, last_action):
        return np.array(
            [
                _clip_norm(hero_pos[0], Config.MAP_SIZE - 1),
                _clip_norm(hero_pos[1], Config.MAP_SIZE - 1),
                _clip_norm(flash_cd, flash_cd_max),
                1.0 if flash_cd <= 0 else 0.0,
                _clip_norm(buff_remain, Config.MAX_BUFF_DURATION),
                has_buff,
                _clip_norm(self.step_no, max(self.max_step, 1)),
                _clip_norm(max(self.max_step - self.step_no, 0), max(self.max_step, 1)),
                1.0 if last_action is not None and last_action >= 8 else 0.0,
                self._recent_flash_ratio(),
            ],
            dtype=np.float32,
        )

    def _build_monster_feature(self, monsters, hero_pos):
        features = []
        monster_states = []
        min_dist = Config.MAX_MONSTER_DIST
        current_threat = 0.0

        for idx in range(2):
            if idx < len(monsters) and isinstance(monsters[idx], dict):
                monster = monsters[idx]
                pos = self._extract_position(monster, fallback=None)
                if pos is None:
                    pos = self._estimate_position_from_relative(monster, hero_pos)
                raw_dist = _l2(hero_pos, pos)
                min_dist = min(min_dist, raw_dist)
                rel_x = pos[0] - hero_pos[0]
                rel_z = pos[1] - hero_pos[1]
                speed = _safe_float(monster.get("speed", 1.0), 1.0)
                threat = self._threat_from_distance(raw_dist)
                current_threat = max(current_threat, threat)
                monster_states.append({"exists": True, "pos": pos, "speed": speed, "threat": threat})
                features.extend(
                    [
                        1.0,
                        _signed_norm(rel_x, Config.RELATIVE_COORD_CLIP),
                        _signed_norm(rel_z, Config.RELATIVE_COORD_CLIP),
                        _clip_norm(raw_dist, Config.MAX_MONSTER_DIST),
                        float(np.sign(rel_x)),
                        float(np.sign(rel_z)),
                        _clip_norm(speed, Config.MAX_MONSTER_SPEED),
                        1.0 if raw_dist <= Config.CLOSE_THREAT_DIST else 0.0,
                    ]
                )
            else:
                monster_states.append({"exists": False, "pos": hero_pos, "speed": 0.0, "threat": 0.0})
                features.extend([0.0] * 8)

        return np.array(features, dtype=np.float32), monster_states, float(min_dist), float(current_threat)

    def _build_item_feature(self, source, hero_pos, monster_states, current_threat, is_buff, has_buff):
        ranked_items = []
        for key, value in source.items():
            pos = value["pos"]
            dist = _l2(hero_pos, pos)
            safety = self._item_safety(pos, dist, monster_states)
            if is_buff:
                priority = np.clip(
                    0.30 * (1.0 - _clip_norm(dist, Config.ITEM_DISTANCE_CLIP))
                    + 0.35 * (1.0 - has_buff)
                    + 0.25 * current_threat
                    + 0.30 * safety,
                    0.0,
                    1.0,
                )
            else:
                priority = np.clip(
                    0.72 * (1.0 - _clip_norm(dist, Config.ITEM_DISTANCE_CLIP))
                    + 0.28 * safety
                    + 0.12 * has_buff
                    - 0.08 * current_threat,
                    0.0,
                    1.0,
                )
            ranked_items.append((priority, dist, key, pos))
        ranked_items.sort(key=lambda x: (-x[0], x[1]))

        features = []
        for idx in range(Config.MAX_TRACKED_ITEMS):
            if idx >= len(ranked_items):
                features.extend([0.0] * 5)
                continue
            priority, dist, _, pos = ranked_items[idx]
            rel_x = pos[0] - hero_pos[0]
            rel_z = pos[1] - hero_pos[1]
            features.extend(
                [
                    1.0,
                    _signed_norm(rel_x, Config.RELATIVE_COORD_CLIP),
                    _signed_norm(rel_z, Config.RELATIVE_COORD_CLIP),
                    _clip_norm(dist, Config.ITEM_DISTANCE_CLIP),
                    float(priority),
                ]
            )
        return np.array(features, dtype=np.float32)

    def _select_target_item(self, source, hero_pos, monster_states, current_threat, is_buff, has_buff):
        best_item = None
        best_score = -1e9
        for key, value in source.items():
            pos = value["pos"]
            dist = _l2(hero_pos, pos)
            safety = self._item_safety(pos, dist, monster_states)
            if is_buff:
                priority = (
                    0.28 * (1.0 - _clip_norm(dist, Config.ITEM_DISTANCE_CLIP))
                    + 0.32 * current_threat
                    + 0.30 * (1.0 - has_buff)
                    + 0.20 * safety
                )
            else:
                priority = (
                    0.78 * (1.0 - _clip_norm(dist, Config.ITEM_DISTANCE_CLIP))
                    + 0.30 * safety
                    + 0.16 * has_buff
                )
            if not is_buff and current_threat > Config.SAFE_ITEM_THREAT_THRESHOLD and dist > 10.0:
                priority *= 0.85
            score = priority - 0.05 * _clip_norm(dist, Config.ITEM_DISTANCE_CLIP)
            if score > best_score:
                best_score = score
                best_item = {
                    "key": key,
                    "pos": pos,
                    "dist": dist,
                    "priority": float(np.clip(priority, 0.0, 1.0)),
                }
        return best_item

    def _build_local_route_feature(self, map_info, hero_pos, hero_speed):
        if not isinstance(map_info, (list, tuple)) or not map_info:
            return np.zeros((24,), dtype=np.float32), self._empty_action_routes(hero_pos)

        size = len(map_info)
        center = size // 2
        features = []
        routes = []

        for delta in MOVE_DIRS:
            step1 = self._local_step_passable(map_info, center, center, delta, 1)
            step2 = step1 and self._local_step_passable(map_info, center, center, delta, 2)
            corridor = self._local_corridor_length(map_info, center, center, delta)
            features.extend(
                [
                    1.0 if step1 else 0.0,
                    1.0 if step2 else 0.0,
                    _clip_norm(corridor, Config.ROUTE_SCAN_LIMIT),
                ]
            )

        for action_idx in range(Config.ACTION_NUM):
            routes.append(self._simulate_local_action(map_info, hero_pos, hero_speed, action_idx))

        return np.array(features, dtype=np.float32), routes

    def _empty_action_routes(self, hero_pos):
        routes = []
        for action_idx in range(Config.ACTION_NUM):
            routes.append(
                {
                    "action_idx": action_idx,
                    "endpoint": hero_pos,
                    "path": [],
                    "moved": False,
                    "progress_steps": 0,
                    "corridor_norm": 0.0,
                    "step1": 0.0,
                    "step2": 0.0,
                    "is_flash": action_idx >= 8,
                }
            )
        return routes

    def _simulate_local_action(self, map_info, hero_pos, hero_speed, action_idx):
        size = len(map_info)
        center = size // 2
        delta = MOVE_DIRS[action_idx % 8]
        is_flash = action_idx >= 8
        max_step = FLASH_DISTANCE[action_idx % 8] if is_flash else hero_speed

        current_row, current_col = center, center
        current_pos = hero_pos
        path = []
        moved = False

        for step in range(1, max_step + 1):
            if is_flash:
                step_passable = self._local_flash_step_passable(map_info, center, center, delta, step)
            else:
                step_passable = self._local_step_passable(map_info, center, center, delta, step)
            if not step_passable:
                break
            current_row = center + delta[1] * step
            current_col = center + delta[0] * step
            current_pos = (hero_pos[0] + delta[0] * step, hero_pos[1] + delta[1] * step)
            path.append(current_pos)
            moved = True

        corridor = 0
        if moved:
            corridor = self._local_corridor_length(map_info, current_row, current_col, delta)

        return {
            "action_idx": action_idx,
            "endpoint": current_pos,
            "path": path,
            "moved": moved,
            "progress_steps": len(path),
            "corridor_norm": _clip_norm(corridor, Config.ROUTE_SCAN_LIMIT),
            "step1": 1.0 if self._local_step_passable(map_info, center, center, delta, 1) else 0.0,
            "step2": 1.0 if self._local_step_passable(map_info, center, center, delta, 2) else 0.0,
            "is_flash": is_flash,
        }

    def _build_action_bias(
        self,
        legal_action,
        action_routes,
        hero_pos,
        monster_states,
        min_monster_dist,
        current_threat,
        target_treasure,
        target_buff,
        has_buff,
        flash_cd,
    ):
        current_min = min_monster_dist
        raw_scores = np.full((Config.ACTION_NUM,), -Config.ACTION_BIAS_CLIP, dtype=np.float32)

        for action_idx in range(Config.ACTION_NUM):
            if legal_action[action_idx] <= 0:
                continue

            route = action_routes[action_idx]
            endpoint = route["endpoint"]
            endpoint_min = self._min_monster_distance(endpoint, monster_states)
            safety_gain = np.clip((endpoint_min - current_min) / max(Config.EARLY_ESCAPE_MARGIN, 1.0), -1.0, 1.0)
            route_bonus = route["corridor_norm"] + 0.5 * route["step2"]
            revisit_penalty = self._endpoint_revisit_penalty(endpoint) + self._path_revisit_penalty(route["path"])
            novelty_bonus = 1.0 - min(1.0, revisit_penalty)
            score = Config.ACTION_ESCAPE_WEIGHT * (0.35 + current_threat) * safety_gain
            score += Config.ACTION_ROUTE_WEIGHT * route_bonus
            score += Config.ACTION_NOVELTY_WEIGHT * novelty_bonus
            score -= Config.ACTION_REVISIT_PENALTY * revisit_penalty

            treasure_progress, treasure_path_bonus = self._item_action_score(endpoint, route["path"], target_treasure)
            if target_treasure is not None:
                treasure_weight = Config.ACTION_TREASURE_WEIGHT * (1.0 + 0.45 * has_buff)
                score += (
                    treasure_weight
                    * target_treasure["priority"]
                    * max(0.0, treasure_progress)
                    * max(0.25, 1.0 - 0.45 * current_threat)
                )
                score += Config.ACTION_TREASURE_PATH_BONUS * target_treasure["priority"] * treasure_path_bonus

            buff_progress, buff_path_bonus = self._item_action_score(endpoint, route["path"], target_buff)
            if target_buff is not None and (has_buff <= 0.0 or current_threat > 0.25):
                score += (
                    Config.ACTION_BUFF_WEIGHT
                    * target_buff["priority"]
                    * max(0.0, buff_progress)
                    * (1.0 - 0.3 * has_buff + 0.5 * current_threat)
                )
                score += Config.ACTION_BUFF_PATH_BONUS * target_buff["priority"] * buff_path_bonus

            path_item_bonus = max(treasure_path_bonus, buff_path_bonus)
            if route["is_flash"]:
                score -= 0.7
                if current_threat >= Config.FLASH_GATE_HIGH_THREAT and safety_gain > 0.08:
                    score += Config.ACTION_FLASH_ESCAPE_BONUS * safety_gain * (0.5 + current_threat)
                if path_item_bonus > 0.0:
                    score += Config.ACTION_FLASH_ITEM_BONUS * path_item_bonus
                move_route = action_routes[action_idx - 8]
                if route["progress_steps"] - move_route["progress_steps"] >= 2 and safety_gain > 0.05:
                    score += Config.ACTION_FLASH_WALL_ESCAPE_BONUS
                if current_threat < Config.FLASH_GATE_LOW_THREAT and path_item_bonus <= 0.0:
                    score -= Config.ACTION_FLASH_IDLE_PENALTY

            if not route["moved"]:
                score -= Config.ACTION_STAY_PENALTY

            if flash_cd > 0 and route["is_flash"]:
                score -= Config.ACTION_FLASH_IDLE_PENALTY

            raw_scores[action_idx] = score

        legal_scores = raw_scores[legal_action > 0]
        if legal_scores.size <= 0:
            return raw_scores

        mean = float(legal_scores.mean())
        std = float(legal_scores.std())
        std = std if std > 1e-6 else 1.0
        normalized = (raw_scores - mean) / std
        scaled = np.clip(normalized * Config.ACTION_BIAS_SCALE, -Config.ACTION_BIAS_CLIP, Config.ACTION_BIAS_CLIP)
        scaled[legal_action <= 0] = -Config.ACTION_BIAS_CLIP
        return scaled.astype(np.float32)

    def _apply_action_constraints(
        self,
        legal_action,
        action_routes,
        action_bias,
        current_threat,
        flash_cd,
        target_treasure,
        target_buff,
    ):
        constrained = np.array(legal_action, dtype=np.float32, copy=True)
        if constrained.shape[0] != Config.ACTION_NUM:
            return constrained
        if constrained[8:].sum() <= 0 or flash_cd > 0:
            return constrained

        nearest_item_dist = min(
            target_treasure["dist"] if target_treasure is not None else float("inf"),
            target_buff["dist"] if target_buff is not None else float("inf"),
        )

        disable_all_flash = False
        if self.runtime_mode == "train":
            if self.curriculum_stage == "stage_a_survive":
                disable_all_flash = current_threat < 0.65 and nearest_item_dist > 5.0
            elif self.curriculum_stage == "stage_b_standard":
                disable_all_flash = (
                    self.step_no < 40
                    and current_threat < Config.FLASH_GATE_LOW_THREAT
                    and nearest_item_dist > Config.FLASH_GATE_ITEM_DIST
                )
            elif self.curriculum_stage == "stage_c_generalize":
                disable_all_flash = (
                    self.step_no < 10
                    and current_threat < Config.FLASH_GATE_LOW_THREAT
                    and nearest_item_dist > Config.FLASH_GATE_ITEM_DIST
                )

        if self._recent_flash_ratio() >= Config.FLASH_GATE_RECENT_RATIO:
            disable_all_flash = disable_all_flash or (
                current_threat < Config.FLASH_GATE_HIGH_THREAT and nearest_item_dist > Config.FLASH_GATE_ITEM_DIST
            )

        for action_idx in range(8, Config.ACTION_NUM):
            if constrained[action_idx] <= 0:
                continue
            route = action_routes[action_idx]
            if action_bias[action_idx] < -1.4 and current_threat < Config.FLASH_GATE_HIGH_THREAT:
                constrained[action_idx] = 0.0
            if (
                not disable_all_flash
                and nearest_item_dist <= 3.0
                and action_bias[action_idx] < 0.2
                and current_threat < Config.FLASH_GATE_HIGH_THREAT
            ):
                constrained[action_idx] = 0.0
            if not route["moved"]:
                constrained[action_idx] = 0.0

        if disable_all_flash:
            constrained[8:] = 0.0

        if constrained.sum() <= 0:
            constrained[:8] = 1.0

        return constrained

    def _item_action_score(self, endpoint, path, target_item):
        if target_item is None:
            return 0.0, 0.0
        current_dist = target_item["dist"]
        next_dist = _l2(endpoint, target_item["pos"])
        progress = np.clip((current_dist - next_dist) / max(Config.ITEM_DISTANCE_CLIP, 1.0), -1.0, 1.0)
        path_bonus = 0.0
        for pos in path:
            if _l2(pos, target_item["pos"]) <= 1.5:
                path_bonus = 1.0
                break
        return float(progress), float(path_bonus)

    def _build_progress_feature(self, env_info, treasure_count, buff_count):
        total_treasure = max(
            _safe_int(env_info.get("total_treasure", Config.DEFAULT_TREASURE_COUNT), Config.DEFAULT_TREASURE_COUNT),
            1,
        )
        total_buff = max(
            _safe_int(env_info.get("total_buff", Config.DEFAULT_BUFF_COUNT), Config.DEFAULT_BUFF_COUNT),
            1,
        )
        return np.array(
            [
                self._eta_norm(self.monster_interval),
                self._eta_norm(self.monster_speedup),
                _clip_norm(treasure_count, total_treasure),
                _clip_norm(buff_count, total_buff),
            ],
            dtype=np.float32,
        )

    def _eta_norm(self, event_step):
        if event_step <= 0:
            return 0.0
        return _clip_norm(max(event_step - self.step_no, 0), max(event_step, 1))

    def _compute_reward(
        self,
        hero_pos,
        min_monster_dist,
        current_threat,
        treasure_count,
        buff_count,
        flash_count,
        total_score,
        last_action,
        target_treasure,
        target_buff,
        has_buff,
        visit_before,
    ):
        reward = Config.SURVIVE_REWARD
        reward_info = {}

        prev_min_dist_norm = _clip_norm(self.last_min_monster_dist, Config.MAX_MONSTER_DIST)
        cur_min_dist_norm = _clip_norm(min_monster_dist, Config.MAX_MONSTER_DIST)
        reward += Config.DANGER_ESCAPE_REWARD_SCALE * (cur_min_dist_norm - prev_min_dist_norm)

        treasure_delta = treasure_count - self.last_treasure_count
        buff_delta = buff_count - self.last_buff_count
        flash_delta = flash_count - self.last_flash_count

        if treasure_delta > 0:
            reward += treasure_delta * Config.TREASURE_REWARD
            if has_buff > 0:
                reward += treasure_delta * Config.BUFF_TREASURE_SYNERGY_REWARD

        if buff_delta > 0:
            danger_bonus = 1.0 - cur_min_dist_norm
            reward += buff_delta * (Config.BUFF_REWARD + Config.BUFF_DANGER_BONUS * danger_bonus)

        reward += self._target_progress_reward(
            target=target_treasure,
            last_key=self.last_target_treasure_key,
            last_dist=self.last_target_treasure_dist,
            current_threat=current_threat,
            scale=Config.TREASURE_PROGRESS_REWARD_SCALE,
        )
        reward += self._target_progress_reward(
            target=target_buff,
            last_key=self.last_target_buff_key,
            last_dist=self.last_target_buff_dist,
            current_threat=current_threat * 0.7,
            scale=Config.BUFF_PROGRESS_REWARD_SCALE,
        )

        if visit_before <= 0:
            reward += Config.EXPLORATION_REWARD_FIRST
        else:
            reward += Config.EXPLORATION_REWARD_REPEAT * max(
                0.0,
                1.0 - min(1.0, visit_before / Config.MAX_VISIT_COUNT),
            )
            reward -= Config.REVISIT_REWARD_PENALTY * min(1.0, visit_before / Config.MAX_VISIT_COUNT)

        if self.last_pos is not None and hero_pos == self.last_pos:
            self.stuck_streak += 1
            self.stuck_count += 1
            reward += Config.STUCK_PENALTY * min(self.stuck_streak, Config.STUCK_STREAK_LIMIT) / Config.STUCK_STREAK_LIMIT
        else:
            self.stuck_streak = 0

        if self._detect_loop():
            reward += Config.LOOP_PENALTY

        if min_monster_dist < Config.EARLY_ESCAPE_MARGIN:
            reward += Config.EARLY_ESCAPE_PENALTY * (
                (Config.EARLY_ESCAPE_MARGIN - min_monster_dist) / max(Config.EARLY_ESCAPE_MARGIN, 1.0)
            )

        threat_penalty = 0.0
        if min_monster_dist <= Config.CLOSE_THREAT_DIST:
            threat_penalty += Config.THREAT_PENALTY * (
                (Config.CLOSE_THREAT_DIST - min_monster_dist + 1.0) / max(Config.CLOSE_THREAT_DIST, 1.0)
            )
        if min_monster_dist <= Config.CRITICAL_THREAT_DIST:
            threat_penalty += Config.CRITICAL_THREAT_PENALTY * (
                (Config.CRITICAL_THREAT_DIST - min_monster_dist + 1.0) / max(Config.CRITICAL_THREAT_DIST, 1.0)
            )
        reward += threat_penalty

        if last_action is not None and last_action >= 8:
            flash_gain = cur_min_dist_norm - prev_min_dist_norm
            target_progress = 0.0
            if target_treasure is not None and self.last_target_treasure_key == target_treasure["key"] and self.last_target_treasure_dist is not None:
                target_progress = max(target_progress, self.last_target_treasure_dist - target_treasure["dist"])
            if target_buff is not None and self.last_target_buff_key == target_buff["key"] and self.last_target_buff_dist is not None:
                target_progress = max(target_progress, self.last_target_buff_dist - target_buff["dist"])
            useful_flash = (
                flash_gain >= Config.GOOD_FLASH_DISTANCE_GAIN
                or treasure_delta > 0
                or buff_delta > 0
                or target_progress > 4.0
            )
            reward += Config.GOOD_FLASH_REWARD if useful_flash else Config.BAD_FLASH_PENALTY
            reward_info["flash_was_useful"] = 1.0 if useful_flash else 0.0
        else:
            reward_info["flash_was_useful"] = 0.0

        reward_info["min_monster_dist"] = float(min_monster_dist)
        reward_info["current_threat"] = float(current_threat)
        reward_info["threat_penalty"] = float(threat_penalty)
        reward_info["treasure_delta"] = int(treasure_delta)
        reward_info["buff_delta"] = int(buff_delta)
        reward_info["flash_delta"] = int(flash_delta)
        reward_info["score_delta"] = float(total_score - self.last_total_score)
        return float(reward), reward_info

    def _target_progress_reward(self, target, last_key, last_dist, current_threat, scale):
        if target is None or last_key != target["key"] or last_dist is None:
            return 0.0
        progress = last_dist - target["dist"]
        if progress <= 0:
            return 0.0
        return scale * progress * max(0.0, 1.0 - current_threat)

    def _min_monster_distance(self, pos, monster_states):
        dist = Config.MAX_MONSTER_DIST
        for monster in monster_states:
            if not monster.get("exists", False):
                continue
            dist = min(dist, _l2(pos, monster["pos"]))
        return float(dist)

    def _item_safety(self, item_pos, hero_to_item_dist, monster_states):
        best_monster_dist = Config.ITEM_DISTANCE_CLIP
        for monster in monster_states:
            if not monster.get("exists", False):
                continue
            best_monster_dist = min(best_monster_dist, _l2(item_pos, monster["pos"]))
        return float(
            np.clip(
                0.5 + 0.5 * (best_monster_dist - hero_to_item_dist) / max(Config.ITEM_DISTANCE_CLIP, 1.0),
                0.0,
                1.0,
            )
        )

    def _threat_from_distance(self, min_monster_dist):
        if min_monster_dist >= Config.EARLY_ESCAPE_MARGIN:
            return 0.0
        return float(np.clip(1.0 - min_monster_dist / max(Config.EARLY_ESCAPE_MARGIN, 1.0), 0.0, 1.0))

    def _recent_flash_ratio(self):
        if not self.recent_actions:
            return 0.0
        flash_count = sum(1 for action in self.recent_actions if action >= 8)
        return flash_count / float(len(self.recent_actions))

    def _detect_loop(self):
        if len(self.recent_positions) < 4:
            return False
        p0, p1, p2, p3 = list(self.recent_positions)[-4:]
        return p0 == p2 and p1 == p3 and p0 != p1

    def _local_corridor_length(self, map_info, center_row, center_col, delta):
        length = 0
        for step in range(1, Config.ROUTE_SCAN_LIMIT + 1):
            if not self._local_step_passable(map_info, center_row, center_col, delta, step):
                break
            length += 1
        return length

    def _local_step_passable(self, map_info, center_row, center_col, delta, step):
        target_row = center_row + delta[1] * step
        target_col = center_col + delta[0] * step
        if not self._local_cell_passable(map_info, target_row, target_col):
            return False
        if delta[0] != 0 and delta[1] != 0:
            side_row = center_row + delta[1] * step
            side_col = center_col + delta[0] * (step - 1)
            other_row = center_row + delta[1] * (step - 1)
            other_col = center_col + delta[0] * step
            return self._local_cell_passable(map_info, side_row, side_col) or self._local_cell_passable(
                map_info, other_row, other_col
            )
        return True

    def _local_flash_step_passable(self, map_info, center_row, center_col, delta, step):
        target_row = center_row + delta[1] * step
        target_col = center_col + delta[0] * step
        return self._local_cell_passable(map_info, target_row, target_col)

    def _local_cell_passable(self, map_info, row, col):
        if row < 0 or col < 0 or row >= len(map_info) or col >= len(map_info[row]):
            return False
        return _safe_int(map_info[row][col], 0) == 1

    def _endpoint_revisit_penalty(self, endpoint):
        x, z = endpoint
        x = int(np.clip(x, 0, Config.MAP_SIZE - 1))
        z = int(np.clip(z, 0, Config.MAP_SIZE - 1))
        visit_penalty = min(1.0, float(self.visit_count_map[z, x]) / Config.MAX_VISIT_COUNT)
        recent_penalty = sum(1 for pos in self.recent_positions if pos == (x, z)) / max(len(self.recent_positions), 1)
        return 0.65 * visit_penalty + 0.35 * recent_penalty

    def _path_revisit_penalty(self, path):
        if not path:
            return 0.0
        penalties = [self._endpoint_revisit_penalty(pos) for pos in path]
        return float(np.mean(penalties))
