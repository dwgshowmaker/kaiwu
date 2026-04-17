#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Feature preprocessor and reward shaping for the DIY Gorge Chase PPO agent.
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

        self.stuck_count = 0
        self.stuck_streak = 0
        self.total_min_monster_dist = 0.0
        self.frame_count = 0

        self.recent_positions = deque(maxlen=Config.RECENT_POSITION_WINDOW)
        self.recent_actions = deque(maxlen=Config.RECENT_ACTION_WINDOW)
        self.latest_info = {}

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
        self.recent_positions.append(hero_pos)
        if last_action is not None and last_action >= 0:
            self.recent_actions.append(int(last_action))

        legal_action = self._parse_legal_action(
            observation.get("legal_action", observation.get("legal_act", [1] * Config.ACTION_NUM))
        )

        monsters = self._extract_monsters(frame_state)
        monster_feature, monster_states, min_monster_dist = self._build_monster_feature(monsters, hero_pos)

        organs = self._extract_organs(frame_state)
        treasure_items = self._extract_items(organs, sub_type=1)
        buff_items = self._extract_items(organs, sub_type=2)

        treasure_count = _safe_int(
            hero.get("treasure_collected_count", env_info.get("treasures_collected", self.last_treasure_count)),
            self.last_treasure_count,
        )
        buff_count = _safe_int(env_info.get("collected_buff", self.last_buff_count), self.last_buff_count)
        flash_count = _safe_int(env_info.get("flash_count", self.last_flash_count), self.last_flash_count)
        total_score = _safe_float(env_info.get("total_score", self.last_total_score), self.last_total_score)

        flash_cd = _safe_int(
            hero.get("flash_cooldown", hero.get("talent_cooldown", 0)),
            0,
        )
        flash_cd_max = max(_safe_int(env_info.get("flash_cooldown", Config.MAX_FLASH_CD), Config.MAX_FLASH_CD), 1)
        buff_remain = _safe_int(hero.get("buff_remaining_time", hero.get("buff_remain", 0)), 0)
        has_buff = 1.0 if buff_remain > 0 else 0.0

        hero_feature = self._build_hero_feature(
            hero_pos=hero_pos,
            flash_cd=flash_cd,
            flash_cd_max=flash_cd_max,
            buff_remain=buff_remain,
            has_buff=has_buff,
            last_action=last_action,
        )
        treasure_feature = self._build_item_feature(
            items=treasure_items,
            hero_pos=hero_pos,
            monster_states=monster_states,
            current_threat=self._threat_from_distance(min_monster_dist),
            has_buff=has_buff,
            is_buff=False,
        )
        buff_feature = self._build_item_feature(
            items=buff_items,
            hero_pos=hero_pos,
            monster_states=monster_states,
            current_threat=self._threat_from_distance(min_monster_dist),
            has_buff=has_buff,
            is_buff=True,
        )
        legal_action = self._apply_curriculum_action_mask(
            legal_action=legal_action,
            hero_pos=hero_pos,
            treasure_items=treasure_items,
            buff_items=buff_items,
            min_monster_dist=min_monster_dist,
            flash_cd=flash_cd,
        )
        local_route_feature = self._build_local_route_feature(map_info)
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
            treasure_count=treasure_count,
            buff_count=buff_count,
            flash_count=flash_count,
            total_score=total_score,
            last_action=last_action,
        )

        self.total_min_monster_dist += min_monster_dist
        self.frame_count += 1

        self.last_pos = hero_pos
        self.last_min_monster_dist = min_monster_dist
        self.last_treasure_count = treasure_count
        self.last_buff_count = buff_count
        self.last_flash_count = flash_count
        self.last_total_score = total_score

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

        return feature, legal_action, [reward], self.latest_info

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

    def _extract_items(self, organs, sub_type):
        items = []
        for organ in organs:
            if _safe_int(organ.get("sub_type", 0), 0) != sub_type:
                continue
            if _safe_int(organ.get("status", 1), 1) != 1:
                continue
            items.append(organ)
        return items

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
        return legal_action.astype(np.float32)

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
                monster_states.append({"exists": True, "pos": pos, "speed": speed})
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
                monster_states.append({"exists": False, "pos": hero_pos, "speed": 0.0})
                features.extend([0.0] * 8)

        return np.array(features, dtype=np.float32), monster_states, float(min_dist)

    def _build_item_feature(self, items, hero_pos, monster_states, current_threat, has_buff, is_buff):
        item_infos = []
        for item in items:
            pos = self._extract_position(item, fallback=None)
            if pos is None:
                pos = self._estimate_position_from_relative(item, hero_pos)
            dist = _l2(hero_pos, pos)
            item_infos.append((dist, pos))
        item_infos.sort(key=lambda x: x[0])

        features = []
        step_norm = _clip_norm(self.step_no, max(self.max_step, 1))
        for idx in range(Config.MAX_TRACKED_ITEMS):
            if idx >= len(item_infos):
                features.extend([0.0] * 5)
                continue

            dist, pos = item_infos[idx]
            rel_x = pos[0] - hero_pos[0]
            rel_z = pos[1] - hero_pos[1]
            safety = self._item_safety(pos, dist, monster_states)

            if is_buff:
                priority = np.clip(
                    0.35 * (1.0 - _clip_norm(dist, Config.ITEM_DISTANCE_CLIP))
                    + 0.35 * current_threat
                    + 0.2 * (1.0 - has_buff)
                    + 0.15 * (1.0 - step_norm)
                    - 0.25 * (1.0 - safety),
                    0.0,
                    1.0,
                )
            else:
                priority = np.clip(
                    0.55 * (1.0 - _clip_norm(dist, Config.ITEM_DISTANCE_CLIP))
                    + 0.45 * safety,
                    0.0,
                    1.0,
                )

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

    def _build_local_route_feature(self, map_info):
        if not isinstance(map_info, (list, tuple)) or not map_info:
            return np.zeros((24,), dtype=np.float32)

        size = len(map_info)
        center = size // 2
        features = []
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
        return np.array(features, dtype=np.float32)

    def _build_progress_feature(self, env_info, treasure_count, buff_count):
        total_treasure = max(
            _safe_int(env_info.get("total_treasure", Config.DEFAULT_TREASURE_COUNT), Config.DEFAULT_TREASURE_COUNT),
            1,
        )
        total_buff = max(
            _safe_int(env_info.get("total_buff", Config.DEFAULT_BUFF_COUNT), Config.DEFAULT_BUFF_COUNT),
            1,
        )
        monster2_eta_norm = self._eta_norm(self.monster_interval)
        monster_speedup_eta_norm = self._eta_norm(self.monster_speedup)
        return np.array(
            [
                monster2_eta_norm,
                monster_speedup_eta_norm,
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
        treasure_count,
        buff_count,
        flash_count,
        total_score,
        last_action,
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
        if buff_delta > 0:
            danger_bonus = 1.0 - cur_min_dist_norm
            reward += buff_delta * (Config.BUFF_REWARD + Config.BUFF_DANGER_BONUS * danger_bonus)

        if self.last_pos is not None and hero_pos == self.last_pos:
            self.stuck_streak += 1
            self.stuck_count += 1
            reward += Config.STUCK_PENALTY * min(self.stuck_streak, Config.STUCK_STREAK_LIMIT) / Config.STUCK_STREAK_LIMIT
        else:
            self.stuck_streak = 0

        if self._detect_loop():
            reward += Config.LOOP_PENALTY

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
            useful_flash = flash_gain >= Config.GOOD_FLASH_DISTANCE_GAIN or treasure_delta > 0 or buff_delta > 0
            reward += Config.GOOD_FLASH_REWARD if useful_flash else Config.BAD_FLASH_PENALTY
            reward_info["flash_was_useful"] = 1.0 if useful_flash else 0.0
        else:
            reward_info["flash_was_useful"] = 0.0

        reward_info["min_monster_dist"] = float(min_monster_dist)
        reward_info["threat_penalty"] = float(threat_penalty)
        reward_info["treasure_delta"] = int(treasure_delta)
        reward_info["buff_delta"] = int(buff_delta)
        reward_info["flash_delta"] = int(flash_delta)
        reward_info["score_delta"] = float(total_score - self.last_total_score)
        return float(reward), reward_info

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
        if min_monster_dist >= Config.CLOSE_THREAT_DIST:
            return 0.0
        return float(np.clip(1.0 - min_monster_dist / max(Config.CLOSE_THREAT_DIST, 1.0), 0.0, 1.0))

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

    def _local_cell_passable(self, map_info, row, col):
        if row < 0 or col < 0 or row >= len(map_info) or col >= len(map_info[row]):
            return False
        return _safe_int(map_info[row][col], 0) == 1

    def _apply_curriculum_action_mask(
        self,
        legal_action,
        hero_pos,
        treasure_items,
        buff_items,
        min_monster_dist,
        flash_cd,
    ):
        legal_action = np.array(legal_action, dtype=np.float32, copy=True)
        if legal_action.shape[0] != Config.ACTION_NUM or legal_action[8:].sum() <= 0:
            return legal_action

        if flash_cd > 0:
            return legal_action

        threat = self._threat_from_distance(min_monster_dist)
        near_item_dist = self._nearest_item_distance(hero_pos, treasure_items, buff_items)
        recent_flash_ratio = self._recent_flash_ratio()
        disable_flash = False

        if self.curriculum_stage == "stage_a_survive":
            disable_flash = True
        elif self.curriculum_stage == "stage_b_standard":
            disable_flash = (
                self.step_no < Config.FLASH_GATE_STAGE_B_STEP
                and threat < Config.FLASH_GATE_LOW_THREAT
                and near_item_dist > Config.FLASH_GATE_ITEM_DIST
            )
        elif self.curriculum_stage == "stage_c_generalize":
            disable_flash = (
                self.step_no < Config.FLASH_GATE_STAGE_C_STEP
                and threat < Config.FLASH_GATE_LOW_THREAT
                and near_item_dist > Config.FLASH_GATE_ITEM_DIST
            )

        if (
            not disable_flash
            and recent_flash_ratio >= Config.FLASH_GATE_RECENT_RATIO
            and threat < Config.FLASH_GATE_HIGH_THREAT
            and near_item_dist > Config.FLASH_GATE_ITEM_DIST
        ):
            disable_flash = True

        if (
            not disable_flash
            and near_item_dist <= 3.0
            and threat < Config.FLASH_GATE_HIGH_THREAT
        ):
            disable_flash = True

        if disable_flash:
            legal_action[8:] = 0.0
            if legal_action.sum() <= 0:
                legal_action[:8] = 1.0

        return legal_action

    def _nearest_item_distance(self, hero_pos, treasure_items, buff_items):
        best = float("inf")
        for item in list(treasure_items) + list(buff_items):
            pos = self._extract_position(item, fallback=None)
            if pos is None:
                pos = self._estimate_position_from_relative(item, hero_pos)
            best = min(best, _l2(hero_pos, pos))
        return best
