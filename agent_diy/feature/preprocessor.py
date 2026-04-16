#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Feature preprocessor and reward design for the DIY PPO agent.
DIY PPO 智能体特征预处理与奖励设计。
"""


import math

import numpy as np

from agent_diy.conf.conf import Config


MAP_SIZE = 128.0
MAX_MAP_DISTANCE = MAP_SIZE * math.sqrt(2.0)
MAX_MONSTER_SPEED = 5.0
MAX_FLASH_CD = 2000.0
MAX_BUFF_DURATION = 50.0
MAX_DIRECTIONAL_BUCKET = 5.0
LOCAL_MAP_SIZE = 7
CLOSE_THREAT_DISTANCE = 8.0
DANGER_PRIOR_DISTANCE = 18.0
ESCAPE_LOOKAHEAD_STEPS = 3
BLOCKED_ACTION_COOLDOWN = 4
FLASH_ORTHOGONAL_DISTANCE = 10
FLASH_DIAGONAL_DISTANCE = 8
GOOD_FLASH_ESCAPE_GAIN = 6.0
BAD_FLASH_ESCAPE_GAIN = 2.0
MOVE_ACTION_NUM = 8
FLASH_REVIEW_STEPS = 3
LATE_GAME_STEP_THRESHOLD = 400
FLASH_LOW_DANGER_THRESHOLD = 0.45
FLASH_CLEAR_DANGER_THRESHOLD = 0.4
FLASH_TRAP_DANGER_THRESHOLD = 0.7
LATE_GAME_SAFE_DISTANCE = 10.0
POTENTIAL_GAMMA = 0.99
RESOURCE_POTENTIAL_DISTANCE = 45.0
BUFF_POTENTIAL_DISTANCE = 36.0
FLASH_HOLD_POTENTIAL_GAIN = 0.08
FLASH_TRAP_POTENTIAL_GAIN = 0.02
SURVIVAL_MILESTONES = {
    200: 0.15,
    400: 0.25,
    600: 0.4,
    800: 0.6,
}

DIRECTION_TO_VECTOR = {
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

ACTION_TO_VECTOR = {
    0: (1, 0),
    1: (1, -1),
    2: (0, -1),
    3: (-1, -1),
    4: (-1, 0),
    5: (-1, 1),
    6: (0, 1),
    7: (1, 1),
    8: (FLASH_ORTHOGONAL_DISTANCE, 0),
    9: (FLASH_DIAGONAL_DISTANCE, -FLASH_DIAGONAL_DISTANCE),
    10: (0, -FLASH_ORTHOGONAL_DISTANCE),
    11: (-FLASH_DIAGONAL_DISTANCE, -FLASH_DIAGONAL_DISTANCE),
    12: (-FLASH_ORTHOGONAL_DISTANCE, 0),
    13: (-FLASH_DIAGONAL_DISTANCE, FLASH_DIAGONAL_DISTANCE),
    14: (0, FLASH_ORTHOGONAL_DISTANCE),
    15: (FLASH_DIAGONAL_DISTANCE, FLASH_DIAGONAL_DISTANCE),
}


def _norm(v, v_max, v_min=0.0):
    """Normalize value to [0, 1]."""
    v = float(np.clip(v, v_min, v_max))
    return (v - v_min) / (v_max - v_min) if (v_max - v_min) > 1e-6 else 0.0


def _norm_signed(v, scale):
    """Normalize signed value to [-1, 1]."""
    if scale <= 1e-6:
        return 0.0
    return float(np.clip(v / scale, -1.0, 1.0))


def _distance(dx, dz):
    return float(math.sqrt(dx * dx + dz * dz))


def _safe_pos(pos):
    if not isinstance(pos, dict):
        return None
    if "x" not in pos or "z" not in pos:
        return None
    return float(pos["x"]), float(pos["z"])


class Preprocessor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.step_no = 0
        self.max_step = 200
        self.has_last_state = False
        self.last_min_monster_dist = MAX_MAP_DISTANCE
        self.last_hero_pos = None
        self.last_treasure_count = 0
        self.last_buff_count = 0
        self.last_flash_count = 0
        self.stuck_steps = 0
        self.total_stuck_count = 0
        self.blocked_action_cooldowns = [0] * Config.ACTION_NUM
        self.blocked_this_step = 0
        self.total_blocked_count = 0
        self.danger_steps = 0
        self.near_death_count = 0
        self.good_flash_count = 0
        self.bad_flash_count = 0
        self.flash_escape_gain_sum = 0.0
        self.flash_trap_count = 0
        self.flash_hold_count = 0
        self.late_game_steps = 0
        self.survival_milestones = set()
        self.flash_review_steps = 0
        self.flash_review_origin_dist = MAX_MAP_DISTANCE
        self.flash_review_origin_potential = 0.0
        self.flash_review_peak_potential = 0.0
        self.flash_review_bad = False
        self.flash_review_min_danger = 1.0
        self.flash_review_clear_steps = 0
        self.last_state_potential = 0.0

    def feature_process(self, env_obs, last_action):
        """Process env_obs into feature vector, legal_action mask, reward and metrics."""
        observation = env_obs["observation"]
        frame_state = observation["frame_state"]
        env_info = observation["env_info"]
        map_info = observation["map_info"]
        legal_act_raw = observation["legal_action"]

        self.step_no = observation["step_no"]
        self.max_step = max(1, int(env_info.get("max_step", 200)))

        hero = frame_state["heroes"]
        hero_pos = hero["pos"]
        hero_x = float(hero_pos["x"])
        hero_z = float(hero_pos["z"])

        monsters = frame_state.get("monsters", [])
        organs = frame_state.get("organs", [])

        monster_feats, min_monster_dist, closest_monster_rel = self._build_monster_features(
            monsters, hero_x, hero_z
        )
        hero_feat = self._build_hero_features(hero, hero_x, hero_z)
        flash_ready = bool(hero_feat[3] > 0.5)
        has_buff = bool(hero_feat[5] > 0.5)
        treasure_feat = self._build_organ_features(
            organs=organs,
            hero_x=hero_x,
            hero_z=hero_z,
            sub_type=1,
            closest_monster_rel=closest_monster_rel,
            is_buff=False,
        )
        buff_feat = self._build_organ_features(
            organs=organs,
            hero_x=hero_x,
            hero_z=hero_z,
            sub_type=2,
            closest_monster_rel=closest_monster_rel,
            is_buff=True,
            has_buff=has_buff,
            min_monster_dist=min_monster_dist,
        )
        nearest_treasure_dist = self._closest_active_organ_dist(organs, hero_x, hero_z, sub_type=1)
        nearest_buff_dist = self._closest_active_organ_dist(organs, hero_x, hero_z, sub_type=2)
        map_feat = self._build_local_map_features(map_info)
        legal_action = self._build_legal_action(legal_act_raw)
        progress_feat = self._build_progress_features(env_info)
        self._update_navigation_memory(hero_pos=(hero_x, hero_z), last_action=last_action)

        feature = np.concatenate(
            [
                hero_feat,
                monster_feats,
                treasure_feat,
                buff_feat,
                map_feat,
                np.array(legal_action, dtype=np.float32),
                progress_feat,
            ]
        ).astype(np.float32)

        if len(feature) != Config.DIM_OF_OBSERVATION:
            raise ValueError(
                f"Feature length mismatch: got {len(feature)}, expected {Config.DIM_OF_OBSERVATION}"
            )

        (
            safe_action,
            danger_level,
            safe_action_score,
            safe_path_len,
            safe_action_margin,
            safe_is_flash,
            safe_trap_risk,
        ) = self._select_safe_action(
            closest_monster_rel=closest_monster_rel,
            min_monster_dist=min_monster_dist,
            map_info=map_info,
            legal_action=legal_action,
        )

        reward, metrics = self._calc_reward_and_metrics(
            env_info=env_info,
            hero=hero,
            hero_pos=(hero_x, hero_z),
            min_monster_dist=min_monster_dist,
            last_action=last_action,
            danger_level=danger_level,
            safe_path_len=safe_path_len,
            safe_trap_risk=safe_trap_risk,
            flash_ready=flash_ready,
            has_buff=has_buff,
            nearest_treasure_dist=nearest_treasure_dist,
            nearest_buff_dist=nearest_buff_dist,
        )
        metrics["safe_action"] = float(safe_action)
        metrics["danger_level"] = float(danger_level)
        metrics["safe_action_score"] = float(safe_action_score)
        metrics["safe_path_len"] = float(safe_path_len)
        metrics["safe_action_margin"] = float(safe_action_margin)
        metrics["safe_is_flash"] = float(safe_is_flash)
        metrics["safe_trap_risk"] = float(safe_trap_risk)

        return feature, legal_action, reward, metrics

    def _build_hero_features(self, hero, hero_x, hero_z):
        flash_cd = float(hero.get("flash_cooldown", 0))
        buff_remain = float(hero.get("buff_remaining_time", 0))
        step_norm = _norm(self.step_no, self.max_step)
        return np.array(
            [
                _norm(hero_x, MAP_SIZE),
                _norm(hero_z, MAP_SIZE),
                _norm(flash_cd, MAX_FLASH_CD),
                float(flash_cd <= 0),
                _norm(buff_remain, MAX_BUFF_DURATION),
                float(buff_remain > 0),
                step_norm,
                1.0 - step_norm,
            ],
            dtype=np.float32,
        )

    def _build_monster_features(self, monsters, hero_x, hero_z):
        features = []
        min_dist = MAX_MAP_DISTANCE
        closest_rel = None

        for i in range(2):
            if i >= len(monsters):
                features.extend([0.0] * 8)
                continue

            monster = monsters[i]
            pos = _safe_pos(monster.get("pos"))
            if pos is not None:
                dx = pos[0] - hero_x
                dz = pos[1] - hero_z
                raw_dist = _distance(dx, dz)
                dir_x = float(np.sign(dx))
                dir_z = float(np.sign(dz))
            else:
                dir_x, dir_z = DIRECTION_TO_VECTOR.get(
                    int(monster.get("hero_relative_direction", 0)), (0.0, 0.0)
                )
                dist_bucket = float(monster.get("hero_l2_distance", MAX_DIRECTIONAL_BUCKET))
                raw_dist = (dist_bucket + 0.5) * 30.0
                dx = dir_x * raw_dist
                dz = dir_z * raw_dist

            if raw_dist < min_dist:
                min_dist = raw_dist
                closest_rel = (dx, dz)

            features.extend(
                [
                    1.0,
                    _norm_signed(dx, MAP_SIZE),
                    _norm_signed(dz, MAP_SIZE),
                    _norm(raw_dist, MAX_MAP_DISTANCE),
                    dir_x,
                    dir_z,
                    _norm(monster.get("speed", 1), MAX_MONSTER_SPEED),
                    float(raw_dist <= CLOSE_THREAT_DISTANCE),
                ]
            )

        return np.array(features, dtype=np.float32), min_dist, closest_rel

    def _build_organ_features(
        self,
        organs,
        hero_x,
        hero_z,
        sub_type,
        closest_monster_rel,
        is_buff,
        has_buff=False,
        min_monster_dist=MAX_MAP_DISTANCE,
    ):
        candidates = []
        for organ in organs:
            if int(organ.get("sub_type", -1)) != sub_type:
                continue
            if int(organ.get("status", 1)) != 1:
                continue
            pos = _safe_pos(organ.get("pos"))
            if pos is None:
                continue
            dx = pos[0] - hero_x
            dz = pos[1] - hero_z
            candidates.append((dx, dz, _distance(dx, dz)))

        candidates.sort(key=lambda item: item[2])

        features = []
        for idx in range(2):
            if idx >= len(candidates):
                features.extend([0.0] * 5)
                continue

            dx, dz, raw_dist = candidates[idx]
            if is_buff:
                utility_flag = float(
                    (not has_buff) and (min_monster_dist <= 25.0 or self.step_no < self.max_step * 0.7)
                )
            else:
                utility_flag = self._same_escape_quadrant(dx, dz, closest_monster_rel)

            features.extend(
                [
                    1.0,
                    _norm_signed(dx, MAP_SIZE),
                    _norm_signed(dz, MAP_SIZE),
                    _norm(raw_dist, MAX_MAP_DISTANCE),
                    utility_flag,
                ]
            )

        return np.array(features, dtype=np.float32)

    def _closest_active_organ_dist(self, organs, hero_x, hero_z, sub_type):
        min_dist = MAX_MAP_DISTANCE
        for organ in organs:
            if int(organ.get("sub_type", -1)) != sub_type:
                continue
            if int(organ.get("status", 1)) != 1:
                continue
            pos = _safe_pos(organ.get("pos"))
            if pos is None:
                continue
            dx = pos[0] - hero_x
            dz = pos[1] - hero_z
            min_dist = min(min_dist, _distance(dx, dz))
        return min_dist

    def _same_escape_quadrant(self, dx, dz, closest_monster_rel):
        if closest_monster_rel is None:
            return 0.0
        monster_dx, monster_dz = closest_monster_rel
        escape_dx = -monster_dx
        escape_dz = -monster_dz
        dot_value = dx * escape_dx + dz * escape_dz
        return float(dot_value > 0)

    def _build_local_map_features(self, map_info):
        map_feat = np.zeros(LOCAL_MAP_SIZE * LOCAL_MAP_SIZE, dtype=np.float32)
        if map_info is None or len(map_info) == 0:
            return map_feat

        map_arr = np.array(map_info, dtype=np.float32)
        if map_arr.ndim != 2 or map_arr.size == 0:
            return map_feat

        h, w = map_arr.shape
        flat_idx = 0
        for row in range(LOCAL_MAP_SIZE):
            row_start = int(row * h / LOCAL_MAP_SIZE)
            row_end = max(row_start + 1, int((row + 1) * h / LOCAL_MAP_SIZE))
            for col in range(LOCAL_MAP_SIZE):
                col_start = int(col * w / LOCAL_MAP_SIZE)
                col_end = max(col_start + 1, int((col + 1) * w / LOCAL_MAP_SIZE))
                block = map_arr[row_start:row_end, col_start:col_end]
                map_feat[flat_idx] = float(np.mean(block != 0)) if block.size else 0.0
                flat_idx += 1

        return map_feat

    def _build_legal_action(self, legal_act_raw):
        action_num = Config.ACTION_NUM
        legal_action = [1] * action_num
        if isinstance(legal_act_raw, list) and legal_act_raw:
            if isinstance(legal_act_raw[0], bool):
                for idx in range(min(action_num, len(legal_act_raw))):
                    legal_action[idx] = int(legal_act_raw[idx])
            else:
                valid_set = {int(action) for action in legal_act_raw if int(action) < action_num}
                legal_action = [1 if idx in valid_set else 0 for idx in range(action_num)]

        if sum(legal_action) == 0:
            legal_action = [1] * action_num
        return legal_action

    def _build_progress_features(self, env_info):
        monster_interval = int(env_info.get("monster_interval", 300))
        monster2_eta = self._eta_norm(monster_interval)
        speedup_interval = int(env_info.get("monster_speedup", env_info.get("monster_speedup_time", -1)))
        speedup_eta = self._eta_norm(speedup_interval) if speedup_interval > 0 else 0.5

        total_treasure = max(1, int(env_info.get("total_treasure", 10)))
        treasure_count = int(env_info.get("treasures_collected", 0))
        total_buff = max(1, int(env_info.get("total_buff", 2)))
        buff_count = int(env_info.get("collected_buff", 0))

        return np.array(
            [
                monster2_eta,
                speedup_eta,
                _norm(treasure_count, total_treasure),
                _norm(buff_count, total_buff),
            ],
            dtype=np.float32,
        )

    def _eta_norm(self, interval):
        if interval <= 0:
            return 0.5
        return _norm(max(interval - self.step_no, 0), interval)

    def _update_navigation_memory(self, hero_pos, last_action):
        self.blocked_this_step = 0
        for idx, cooldown in enumerate(self.blocked_action_cooldowns):
            if cooldown > 0:
                self.blocked_action_cooldowns[idx] = cooldown - 1

        action = int(last_action)
        if not self.has_last_state or not (0 <= action < len(self.blocked_action_cooldowns)):
            return

        if self.last_hero_pos == hero_pos:
            self.blocked_this_step = 1
            self.total_blocked_count += 1
            self.blocked_action_cooldowns[action] = BLOCKED_ACTION_COOLDOWN

    def _select_safe_action(self, closest_monster_rel, min_monster_dist, map_info, legal_action):
        if closest_monster_rel is None or min_monster_dist > DANGER_PRIOR_DISTANCE:
            return -1, 0.0, 0.0, 0.0, 0.0, False, 0.0

        escape_dx = -closest_monster_rel[0]
        escape_dz = -closest_monster_rel[1]
        escape_norm = max(_distance(escape_dx, escape_dz), 1e-6)
        danger_level = 1.0 - _norm(min_monster_dist, DANGER_PRIOR_DISTANCE)

        best_move_action = -1
        best_move_score = -1e9
        best_move_path_len = 0
        best_flash_action = -1
        best_flash_score = -1e9
        best_flash_path_len = 0
        best_flash_trap_risk = 1.0
        for action, move in ACTION_TO_VECTOR.items():
            if action >= len(legal_action) or not legal_action[action]:
                continue

            move_dx, move_dz = move
            score, path_len, trap_risk = self._score_escape_action(
                map_info=map_info,
                action=action,
                move_dx=move_dx,
                move_dz=move_dz,
                closest_monster_rel=closest_monster_rel,
                min_monster_dist=min_monster_dist,
                escape_dx=escape_dx,
                escape_dz=escape_dz,
                escape_norm=escape_norm,
                danger_level=danger_level,
            )
            if action < MOVE_ACTION_NUM:
                if score > best_move_score:
                    best_move_score = score
                    best_move_action = action
                    best_move_path_len = path_len
            else:
                if score > best_flash_score:
                    best_flash_score = score
                    best_flash_action = action
                    best_flash_path_len = path_len
                    best_flash_trap_risk = trap_risk

        if best_move_action < 0:
            return (
                best_flash_action,
                danger_level,
                best_flash_score,
                best_flash_path_len,
                best_flash_score,
                True,
                best_flash_trap_risk,
            )

        if best_flash_action >= 0 and self._should_use_flash_prior(
            danger_level=danger_level,
            best_move_score=best_move_score,
            best_flash_score=best_flash_score,
            best_flash_path_len=best_flash_path_len,
            best_flash_trap_risk=best_flash_trap_risk,
        ):
            return (
                best_flash_action,
                danger_level,
                best_flash_score,
                best_flash_path_len,
                best_flash_score - best_move_score,
                True,
                best_flash_trap_risk,
            )

        move_margin = best_move_score - best_flash_score if best_flash_action >= 0 else best_move_score
        return best_move_action, danger_level, best_move_score, best_move_path_len, move_margin, False, 0.0

    def _should_use_flash_prior(
        self, danger_level, best_move_score, best_flash_score, best_flash_path_len, best_flash_trap_risk
    ):
        if best_flash_score <= -1e8:
            return False
        if best_flash_trap_risk >= 0.9:
            return False
        if best_flash_path_len < 1:
            return False
        if danger_level >= 0.9:
            return best_flash_trap_risk <= 0.65 and best_flash_score >= best_move_score - 0.15
        if danger_level >= 0.85:
            return best_flash_trap_risk <= 0.55 and best_flash_score >= best_move_score - 0.05
        if danger_level >= 0.65:
            return best_flash_trap_risk <= 0.45 and best_flash_score >= best_move_score + 0.2
        return best_flash_trap_risk <= 0.25 and best_flash_score >= best_move_score + 1.1

    def _score_escape_action(
        self,
        map_info,
        action,
        move_dx,
        move_dz,
        closest_monster_rel,
        min_monster_dist,
        escape_dx,
        escape_dz,
        escape_norm,
        danger_level,
    ):
        is_flash = action >= MOVE_ACTION_NUM
        move_norm = max(_distance(move_dx, move_dz), 1e-6)
        escape_alignment = (move_dx * escape_dx + move_dz * escape_dz) / (move_norm * escape_norm)

        after_dx = closest_monster_rel[0] - move_dx
        after_dz = closest_monster_rel[1] - move_dz
        dist_gain = _distance(after_dx, after_dz) - min_monster_dist

        path_len = self._escape_corridor_len(map_info, move_dx, move_dz, is_flash=is_flash)
        open_neighbors = self._local_open_count(map_info, move_dx, move_dz)
        edge_margin = self._landing_edge_margin(map_info, move_dx, move_dz)
        is_passable = self._is_adjacent_passable(map_info, move_dx, move_dz)

        blocked_cooldown = self.blocked_action_cooldowns[action]
        passable_score = 1.0 if is_passable else -6.0
        dead_end_penalty = 0.0 if path_len >= 2 else -(1.7 if is_flash else 1.5) * danger_level
        openness_penalty = 0.0
        if open_neighbors < 3:
            openness_penalty -= (0.24 if is_flash else 0.15) * (3 - open_neighbors) * (0.5 + danger_level)
        edge_penalty = 0.0
        if edge_margin < 2:
            edge_penalty -= (0.3 if is_flash else 0.12) * (2 - edge_margin) * (0.5 + danger_level)
        flash_penalty = -2.2 if is_flash and danger_level < FLASH_LOW_DANGER_THRESHOLD else 0.0
        if is_flash and dist_gain < 2.0:
            flash_penalty -= 0.18 * (2.0 - dist_gain)
        flash_bonus = (0.65 + 0.28 * max(path_len - 1, 0.0)) * danger_level if is_flash else 0.0
        gain_weight = 1.95 if is_flash else 1.4
        path_weight = 0.38 if is_flash else 0.45
        open_weight = 0.15 if is_flash else 0.12

        score = (
            gain_weight * dist_gain
            + 1.1 * escape_alignment
            + path_weight * path_len
            + open_weight * open_neighbors
            + passable_score
            + dead_end_penalty
            + openness_penalty
            + edge_penalty
            + flash_bonus
            + flash_penalty
            - (0.6 if is_flash else 0.55) * blocked_cooldown
        )
        trap_risk = 0.0
        if is_flash:
            trap_risk += 0.16 if path_len < 2 else 0.0
            trap_risk += 0.08 * max(0.0, 3.0 - float(open_neighbors))
            trap_risk += 0.09 * max(0.0, 2.0 - float(edge_margin))
            trap_risk += 0.08 * max(0.0, GOOD_FLASH_ESCAPE_GAIN - dist_gain) / GOOD_FLASH_ESCAPE_GAIN
            if danger_level >= 0.85 and dist_gain >= 4.0:
                trap_risk *= 0.85
            trap_risk = float(np.clip(trap_risk, 0.0, 1.0))
        return score, path_len, trap_risk

    def _escape_corridor_len(self, map_info, dx, dz, is_flash=False):
        if map_info is None or len(map_info) == 0:
            return ESCAPE_LOOKAHEAD_STEPS

        if is_flash:
            if not self._is_offset_passable(map_info, dx, dz):
                return 0
            corridor_len = 1
            step_dx = int(np.sign(dx))
            step_dz = int(np.sign(dz))
            if step_dx == 0 and step_dz == 0:
                return corridor_len
            for step in range(1, ESCAPE_LOOKAHEAD_STEPS + 1):
                if not self._is_offset_passable(map_info, dx + step_dx * step, dz + step_dz * step):
                    break
                corridor_len += 1
            return corridor_len

        corridor_len = 0
        for step in range(1, ESCAPE_LOOKAHEAD_STEPS + 1):
            if not self._is_offset_passable(map_info, dx * step, dz * step):
                break
            corridor_len += 1
        return corridor_len

    def _local_open_count(self, map_info, dx, dz):
        if map_info is None or len(map_info) == 0:
            return MOVE_ACTION_NUM

        open_count = 0
        for action in range(MOVE_ACTION_NUM):
            ndx, ndz = ACTION_TO_VECTOR[action]
            if self._is_offset_passable(map_info, dx + ndx, dz + ndz):
                open_count += 1
        return open_count

    def _is_adjacent_passable(self, map_info, dx, dz):
        return self._is_offset_passable(map_info, dx, dz)

    def _landing_edge_margin(self, map_info, dx, dz):
        if map_info is None or len(map_info) == 0:
            return 3.0

        center_row = len(map_info) // 2
        center_col = len(map_info[0]) // 2 if len(map_info[0]) > 0 else 0
        row = center_row + int(dz)
        col = center_col + int(dx)
        if row < 0 or row >= len(map_info) or col < 0 or col >= len(map_info[0]):
            return 0.0
        return float(min(row, col, len(map_info) - 1 - row, len(map_info[0]) - 1 - col))

    def _is_offset_passable(self, map_info, dx, dz):
        if map_info is None or len(map_info) == 0:
            return True
        center_row = len(map_info) // 2
        center_col = len(map_info[0]) // 2 if len(map_info[0]) > 0 else 0
        row = center_row + int(dz)
        col = center_col + int(dx)
        if row < 0 or row >= len(map_info) or col < 0 or col >= len(map_info[0]):
            return False
        return bool(map_info[row][col] != 0)

    def _calc_reward_and_metrics(
        self,
        env_info,
        hero,
        hero_pos,
        min_monster_dist,
        last_action,
        danger_level,
        safe_path_len,
        safe_trap_risk,
        flash_ready,
        has_buff,
        nearest_treasure_dist,
        nearest_buff_dist,
    ):
        treasure_count = int(hero.get("treasure_collected_count", env_info.get("treasures_collected", 0)))
        buff_count = int(env_info.get("collected_buff", 0))
        flash_count = int(env_info.get("flash_count", 0))
        flash_escape_gain = 0.0
        is_late_game = self.step_no >= LATE_GAME_STEP_THRESHOLD
        state_potential, potential_parts = self._calc_state_potential(
            min_monster_dist=min_monster_dist,
            danger_level=danger_level,
            safe_path_len=safe_path_len,
            safe_trap_risk=safe_trap_risk,
            flash_ready=flash_ready,
            has_buff=has_buff,
            nearest_treasure_dist=nearest_treasure_dist,
            nearest_buff_dist=nearest_buff_dist,
            is_late_game=is_late_game,
        )

        reward = 0.02
        if danger_level >= 0.25:
            self.danger_steps += 1
        if min_monster_dist <= 5.0:
            self.near_death_count += 1
        if is_late_game:
            self.late_game_steps += 1

        for milestone, bonus in SURVIVAL_MILESTONES.items():
            if self.step_no >= milestone and milestone not in self.survival_milestones:
                self.survival_milestones.add(milestone)
                reward += bonus

        if self.has_last_state:
            dist_delta = min_monster_dist - self.last_min_monster_dist
            last_danger_level = self._danger_from_dist(self.last_min_monster_dist)
            reward += POTENTIAL_GAMMA * state_potential - self.last_state_potential

            treasure_delta = max(0, treasure_count - self.last_treasure_count)
            reward += 1.2 * treasure_delta
            if treasure_delta > 0 and danger_level <= 0.45:
                reward += 0.15 * treasure_delta

            buff_delta = max(0, buff_count - self.last_buff_count)
            reward += (0.7 if min_monster_dist <= 25.0 else 0.5) * buff_delta

            if self.flash_review_steps > 0:
                self.flash_review_min_danger = min(self.flash_review_min_danger, danger_level)
                self.flash_review_peak_potential = max(self.flash_review_peak_potential, state_potential)
                if (
                    not self.blocked_this_step
                    and danger_level <= FLASH_CLEAR_DANGER_THRESHOLD
                    and min_monster_dist >= CLOSE_THREAT_DISTANCE
                ):
                    self.flash_review_clear_steps += 1
                if self.blocked_this_step or min_monster_dist <= 5.0:
                    self.flash_review_bad = True
                self.flash_review_steps -= 1
                if self.flash_review_steps == 0:
                    sustained_flash_gain = min_monster_dist - self.flash_review_origin_dist
                    flash_potential_gain = self.flash_review_peak_potential - self.flash_review_origin_potential
                    trap_condition = (
                        self.flash_review_bad
                        or (
                            sustained_flash_gain < BAD_FLASH_ESCAPE_GAIN
                            and flash_potential_gain < FLASH_TRAP_POTENTIAL_GAIN
                        )
                        or (
                            self.flash_review_min_danger > FLASH_TRAP_DANGER_THRESHOLD
                            and flash_potential_gain < FLASH_HOLD_POTENTIAL_GAIN
                        )
                    )
                    hold_condition = (
                        not trap_condition
                        and (
                            flash_potential_gain >= FLASH_HOLD_POTENTIAL_GAIN
                            or sustained_flash_gain >= (GOOD_FLASH_ESCAPE_GAIN - 1.0)
                        )
                        and (
                            self.flash_review_min_danger <= FLASH_CLEAR_DANGER_THRESHOLD
                            or self.flash_review_clear_steps >= 1
                        )
                    )
                    if trap_condition:
                        self.flash_trap_count += 1
                        reward -= 0.22 if is_late_game else 0.14
                    elif hold_condition:
                        self.flash_hold_count += 1
                        reward += 0.14 if is_late_game else 0.08

            flash_delta = max(0, flash_count - self.last_flash_count)
            if flash_delta > 0:
                flash_escape_gain = dist_delta
                self.flash_escape_gain_sum += flash_escape_gain
                got_flash_value = treasure_delta > 0 or buff_delta > 0
                flash_potential_delta = state_potential - self.last_state_potential
                low_danger_flash = last_danger_level < 0.12 and not got_flash_value
                if (
                    flash_escape_gain >= GOOD_FLASH_ESCAPE_GAIN
                    or (self.last_min_monster_dist <= CLOSE_THREAT_DISTANCE and flash_escape_gain >= 4.0)
                    or flash_potential_delta >= FLASH_HOLD_POTENTIAL_GAIN
                    or got_flash_value
                ):
                    self.good_flash_count += flash_delta
                    reward += 0.42 * flash_delta + 0.05 * float(np.clip(flash_escape_gain, 0.0, 10.0))
                    if is_late_game and flash_escape_gain >= GOOD_FLASH_ESCAPE_GAIN:
                        reward += 0.1 * flash_delta
                    if got_flash_value:
                        reward += 0.2
                elif (
                    flash_escape_gain <= BAD_FLASH_ESCAPE_GAIN
                    and flash_potential_delta < FLASH_TRAP_POTENTIAL_GAIN
                    and not got_flash_value
                ):
                    self.bad_flash_count += flash_delta
                    reward -= (0.24 if is_late_game else 0.18) * flash_delta
                elif low_danger_flash:
                    self.bad_flash_count += flash_delta
                    reward -= 0.12 * flash_delta

                self.flash_review_steps = FLASH_REVIEW_STEPS
                self.flash_review_origin_dist = self.last_min_monster_dist
                self.flash_review_origin_potential = self.last_state_potential
                self.flash_review_peak_potential = state_potential
                self.flash_review_bad = bool(self.blocked_this_step or min_monster_dist <= 5.0)
                self.flash_review_min_danger = danger_level
                self.flash_review_clear_steps = int(
                    not self.blocked_this_step
                    and danger_level <= FLASH_CLEAR_DANGER_THRESHOLD
                    and min_monster_dist >= CLOSE_THREAT_DISTANCE
                )

            if self.last_hero_pos == hero_pos and 0 <= int(last_action) < Config.ACTION_NUM:
                self.stuck_steps += 1
            else:
                self.stuck_steps = 0

            if self.stuck_steps >= 2:
                self.total_stuck_count += 1
                reward -= 0.08 * min(self.stuck_steps, 5)

            if self.blocked_this_step:
                reward -= 0.14 if danger_level >= 0.25 else 0.06

            if min_monster_dist <= 2.0:
                reward -= 0.2
            elif min_monster_dist <= 5.0:
                reward -= 0.08

        self.has_last_state = True
        self.last_min_monster_dist = min_monster_dist
        self.last_hero_pos = hero_pos
        self.last_treasure_count = treasure_count
        self.last_buff_count = buff_count
        self.last_flash_count = flash_count
        self.last_state_potential = state_potential

        metrics = {
            "min_monster_dist": float(min_monster_dist),
            "treasure_count": float(treasure_count),
            "buff_count": float(buff_count),
            "flash_count": float(flash_count),
            "stuck_count": float(self.total_stuck_count),
            "blocked_count": float(self.total_blocked_count),
            "blocked_this_step": float(self.blocked_this_step),
            "danger_steps": float(self.danger_steps),
            "near_death_count": float(self.near_death_count),
            "good_flash_count": float(self.good_flash_count),
            "bad_flash_count": float(self.bad_flash_count),
            "flash_escape_gain": float(self.flash_escape_gain_sum),
            "flash_trap_count": float(self.flash_trap_count),
            "flash_hold_count": float(self.flash_hold_count),
            "late_game_steps": float(self.late_game_steps),
            "total_score": float(env_info.get("total_score", 0.0)),
            "danger_level": float(danger_level),
            "state_potential": float(state_potential),
            "safety_potential": float(potential_parts["safety"]),
            "resource_potential": float(potential_parts["resource"]),
            "flash_potential": float(potential_parts["flash"]),
        }
        return [float(reward)], metrics

    def _calc_state_potential(
        self,
        min_monster_dist,
        danger_level,
        safe_path_len,
        safe_trap_risk,
        flash_ready,
        has_buff,
        nearest_treasure_dist,
        nearest_buff_dist,
        is_late_game,
    ):
        safety_potential = 0.58 * (1.0 - danger_level)
        safety_potential += 0.12 * float(np.clip(safe_path_len / max(1.0, ESCAPE_LOOKAHEAD_STEPS + 1), 0.0, 1.0))
        safety_potential -= 0.12 * float(np.clip(safe_trap_risk, 0.0, 1.0))
        if is_late_game and min_monster_dist >= LATE_GAME_SAFE_DISTANCE and danger_level <= 0.45:
            safety_potential += 0.08
        safety_potential = float(np.clip(safety_potential, 0.0, 0.85))

        resource_gate = float(np.clip((0.65 - danger_level) / 0.65, 0.0, 1.0))
        treasure_potential = 0.0
        if nearest_treasure_dist < MAX_MAP_DISTANCE:
            treasure_potential = 0.24 * resource_gate * (
                1.0 - _norm(nearest_treasure_dist, RESOURCE_POTENTIAL_DISTANCE)
            )
        buff_potential = 0.0
        if not has_buff and nearest_buff_dist < MAX_MAP_DISTANCE:
            buff_potential = 0.16 * resource_gate * (
                1.0 - _norm(nearest_buff_dist, BUFF_POTENTIAL_DISTANCE)
            )
        resource_potential = float(np.clip(treasure_potential + buff_potential, 0.0, 0.35))

        flash_potential = 0.0
        if flash_ready:
            flash_potential = 0.08 + 0.16 * danger_level
            if danger_level < 0.15:
                flash_potential *= 0.35
        flash_potential = float(np.clip(flash_potential, 0.0, 0.24))

        total_potential = float(
            np.clip(safety_potential + resource_potential + flash_potential, 0.0, 1.2)
        )
        return total_potential, {
            "safety": safety_potential,
            "resource": resource_potential,
            "flash": flash_potential,
        }

    def _danger_from_dist(self, min_monster_dist):
        if min_monster_dist > DANGER_PRIOR_DISTANCE:
            return 0.0
        return 1.0 - _norm(min_monster_dist, DANGER_PRIOR_DISTANCE)
