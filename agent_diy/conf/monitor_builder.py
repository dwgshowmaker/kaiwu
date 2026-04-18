#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Monitor panel configuration for the DIY Gorge Chase agent.
"""

from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder


def build_monitor():
    monitor = MonitorConfigBuilder()

    return (
        monitor.title("Gorge Chase DIY")
        .add_group(group_name="Algorithm", group_name_en="algorithm")
        .add_panel(name="Reward", name_en="reward", type="line")
        .add_metric(metrics_name="reward", expr="avg(reward{})")
        .end_panel()
        .add_panel(name="Total Loss", name_en="total_loss", type="line")
        .add_metric(metrics_name="total_loss", expr="avg(total_loss{})")
        .end_panel()
        .add_panel(name="Policy Loss", name_en="policy_loss", type="line")
        .add_metric(metrics_name="policy_loss", expr="avg(policy_loss{})")
        .end_panel()
        .add_panel(name="Value Loss", name_en="value_loss", type="line")
        .add_metric(metrics_name="value_loss", expr="avg(value_loss{})")
        .end_panel()
        .add_panel(name="Entropy Loss", name_en="entropy_loss", type="line")
        .add_metric(metrics_name="entropy_loss", expr="avg(entropy_loss{})")
        .end_panel()
        .end_group()
        .add_group(group_name="Episode", group_name_en="episode")
        .add_panel(name="Episode Steps", name_en="episode_steps", type="line")
        .add_metric(metrics_name="episode_steps", expr="avg(episode_steps{})")
        .end_panel()
        .add_panel(name="Total Score", name_en="total_score", type="line")
        .add_metric(metrics_name="total_score", expr="avg(total_score{})")
        .end_panel()
        .add_panel(name="Treasure Count", name_en="treasure_count", type="line")
        .add_metric(metrics_name="treasure_count", expr="avg(treasure_count{})")
        .end_panel()
        .add_panel(name="Buff Count", name_en="buff_count", type="line")
        .add_metric(metrics_name="buff_count", expr="avg(buff_count{})")
        .end_panel()
        .add_panel(name="Flash Count", name_en="flash_count", type="line")
        .add_metric(metrics_name="flash_count", expr="avg(flash_count{})")
        .end_panel()
        .add_panel(name="Result", name_en="result", type="line")
        .add_metric(metrics_name="result", expr="avg(result{})")
        .end_panel()
        .add_panel(name="Fail Rate", name_en="fail_rate", type="line")
        .add_metric(metrics_name="fail_rate", expr="avg(fail_rate{})")
        .end_panel()
        .add_panel(name="Avg Monster Dist", name_en="avg_min_monster_dist", type="line")
        .add_metric(metrics_name="avg_min_monster_dist", expr="avg(avg_min_monster_dist{})")
        .end_panel()
        .add_panel(name="Stuck Count", name_en="stuck_count", type="line")
        .add_metric(metrics_name="stuck_count", expr="avg(stuck_count{})")
        .end_panel()
        .end_group()
        .build()
    )
