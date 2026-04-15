#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder


def build_monitor():
    """
    # This function is used to create monitoring panel configurations for custom indicators.
    # 该函数用于创建自定义指标的监控面板配置。
    """
    monitor = MonitorConfigBuilder()

    config_dict = (
        monitor.title("峡谷追猎")
        .add_group(
            group_name="算法指标",
            group_name_en="algorithm",
        )
        .add_panel(
            name="累积回报",
            name_en="reward",
            type="line",
        )
        .add_metric(
            metrics_name="reward",
            expr="avg(reward{})",
        )
        .end_panel()
        .add_panel(
            name="总损失",
            name_en="total_loss",
            type="line",
        )
        .add_metric(
            metrics_name="total_loss",
            expr="avg(total_loss{})",
        )
        .end_panel()
        .add_panel(
            name="价值损失",
            name_en="value_loss",
            type="line",
        )
        .add_metric(
            metrics_name="value_loss",
            expr="avg(value_loss{})",
        )
        .end_panel()
        .add_panel(
            name="策略损失",
            name_en="policy_loss",
            type="line",
        )
        .add_metric(
            metrics_name="policy_loss",
            expr="avg(policy_loss{})",
        )
        .end_panel()
        .add_panel(
            name="熵损失",
            name_en="entropy_loss",
            type="line",
        )
        .add_metric(
            metrics_name="entropy_loss",
            expr="avg(entropy_loss{})",
        )
        .end_panel()
        .add_panel(
            name="局内步数",
            name_en="episode_steps",
            type="line",
        )
        .add_metric(
            metrics_name="episode_steps",
            expr="avg(episode_steps{})",
        )
        .end_panel()
        .add_panel(
            name="环境总分",
            name_en="total_score",
            type="line",
        )
        .add_metric(
            metrics_name="total_score",
            expr="avg(total_score{})",
        )
        .end_panel()
        .add_panel(
            name="宝箱数量",
            name_en="treasure_count",
            type="line",
        )
        .add_metric(
            metrics_name="treasure_count",
            expr="avg(treasure_count{})",
        )
        .end_panel()
        .add_panel(
            name="Buff数量",
            name_en="buff_count",
            type="line",
        )
        .add_metric(
            metrics_name="buff_count",
            expr="avg(buff_count{})",
        )
        .end_panel()
        .add_panel(
            name="闪现次数",
            name_en="flash_count",
            type="line",
        )
        .add_metric(
            metrics_name="flash_count",
            expr="avg(flash_count{})",
        )
        .end_panel()
        .add_panel(
            name="失败率",
            name_en="fail_rate",
            type="line",
        )
        .add_metric(
            metrics_name="fail_rate",
            expr="avg(fail_rate{})",
        )
        .end_panel()
        .add_panel(
            name="平均怪物距离",
            name_en="avg_min_monster_dist",
            type="line",
        )
        .add_metric(
            metrics_name="avg_min_monster_dist",
            expr="avg(avg_min_monster_dist{})",
        )
        .end_panel()
        .add_panel(
            name="卡住次数",
            name_en="stuck_count",
            type="line",
        )
        .add_metric(
            metrics_name="stuck_count",
            expr="avg(stuck_count{})",
        )
        .end_panel()
        .end_group()
        .build()
    )
    return config_dict
