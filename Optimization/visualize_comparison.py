#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
visualize_comparison.py
Hard vs Soft 边界处理可视化对比
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import json
from typing import Dict, Any, List, Tuple
import matplotlib

# 设置中文字体（如果需显示中文）
# matplotlib.rcParams['font.sans-serif'] = ['SimHei']
# matplotlib.rcParams['axes.unicode_minus'] = False

def visualize_comparison(comparison_result: Dict[str, Any], 
                         save_dir: str = "comparison_results"):
    """
    可视化对比结果
    """
    os.makedirs(save_dir, exist_ok=True)
    
    theta = comparison_result["theta"]
    hard = comparison_result["hard"]
    soft = comparison_result["soft"]
    target = comparison_result["target"]
    
    # 创建图形
    fig = plt.figure(figsize=(16, 12))
    
    # 1. 专家权重分布
    ax1 = plt.subplot(3, 3, 1)
    x_hard = np.arange(len(hard["expert_ids"]))
    x_soft = np.arange(len(soft["expert_ids"]))
    
    bars1 = ax1.bar(x_hard - 0.2, hard["weights"], width=0.4, 
                   label=f'Hard (Top{len(hard["expert_ids"])})', alpha=0.7)
    bars2 = ax1.bar(x_soft + 0.2, soft["weights"], width=0.4, 
                   label='Soft (GMM)', alpha=0.7)
    
    ax1.set_xlabel('Expert Index')
    ax1.set_ylabel('Weight')
    ax1.set_title('Expert Weight Distribution')
    ax1.set_xticks(np.arange(max(len(hard["expert_ids"]), len(soft["expert_ids"]))))
    ax1.set_xticklabels([f'E{id}' for id in range(1, max(len(hard["expert_ids"]), len(soft["expert_ids"])) + 1)])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 添加权重值标签
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    # 2. 预测与目标对比（折线图）
    ax2 = plt.subplot(3, 3, 2)
    feature_indices = np.arange(len(target))
    
    ax2.plot(feature_indices, target, 'ko-', label='Target', linewidth=2, markersize=8)
    ax2.plot(feature_indices, hard["prediction"], 'bs--', label=f'Hard (MSE={hard["error"]:.2e})', 
            alpha=0.7, markersize=6)
    ax2.plot(feature_indices, soft["prediction"], 'r^--', label=f'Soft (MSE={soft["error"]:.2e})', 
            alpha=0.7, markersize=6)
    
    ax2.set_xlabel('Feature Dimension')
    ax2.set_ylabel('Value')
    ax2.set_title('Predictions vs Target')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 误差对比
    ax3 = plt.subplot(3, 3, 3)
    errors = [hard["error"], soft["error"]]
    methods = ['Hard TopK', 'Soft GMM']
    colors = ['blue', 'red']
    
    bars = ax3.bar(methods, errors, color=colors, alpha=0.7)
    ax3.set_ylabel('Mean Squared Error')
    ax3.set_title('Prediction Error Comparison')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 添加误差值标签
    for bar, error in zip(bars, errors):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height * 1.05,
                f'{error:.2e}', ha='center', va='bottom', fontsize=9)
    
    # 4. 参数空间可视化（n vs eta）
    ax4 = plt.subplot(3, 3, 4)
    
    # 显示当前参数点
    ax4.scatter(theta[0], theta[1], s=200, c='red', marker='*', 
               label=f'Current: n={theta[0]:.3f}, η={theta[1]:.3f}')
    
    # 标记使用的专家
    for i, (cid, _, weight) in enumerate(hard["individual"]):
        ax4.text(theta[0] + 0.02, theta[1] - i*0.1, f'H{cid}: {weight:.2f}', 
                fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor="blue", alpha=0.3))
    
    for i, (cid, _, weight) in enumerate(soft["individual"]):
        if weight > 0.05:  # 只显示权重较大的
            ax4.text(theta[0] - 0.1, theta[1] + 0.05 + i*0.08, f'S{cid}: {weight:.2f}', 
                    fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.3))
    
    ax4.set_xlabel('n (Flow Index)')
    ax4.set_ylabel('η (Viscosity)')
    ax4.set_title('Parameter Space & Expert Assignment')
    ax4.set_xlim([0.25, 1.05])
    ax4.set_ylim([0.0005, 350])
    ax4.set_yscale('log')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 参数空间可视化（sigma_y vs eta）
    ax5 = plt.subplot(3, 3, 5)
    ax5.scatter(theta[2], theta[1], s=200, c='red', marker='*')
    ax5.set_xlabel('σ_Y (Yield Stress)')
    ax5.set_ylabel('η (Viscosity)')
    ax5.set_title('Yield Stress vs Viscosity')
    ax5.set_xlim([-10, 410])
    ax5.set_ylim([0.0005, 350])
    ax5.set_yscale('log')
    ax5.grid(True, alpha=0.3)
    
    # 6. 特征空间可视化（PCA降维，如果特征维度高）
    ax6 = plt.subplot(3, 3, 6)
    phi = comparison_result["phi"]
    
    # 简化显示：使用前两个主要特征维度
    if len(phi) >= 2:
        ax6.scatter(phi[0], phi[1], s=150, c='green', marker='o', label='Feature Vector')
        ax6.set_xlabel('Feature Dim 1')
        ax6.set_ylabel('Feature Dim 2')
        ax6.set_title('Feature Space (First 2 Dims)')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    # 7. 各专家预测偏差热图
    ax7 = plt.subplot(3, 3, (7, 9))
    
    # 收集所有参与预测的专家
    all_experts = set()
    if "individual" in hard:
        for cid, _, _ in hard["individual"]:
            all_experts.add(('hard', cid))
    if "individual" in soft:
        for cid, _, _ in soft["individual"]:
            all_experts.add(('soft', cid))
    
    all_experts = sorted(list(all_experts))
    
    # 计算每个专家在每个特征维度上的绝对误差
    n_features = len(target)
    error_matrix = np.zeros((len(all_experts), n_features))
    
    for i, (method, cid) in enumerate(all_experts):
        # 找到该专家的预测
        pred = None
        if method == 'hard':
            for expert_cid, expert_pred, _ in hard.get("individual", []):
                if expert_cid == cid:
                    pred = expert_pred
                    break
        else:
            for expert_cid, expert_pred, _ in soft.get("individual", []):
                if expert_cid == cid:
                    pred = expert_pred
                    break
        
        if pred is not None:
            error_matrix[i, :] = np.abs(pred - target)
    
    # 绘制热图
    im = ax7.imshow(error_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax7.set_xlabel('Feature Dimension')
    ax7.set_ylabel('Expert')
    ax7.set_title('Absolute Error by Expert and Feature Dimension')
    
    # 设置刻度标签
    ax7.set_xticks(np.arange(n_features))
    ax7.set_xticklabels([f'F{i+1}' for i in range(n_features)], rotation=45)
    ax7.set_yticks(np.arange(len(all_experts)))
    ax7.set_yticklabels([f'{method[0].upper()}{cid}' for method, cid in all_experts])
    
    # 添加颜色条
    plt.colorbar(im, ax=ax7, label='Absolute Error')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图形
    import datetime
    timestamp = datetime.datetime.now().strftime("%H%M%S")
    save_path = os.path.join(save_dir, f'comparison_{timestamp}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {save_path}")
    
    return save_path


def create_summary_report(comparison_results: List[Dict[str, Any]], 
                         save_dir: str = "comparison_results") -> Dict[str, Any]:
    """
    创建多个对比结果的汇总报告
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 收集统计数据
    hard_errors = []
    soft_errors = []
    improvement_ratios = []
    
    for result in comparison_results:
        hard_errors.append(result["hard"]["error"])
        soft_errors.append(result["soft"]["error"])
        if result["hard"]["error"] > 0:
            ratio = result["hard"]["error"] / max(result["soft"]["error"], 1e-10)
            improvement_ratios.append(ratio)
    
    # 创建汇总图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 误差分布箱线图
    ax1 = axes[0, 0]
    data = [hard_errors, soft_errors]
    bp = ax1.boxplot(data, labels=['Hard TopK', 'Soft GMM'], 
                    patch_artist=True)
    
    # 设置颜色
    colors = ['lightblue', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax1.set_ylabel('MSE (log scale)')
    ax1.set_title('Error Distribution Comparison')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. 改进比例直方图
    ax2 = axes[0, 1]
    if improvement_ratios:
        ax2.hist(improvement_ratios, bins=15, alpha=0.7, color='green', edgecolor='black')
        ax2.axvline(1.0, color='red', linestyle='--', label='No Improvement')
        ax2.set_xlabel('Improvement Ratio (Hard MSE / Soft MSE)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Improvement Ratio Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. 平均误差对比
    ax3 = axes[1, 0]
    mean_hard = np.mean(hard_errors) if hard_errors else 0
    mean_soft = np.mean(soft_errors) if soft_errors else 0
    
    bars = ax3.bar(['Hard TopK', 'Soft GMM'], [mean_hard, mean_soft], 
                  color=['blue', 'red'], alpha=0.7)
    ax3.set_ylabel('Mean MSE (log scale)')
    ax3.set_title('Average Error Comparison')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, value in zip(bars, [mean_hard, mean_soft]):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height * 1.05,
                f'{value:.2e}', ha='center', va='bottom', fontsize=10)
    
    # 4. 胜率统计
    ax4 = axes[1, 1]
    if len(hard_errors) > 0:
        soft_wins = sum(1 for h, s in zip(hard_errors, soft_errors) if s < h)
        hard_wins = sum(1 for h, s in zip(hard_errors, soft_errors) if h < s)
        ties = len(hard_errors) - soft_wins - hard_wins
        
        labels = ['Soft Wins', 'Hard Wins', 'Ties']
        sizes = [soft_wins, hard_wins, ties]
        colors = ['lightcoral', 'lightblue', 'lightgray']
        
        ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
               startangle=90, shadow=True)
        ax4.set_title('Win Rate Comparison')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'summary_report.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 保存统计数据到文件
    stats = {
        'n_samples': len(comparison_results),
        'hard_mean_error': float(mean_hard),
        'soft_mean_error': float(mean_soft),
        'mean_improvement_ratio': float(np.mean(improvement_ratios)) if improvement_ratios else 0,
        'soft_win_rate': float(soft_wins) / len(comparison_results) if len(comparison_results) > 0 else 0
    }
    
    stats_path = os.path.join(save_dir, 'statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Summary report saved to: {save_path}")
    print(f"Statistics saved to: {stats_path}")
    
    return stats