#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_boundary_comparison.py
测试边界处理方法的性能对比
"""

import numpy as np
import os
import sys
import torch
import joblib
import json
from typing import List, Dict, Any
from tqdm import tqdm
import matplotlib.pyplot as plt

# 导入您的模块
sys.path.append('.')
from CMAEStest1_2 import load_expert_bundle, build_phi, predict_expert
from soft_interpolate import compare_hard_vs_soft
from visualize_comparison import visualize_comparison, create_summary_report

def load_all_experts(moe_dir: str, device: torch.device, max_experts: int = 100) -> Dict[int, Any]:
    """加载所有专家模型"""
    expert_cache = {}
    for cid in range(1, max_experts + 1):
        try:
            expert_cache[cid] = load_expert_bundle(
                os.path.join(moe_dir, f"expert_{cid}.pt"), device
            )
            print(f"Loaded expert {cid}")
        except Exception as e:
            print(f"Failed to load expert {cid}: {e}")
    return expert_cache

def generate_test_points(n_samples: int = 20) -> List[Dict[str, Any]]:
    """生成测试点（参数和几何条件）"""
    test_points = []
    
    for i in range(n_samples):
        # 随机生成参数（在合理范围内）
        n = np.random.uniform(0.3, 1.0)
        eta = 10 ** np.random.uniform(-3, np.log10(300))
        sigma_y = np.random.uniform(0, 400)
        
        # 随机生成几何条件
        W = np.random.uniform(2, 7)
        H = np.random.uniform(2, 7)
        
        # 模拟目标观测值（这里需要根据您的实际情况）
        # 这里使用一个简单的模拟，实际中应该使用您的模拟器
        y_target = np.random.randn(8) * 0.1 + np.array([i/10 for i in range(8)])
        
        test_points.append({
            'theta': [n, eta, sigma_y],
            'W': W,
            'H': H,
            'y_target': y_target,
            'id': i
        })
    
    return test_points

def main():
    # 配置
    moe_dir = "moe_workspace5"  # 修改为您的模型目录
    output_dir = "boundary_comparison_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载模型
    print("Loading GMM gate...")
    gate_dict = joblib.load(os.path.join(moe_dir, "gmm_gate.joblib"))
    
    print("Loading expert models...")
    expert_bundles = load_all_experts(moe_dir, device, max_experts=100)
    
    print("Generating test points...")
    test_points = generate_test_points(n_samples=30)
    
    # 进行对比测试
    comparison_results = []
    
    for test_point in tqdm(test_points, desc="Testing points"):
        try:
            result = compare_hard_vs_soft(
                theta=test_point['theta'],
                gate_dict=gate_dict,
                expert_bundles=expert_bundles,
                y_target=test_point['y_target'],
                W=test_point['W'],
                H=test_point['H'],
                device=device,
                topk_hard=2,
                confidence_threshold=0.01
            )
            
            # 添加测试点信息
            result['test_id'] = test_point['id']
            comparison_results.append(result)
            
            # 为每个测试点生成可视化
            vis_path = visualize_comparison(result, save_dir=output_dir)
            print(f"  Test {test_point['id']}: Hard MSE={result['hard']['error']:.2e}, "
                  f"Soft MSE={result['soft']['error']:.2e}")
            
        except Exception as e:
            print(f"Error on test point {test_point['id']}: {e}")
    
    # 生成汇总报告
    print("\nGenerating summary report...")
    stats = create_summary_report(comparison_results, save_dir=output_dir)
    
    # 保存详细结果
    with open(os.path.join(output_dir, 'detailed_results.json'), 'w') as f:
        # 简化结果以便保存
        simplified = []
        for r in comparison_results:
            simplified.append({
                'test_id': r['test_id'],
                'theta': r['theta'],
                'hard_error': float(r['hard']['error']),
                'soft_error': float(r['soft']['error']),
                'hard_experts': r['hard']['expert_ids'],
                'soft_experts': r['soft']['expert_ids'],
                'soft_weights': [float(w) for w in r['soft']['weights']]
            })
        json.dump(simplified, f, indent=2)
    
    print(f"\nAll results saved to: {output_dir}")
    print(f"Statistics:")
    print(f"  Total samples: {stats['n_samples']}")
    print(f"  Hard mean error: {stats['hard_mean_error']:.2e}")
    print(f"  Soft mean error: {stats['soft_mean_error']:.2e}")
    print(f"  Mean improvement ratio: {stats['mean_improvement_ratio']:.2f}")
    print(f"  Soft win rate: {stats['soft_win_rate']:.1%}")
    
    # 判断哪种方法更好
    if stats['soft_win_rate'] > 0.6:
        print("\nConclusion: Soft GMM interpolation performs BETTER than Hard TopK")
    elif stats['soft_win_rate'] < 0.4:
        print("\nConclusion: Hard TopK performs BETTER than Soft GMM interpolation")
    else:
        print("\nConclusion: Both methods perform SIMILARLY")

if __name__ == "__main__":
    main()