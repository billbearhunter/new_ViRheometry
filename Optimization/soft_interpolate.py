#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
soft_interpolate.py
GMM Soft 补间预测模块 - 用于对比 Hard TopK 边界处理
"""

import numpy as np
import torch
import joblib
from typing import List, Tuple, Optional, Dict, Any

def build_phi(y: np.ndarray, W: float, H: float, eps: float = 1e-8) -> np.ndarray:
    """构建Gating特征向量"""
    y = np.asarray(y, dtype=float).reshape(1, -1)
    y_norm = y / (y[:, [-1]] + eps)
    feats = [
        y_norm,
        np.diff(y_norm, axis=1),
        np.log(np.abs(y[:, -1]) + eps).reshape(-1, 1)
    ]
    feats.append(np.hstack([
        np.log(np.sqrt(W*H) + eps).reshape(1, 1),
        np.log((W + eps) / (H + eps)).reshape(1, 1)
    ]))
    return np.hstack(feats)

def get_soft_weights(gate_dict: Dict, phi: np.ndarray, 
                     topk: Optional[int] = None,
                     confidence_threshold: float = 0.01) -> Tuple[List[int], np.ndarray]:
    """
    获取GMM软权重
    
    Args:
        gate_dict: 包含GMM和scaler的字典
        phi: 特征向量
        topk: 如果指定，只保留前k个专家
        confidence_threshold: 权重阈值，低于此值的专家将被排除
    
    Returns:
        expert_ids: 专家ID列表 (1-indexed)
        weights: 对应的权重（已归一化）
    """
    gmm = gate_dict["gmm"]
    scaler = gate_dict.get("scaler")
    
    # 标准化特征
    phi_in = phi.reshape(1, -1)
    if scaler is not None:
        phi_in = scaler.transform(phi_in)
    
    # 获取GMM概率
    probs = gmm.predict_proba(phi_in)[0]  # (n_clusters,)
    
    # 应用阈值
    mask = probs >= confidence_threshold
    if not np.any(mask):
        # 如果所有概率都低于阈值，使用top-1
        mask[np.argmax(probs)] = True
    
    # 获取符合条件的专家ID和权重
    all_indices = np.arange(len(probs))
    expert_indices = all_indices[mask]
    filtered_probs = probs[mask]
    
    # 可选：进一步限制到topk
    if topk is not None and len(filtered_probs) > topk:
        topk_idx = np.argsort(-filtered_probs)[:topk]
        expert_indices = expert_indices[topk_idx]
        filtered_probs = filtered_probs[topk_idx]
    
    # 转换为专家ID (1-indexed) 并归一化权重
    expert_ids = [int(idx) + 1 for idx in expert_indices]
    weights = filtered_probs / np.sum(filtered_probs)
    
    return expert_ids, weights

def soft_predict(theta: List[float], 
                 expert_bundles: Dict[int, Any],
                 expert_ids: List[int],
                 weights: np.ndarray,
                 W: float, H: float,
                 device: torch.device) -> np.ndarray:
    """
    软加权预测
    
    Args:
        theta: 参数向量 [n, eta, sigma_y]
        expert_bundles: 专家模型字典 {cid: ExpertBundle}
        expert_ids: 参与预测的专家ID列表
        weights: 对应的权重
        W, H: 几何参数
        device: 计算设备
    
    Returns:
        weighted_pred: 加权平均预测
        individual_preds: 各专家独立预测结果（用于分析）
    """
    n, eta, sigma_y = theta
    
    # 收集各专家预测
    individual_preds = []
    valid_experts = []
    valid_weights = []
    
    for cid, weight in zip(expert_ids, weights):
        if cid not in expert_bundles:
            continue
            
        bundle = expert_bundles[cid]
        
        # 使用现有预测函数（需从原代码中导入）
        try:
            pred = predict_expert(bundle, n, eta, sigma_y, W, H, device)
            individual_preds.append((cid, pred, weight))
            valid_experts.append(cid)
            valid_weights.append(weight)
        except Exception as e:
            print(f"Warning: Expert {cid} prediction failed: {e}")
            continue
    
    if not valid_experts:
        raise ValueError("No valid experts for prediction")
    
    # 重新归一化权重
    valid_weights = np.array(valid_weights)
    valid_weights = valid_weights / np.sum(valid_weights)
    
    # 计算加权平均
    weighted_pred = np.zeros_like(individual_preds[0][1])
    for (cid, pred, weight) in zip(valid_experts, [p[1] for p in individual_preds], valid_weights):
        weighted_pred += weight * pred
    
    return weighted_pred, individual_preds

def compare_hard_vs_soft(theta: List[float],
                         gate_dict: Dict,
                         expert_bundles: Dict[int, Any],
                         y_target: np.ndarray,
                         W: float, H: float,
                         device: torch.device,
                         topk_hard: int = 2,
                         confidence_threshold: float = 0.01) -> Dict[str, Any]:
    """
    对比 Hard TopK 和 Soft GMM 预测结果
    
    Returns:
        包含对比结果的字典
    """
    # 构建特征
    phi = build_phi(y_target, W, H)
    
    # 1. Hard TopK 预测
    hard_ids, hard_probs = route_target_dis_topk(gate_dict, phi, topk=topk_hard)
    hard_pred, hard_individual = soft_predict(
        theta, expert_bundles, hard_ids, 
        np.ones(len(hard_ids))/len(hard_ids),  # 等权重
        W, H, device
    )
    
    # 2. Soft GMM 预测
    soft_ids, soft_weights = get_soft_weights(
        gate_dict, phi, 
        topk=None,  # 使用所有满足阈值的专家
        confidence_threshold=confidence_threshold
    )
    soft_pred, soft_individual = soft_predict(
        theta, expert_bundles, soft_ids, soft_weights,
        W, H, device
    )
    
    # 计算误差
    hard_error = np.mean((hard_pred - y_target) ** 2)
    soft_error = np.mean((soft_pred - y_target) ** 2)
    
    return {
        "theta": theta,
        "hard": {
            "expert_ids": hard_ids,
            "weights": np.ones(len(hard_ids))/len(hard_ids),
            "prediction": hard_pred,
            "error": hard_error,
            "individual": hard_individual
        },
        "soft": {
            "expert_ids": soft_ids,
            "weights": soft_weights,
            "prediction": soft_pred,
            "error": soft_error,
            "individual": soft_individual
        },
        "target": y_target,
        "phi": phi.flatten()
    }