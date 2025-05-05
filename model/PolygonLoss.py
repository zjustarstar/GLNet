import cv2
import torch
import numpy as np
from scipy.spatial.distance import directed_hausdorff

def polygon_loss_with_gt(preds, targets, target_class, epsilon_factor=0.02, device="cuda"):
    """
    针对预测的窗户轮廓与标注轮廓，计算形状差异损失。

    Args:
        preds (torch.Tensor): 模型预测的类别结果，[batch_size, height, width]。
        targets (torch.Tensor): 标注的类别结果，[batch_size, height, width]。
        target_class (int): 窗户类的类别索引。
        epsilon_factor (float): 多边形拟合的精度因子。
        device (str): 使用的设备。

    Returns:
        torch.Tensor: 轮廓形状损失。
    """
    batch_size, height, width = preds.shape
    total_loss = 0.0

    for b in range(batch_size):
        # 提取窗户的预测和标注二值掩码
        pred_mask = (preds[b] == target_class).cpu().numpy().astype(np.uint8)
        target_mask = (targets[b] == target_class).cpu().numpy().astype(np.uint8)

        # 提取轮廓
        _,pred_contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        _,target_contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 初始化当前样本的损失
        sample_loss = 0.0

        for pred_contour, target_contour in zip(pred_contours, target_contours):
            # 对预测和标注的轮廓拟合多边形
            epsilon_pred = epsilon_factor * cv2.arcLength(pred_contour, True)
            approx_pred = cv2.approxPolyDP(pred_contour, epsilon_pred, True)

            epsilon_target = epsilon_factor * cv2.arcLength(target_contour, True)
            approx_target = cv2.approxPolyDP(target_contour, epsilon_target, True)

            # # 损失1：顶点数差异（越接近4越好）
            # vertex_diff = abs(len(approx_pred) - 4)

            # 损失2：Hausdorff距离
            hausdorff_dist = max(
                directed_hausdorff(approx_pred[:, 0, :], approx_target[:, 0, :])[0],
                directed_hausdorff(approx_target[:, 0, :], approx_pred[:, 0, :])[0]
            )

            # 损失3：轮廓点的平均距离
            avg_point_dist = np.mean(
                [np.linalg.norm(pt1 - pt2) for pt1, pt2 in zip(approx_pred[:, 0, :], approx_target[:, 0, :])]
            )

            # 综合损失
            sample_loss +=  hausdorff_dist + avg_point_dist

        # 累计样本损失
        total_loss += sample_loss

    # 归一化损失
    total_loss = torch.tensor(total_loss / batch_size, device=device, requires_grad=True)

    return total_loss
