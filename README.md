# 健身姿态分析系统 - 手肘角度计算优化

## 项目简介
本项目使用 MediaPipe 实现实时人体姿态检测，并针对手肘角度计算进行了特殊优化，以解决在不同视角下角度计算不准确的问题。

## 核心问题
在使用 2D 摄像头进行人体姿态估计时，存在以下挑战：
1. 正面视角下，由于缺乏深度信息，角度计算严重失真
2. 侧面视角下，远离摄像头一侧的手臂检测不准确
3. 不同角度下的置信度难以量化

## 解决方案

### 1. 面部朝向检测
- 使用眼睛和鼻子的相对位置计算面部朝向角度
- 通过眼睛间距估算深度参考值
- 根据面部朝向角度动态调整补偿策略

### 2. 深度补偿算法

#### 2.1 投影比例补偿
- 使用肩宽作为标准参考尺度
- 计算上臂投影长度与肩宽的比例
- 根据解剖学标准比例(0.7:1)进行校正

#### 2.2 角度补偿策略
- 基础角度计算：使用2D投影点计算初始角度
- 面部朝向因子：根据面部角度归一化(0-1)
- 深度补偿：
  ```python
  compensation_factor = 1.0 / max(abs(np.cos(np.radians(face_angle))), 0.5)
  depth_factor = 1.0 + (projection_ratio - 1.0) * (1.0 - face_factor)
  compensated_angle = base_angle * compensation_factor * depth_factor
  ```

### 3. 解剖学约束
- 考虑人体关节活动范围限制
- 根据角度大小采用不同补偿策略：
  - 小于90度：使用比例补偿
  - 大于90度：使用加性补偿
- 最大补偿限制：45度

### 4. 可信度计算
基于以下因素计算角度可信度：
- 面部朝向角度
- 肢体在图像中的投影比例
- 与标准解剖学比例的偏差

可信度显示：
- 绿色 (>80%): 高可信度
- 橙色 (50-80%): 中等可信度
- 红色 (<50%): 低可信度

## 补偿效果
1. 正面视角（±45度）
   - 考虑深度补偿
   - 使用投影比例校正
   - 应用最大补偿限制

2. 侧面视角（>45度）
   - 保持原始角度计算
   - 提供高可信度显示
   - 远端手臂降低可信度

## 使用注意事项
1. 保持良好光照条件
2. 确保人体完整在画面中
3. 避免过于剧烈的运动
4. 注意观察可信度指示

## 未来优化方向
1. 引入时序滤波，减少角度抖动
2. 添加更多解剖学约束
3. 优化深度估算算法
4. 支持多人同时检测
