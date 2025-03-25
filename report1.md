# 线性回归三种优化算法原理

## 1. 最小二乘法（Ordinary Least Squares, OLS）

### 原理公式
通过最小化**残差平方和（RSS）​**求解参数：
$$ J(\theta) = \frac{1}{2m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2 $$
其中假设函数为线性组合：
$$ h_\theta(x) = \theta_0 + \theta_1x = \theta^T X $$

闭式解（正规方程）：
$$ \theta = (X^T X)^{-1} X^T y $$

### 核心步骤
1. 构造设计矩阵$X$（含偏置项$x_0=1$）
2. 计算$X^T X$的逆矩阵
3. 矩阵相乘求解参数$\theta$

### 特点
- ✅ 直接解析解，无需迭代  
- ❌ 计算复杂度$O(n^3)$，不适合高维数据  
- ❌ 矩阵不可逆时需增加正则化项  

---

## 2. 梯度下降法（Gradient Descent, GD）

### 原理公式
沿**负梯度方向**迭代更新参数：  
损失函数（均方误差）：
$$ J(\theta) = \frac{1}{2m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2 $$

参数更新公式：
$$ \theta_j := \theta_j - \alpha \frac{\partial J(\theta)}{\partial \theta_j} $$
展开式为：
$$ \theta_j := \theta_j - \frac{\alpha}{m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)} $$

矩阵形式批量更新：
$$ \theta := \theta - \frac{\alpha}{m} X^T (X\theta - y) $$

### 核心步骤
1. 初始化参数$\theta$和学习率$\alpha$
2. 循环计算梯度并更新参数直至收敛
3. 判定收敛条件（如误差变化<阈值）

### 特点
- ✅ 适用于大规模数据  
- ❌ 需手动设置学习率$\alpha$  
- ❌ 可能陷入局部最优（凸函数可避免）  

---

## 3. 牛顿法（Newton's Method）

### 原理公式
利用**Hessian矩阵**加速收敛的二阶优化方法：  
损失函数的二阶泰勒展开：
$$ J(\theta) \approx J(\theta^{(t)}) + \nabla J(\theta^{(t)})^T (\theta - \theta^{(t)}) + \frac{1}{2} (\theta - \theta^{(t)})^T H(\theta^{(t)}) (\theta - \theta^{(t)}) $$

参数更新公式：
$$ \theta^{(t+1)} = \theta^{(t)} - H^{-1}(\theta^{(t)}) \nabla J(\theta^{(t)}) $$

对于线性回归问题，Hessian矩阵为：
$$ H = \frac{1}{m} X^T X $$

最终闭式解与OLS一致：
$$ \theta = (X^T X)^{-1} X^T y $$

### 核心步骤
1. 计算梯度$\nabla J(\theta)$和Hessian矩阵$H$
2. 求解线性方程组$H\Delta\theta = -\nabla J(\theta)$
3. 更新参数$\theta := \theta + \Delta\theta$

### 特点
- ✅ 二次收敛速度，迭代次数少  
- ❌ 需计算并存储Hessian矩阵（$O(n^2)$内存）  
- ❌ 矩阵病态时数值不稳定  

---

## 算法对比
| 维度           | 最小二乘法          | 梯度下降法          | 牛顿法              |
|----------------|---------------------|---------------------|---------------------|
| ​**收敛速度**   | 一次计算            | 线性收敛            | 二次收敛            |
| ​**内存消耗**   | $O(n^2)$            | $O(n)$              | $O(n^2)$            |
| ​**适用场景**   | 小规模数据（n<1e4） | 大规模数据          | 中规模数据          |
| ​**实现难度**   | 低                  | 中                  | 高                  |
