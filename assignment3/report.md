# 基于LSTM的污染浓度预测实验报告

## 1. 问题描述
针对空气质量监测场景，建立多变量时间序列预测模型，使用历史气象数据（温度、湿度、风速等）和污染物浓度数据，预测未来1小时的PM2.5浓度值。主要挑战包括：
- 时间序列的长期依赖建模
- 多变量特征间的复杂关系捕捉
- 传感器数据中的噪声和缺失值处理

## 2. 概要
本实验构建了双层LSTM神经网络模型，使用过去24小时的8维特征数据（包含气象指标和历史浓度值），预测未来1小时的PM2.5浓度。

## 3. LSTM结构原理

LSTM单元通过三个门控机制实现记忆控制：

​**遗忘门**​：
$$ f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) $$

​**输入门**​：
$$
i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \\
\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)
$$

​**细胞状态更新**​：
$$ C_t = f_t \circ C_{t-1} + i_t \circ \tilde{C}_t $$

​**输出门**​：
$$
o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \\
h_t = o_t \circ \tanh(C_t)
$$

其中：
- $\sigma$为sigmoid激活函数
- $\circ$表示Hadamard积
- $W$为权重矩阵，$b$为偏置项



## 4. 实验配置
### 4.1 模型参数
| 参数层         | 配置说明                  | 参数值       |
|----------------|-------------------------|-------------|
| 输入层         | 时间步长 × 特征维度       | 24×13       |
| LSTM1          | 隐藏单元数 + Return Seq | 64 + True   |
| Dropout1       | 随机失活率               | 0.2         |
| LSTM2          | 隐藏单元数               | 32          |
| Dropout2       | 随机失活率               | 0.2         |
| 输出层         | 全连接层                 | 1 neuron    |

### 4.2 训练参数
```python
{
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "batch_size": 72,
    "epochs": 100,
    "early_stop": {
        "monitor": "val_loss",
        "patience": 5
    },
    "loss": "MSE"
}
