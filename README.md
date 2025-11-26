# ppo_reproduce PPO算法复现

本项目基于 Qwen2.5-0.5B 语言模型全流程复现DPO算法，loss函数设计+policy model训练。

## 🎯 项目概述
TODO(wangjintao): 项目概述

## 🔍 算法原理
DPO算法的解决的问题可以看示意图，就是把RLHF中参考reward model给去掉了，节省了reward model训练的开销，本质上来说DPO属于监督微调。
<p align="center">
    <img src="./docs/dpo-comparison.png">
</p>
但是需要注意DPO方法仅适用于对齐人类偏好，如果是用于训练思维链这种能力，还是需要RL来做，因为DPO中没有规则来评价模型输出结果中是否带思维链的一些指标。

DPO算法目标函数为：
<p align="center">
    <img src="./docs/dpo-loss_form.png">
</p>
DPO的数学推导比较复杂，详细的证明和推导可以看以下两份资料：

* 知乎文章：[DPO（Direct Preference Optimization）直接偏好优化](https://zhuanlan.zhihu.com/p/1959389690101735449)
* DPO原始论文：[Direct Preference Optimization](https://arxiv.org/pdf/2305.18290)

1. DPO方法的思想

    利用了建模人类偏好的Bradley-Terry模型数学形式上的一个trick，通过chosen和rejected之间的减法，能够消去奖励模型。带着消去奖励模型的最终目标去看公式推导，可以理清思路。

2. 为什么损失函数要加一层log函数

    为什么不直接用Bradley-Terry转换后的sigmoid函数作为损失。Bradley-Terry模型给出的是偏好概率，即选择某个偏好大于另一个的概率，DPO的目标是最大化这个概率。即需要最大化大量样本概率的乘积，而乘积难以直接优化。取 log 能把乘法变成加法，让推导可计算、可优化，并且不会改变最大值的位置。所以最大似然推导必然得到 log-likelihood，而不是原始概率。

### 公式与代码实现对应关系
TODO(wangjintao): 公式与代码对应关系

## 📚 数据集
数据集选用M-A-P团队的COIG-P，是一个百万级中文人工智能偏好训练数据集，数据列如下所示
<p align="center">
    <img src="./docs/COIG-P_dataset.png">
</p>
（项目中实际用的是上述偏好数据集中的一个子集做训练，并进行了一道映射）

## 📊 效果展示

### 运行环境
- **policy模型训练**: 2小时（1 × AutoDL vGPU-32GB）

### 训练效果
<p align="center">
    <img src="./docs/dpo-loss.png" width="60%">
</p>
<p align="center">
    <em>Figure 1: DPO 训练过程中的 loss 曲线</em>
</p>
loss值随着训练的进行持续在掉，说明模型持续学习到人类偏好。（为了节省时间，没有等到训练效果稳定就停了）

## 🚀 项目部署运行

### 模型和数据集下载

```bash
# 模型或数据有网络问题可以在modelscope或者hf镜像站下载

# 下载Qwen2.5-0.5B模型
git clone https://huggingface.co/Qwen/Qwen2.5-0.5B

# 偏好数据
git clone https://huggingface.co/datasets/m-a-p/COIG-P
```

### 依赖安装

```bash
pip install uv
uv sync
```

### 训练步骤

```bash
# train policy model
python dpo-train.py
accelerate launch train.py # 兼容accelerate分布式训练
```

## 📖 参考资料

1. [Direct Preference Optimization](https://arxiv.org/pdf/2305.18290)，论文中提出的思想非常巧妙地消去了reward模型，[训练过ppo算法](https://github.com/TeenLucifer/ppo_reproduce)对比可以明显感受到节省了训练reward模型和critic模型的资源，而且仅训练一个模型稳定性也大幅提升。

## 🤝 贡献与交流

欢迎提交Issue和Pull Request来改进项目。这是一个探索性的学习项目，旨在分享多模态模型训练和部署的经验。

## 📄 许可证

本项目采用开源许可证，详见LICENSE文件。

---

**注意**: 这是一个以探索和学习为目标的项目，代码实现可能存在不足之处。如果您发现任何问题，欢迎提出Issue或PR。感谢您的指正和交流！