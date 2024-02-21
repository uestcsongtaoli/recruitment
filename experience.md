## 八股

****

## AI

**1. 请简要介绍 LoRA**

Low-Rank Adaptation。微调模型的过程，可以看成在训练好的权重 $W$ 上加一个 $\Delta W$。
全量的训练需要大量的资源，所以对 $W$ 进行低秩分解例如将 $d \times d$ 的矩阵分解成大小分别为 $d \times r$ 和 $r \times d$ 的 $A B$ 两个矩阵，这样达到了领域数据微调的目的，同时也减少了计算量。

缺点

- 灾难性遗忘
- 生成不可控

资料

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Understanding LoRA](https://towardsdatascience.com/understanding-lora-low-rank-adaptation-for-finetuning-large-models-936bce1a07c6)
- [代码](https://github.com/microsoft/LoRA/blob/main/loralib/layers.py)
