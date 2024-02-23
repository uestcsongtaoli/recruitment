## 八股

**1. Python 多进程与多线程 GIL锁**

## AI

### LoRA

Low-Rank Adaptation。微调模型的过程，可以看成在训练好的权重 $W$ 上加一个 $\Delta W$。
全量的训练需要大量的资源，所以对 $W$ 进行低秩分解例如将 $d \times d$ 的矩阵分解成大小分别为 $d \times r$ 和 $r \times d$ 的 $A B$ 两个矩阵，这样达到了领域数据微调的目的，同时也减少了计算量。

缺点

- 灾难性遗忘
- 生成不可控

资料

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Understanding LoRA](https://towardsdatascience.com/understanding-lora-low-rank-adaptation-for-finetuning-large-models-936bce1a07c6)
- [代码](https://github.com/microsoft/LoRA/blob/main/loralib/layers.py)

### BPE

Byte-Pair Encoding，一种基于subword的分词技术

背景：[BPE是一种数据压缩算法](https://en.wikipedia.org/wiki/Byte_pair_encoding)，其中出现频次最高的一对连续数据字节被替换为该数据中不存在的字节

核心思想：确保最常见的单词在词汇表中表示为单个token，而罕见的单词则被分解为两个或多个sub-token，这与基于subword的tokenize算法的做法一致

从base vocabulary(英文为字母表)开始，不断在语料库中找词频最高、且连续的subword合并，直到达到目标词数，流程如下:

1. pre-tokenizer 将训练语料切分成words，英文可以直接用空格，中文可以分词来拆分
2. words末尾添加额外标记如\</w>，统计语料库词频。设定词表大小超参数
3. 根据词频得到单个字符（也就是构成words的基本词）的频率表，选择最高频的字符合并成新的subword，更新字符频率表
4. 重复步骤3直至subword数量达词表大小

资料

- [Byte-Pair Encoding: Subword-based tokenization algorithm](https://towardsdatascience.com/byte-pair-encoding-subword-based-tokenization-algorithm-77828a70bee0)
- [Byte-Pair Encoding (BPE)](https://huggingface.co/docs/transformers/tokenizer_summary#byte-pair-encoding-bpe)
- [深入理解NLP Subword算法：BPE、WordPiece、ULM](https://zhuanlan.zhihu.com/p/86965595)
- 实现 [minbpe](https://github.com/karpathy/minbpe/tree/master)
- 论文 [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)

### RoBERTa

与Bert相比的修改

1. next sentence prediction-> full sentence
2. Static Masking->Dynamic Masking
3. byte-level BPE
4. 更多数据、更大batch

资料

- [BERT、ALBERT、RoBerta、ERNIE模型对比和改进点总结](https://zhuanlan.zhihu.com/p/347846720)
- 论文 [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)

### 位置编码

为什么要加位置编码？

[Transform详解(超详细) Attention is all you need论文](https://zhuanlan.zhihu.com/p/63191028)
![attention 计算过程](./images/attention%E8%AE%A1%E7%AE%97%E8%BF%87%E7%A8%8B.jpg)

如果我们调整句子中词的顺序，交换Key1和Key2的位置，对应的Value1和Value2的位置也交互，最终的Attention Value值未发生变化。故无法捕捉顺序信息。

Attention利用的是全局信息，交换两个词的顺序，计算出来的 Attention权重仍然相同

#### 绝对位置编码

在输入的第 $k$ 个向量 $\boldsymbol{x}_k$ 中加入位置向量 $\boldsymbol{p}_k$ 变为 $\boldsymbol{x}_k+\boldsymbol{p}_k$ ，其中 $\boldsymbol{p}_k$ 只依赖于位置编号 $k$

##### 训练式

将位置编码当作可训练参数，比如最大长度为512，编码维度为768，那么就初始化一个 $512\times 768$ 的矩阵作为位置向量，让它随着训练过程更新

##### 三角式

一般也称为Sinusoidal位置编码

$$
\begin{equation}\left\{\begin{aligned}  & \boldsymbol{p}_{k,2i}=\sin\Big(k/10000^{2i/d}\Big)    \\
      & \boldsymbol{p}_{k, 2i+1}=\cos\Big(k/10000^{2i/d}\Big)
  \end{aligned}\right.\end{equation}
$$

其中 $\boldsymbol{p}_{k,2i} ,\boldsymbol{p}_{k, 2i+1}$ 分别是位置 $k$ 的编码向量的第 $2i,2i+1$ 个分量，$d$ 是位置向量的维度。

三角函数式位置编码的特点有显式的生成规律

#### 相对位置编码

相对位置并没有完整建模每个输入的位置信息，而是在算Attention的时候考虑当前位置与被Attention的位置的相对距离，由于自然语言一般更依赖于相对位置，所以相对位置编码通常也有着优秀的表现

让研究人员绞尽脑汁的Transformer位置编码-[相对位置编码](https://kexue.fm/archives/8130#%E7%9B%B8%E5%AF%B9%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81)

##### RoPE
>
>通过绝对位置编码的方式实现相对位置编码

Attention的核心运算是内积，所以我们希望的内积的结果带有相对位置信息，因此假设存在恒等关系

$$\begin{equation}\langle\boldsymbol{f}(\boldsymbol{q}, m), \boldsymbol{f}(\boldsymbol{k}, n)\rangle = g(\boldsymbol{q},\boldsymbol{k},m-n)\end{equation}
$$

推导结果是，将向量按两个一组旋转 $m\theta$，其中 $m$ 与代表词在句子中的绝对位置， $\theta$ 可用三角式的绝对位置编码函数，与向量内部的位置有关

![Implementation%20of%20Rotary%20Position%20Embedding(RoPE)](./images/Implementation%20of%20Rotary%20Position%20Embedding(RoPE).png)

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Transformer升级之路：2、博采众长的旋转式位置编码](https://kexue.fm/archives/8265)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- 【code】[应用RoPE到tensors中](https://github.com/bojone/bert4keras/blob/master/bert4keras/backend.py#L359)
- 【code】[BERTEmbeddings](https://github.com/NLPScott/pytorch-pretrained-BERT/blob/master/modeling.py#L128)
- 【code】[TF, Sinusoidal Positional_Encoding](https://github.com/Kyubyong/transformer/blob/master/modules.py#L259)
- 【code】[transformer 库 Sinusoidal Positional_Encoding](<https://github.com/SamLynnEvans/Transformer/blob/master/Embed.py>)
- 【code】[相对位置编码 NEZHA-PyTorch](https://github.com/huawei-noah/Pretrained-Language-Model/blob/master/NEZHA-PyTorch/modeling_nezha.py#L301)

### 交叉熵
