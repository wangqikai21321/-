# R-CNN 问题精读笔记

这份笔记围绕阅读 R-CNN 论文时最容易卡住的几个问题展开。它不是按论文段落逐句翻译，而是按“问题 -> 原理 -> 论文语境 -> 总结”的方式整理，方便复习和答辩。

## 1. 论文先解决什么问题

R-CNN 的目标是把 CNN 从“整图分类”推进到“目标检测”。分类任务只需要回答：

```text
这张图里主要是什么？
```

而检测任务要回答：

```text
图里有哪些物体？
每个物体属于什么类别？
每个物体在哪里？
```

这比分类多了一个核心困难：**定位**。

R-CNN 的基本策略不是让 CNN 直接从整张图回归所有框，而是采用“recognition using regions”的思路：

```text
先提出一批可能有物体的区域
再把每个区域当成一个小图像送进 CNN 识别
```

因此 R-CNN 的名字就是：

```text
Regions with CNN features
```

它的主流程可以概括为四步：

```text
输入图像
-> Selective Search 生成约 2000 个 region proposals
-> CNN 为每个 proposal 提取固定长度特征
-> 每类线性 SVM 判断该 proposal 是否属于该类
```

在这个主体流程之外，论文又加入 bbox regression 来修正定位误差。

一句话总结：

**R-CNN 的核心不是让 CNN 直接“看整图并吐出框”，而是先用区域候选给出可能位置，再用 CNN 特征和 SVM 判断每个区域是什么。**

## 2. 整篇论文的大概步骤

可以把论文方法读成一条完整流水线：

```text
1. ImageNet 预训练 CNN
2. 在检测数据上 fine-tune CNN
3. 用 fine-tuned CNN 提取 proposal 特征
4. 为每个类别训练一个线性 SVM
5. 为每个类别训练一个 bbox regressor
6. 测试时对新图像生成 proposals 并逐个分类
7. 用 NMS 去重，得到最终检测结果
```

这几个步骤各自解决的问题不同：

```text
ImageNet 预训练：解决检测数据太少，CNN 难以从零训练的问题
fine-tuning：让 CNN 从分类任务适配到 warped proposal 检测任务
feature extraction：把任意 proposal 变成固定长度的深度特征
SVM：学习每个类别的最终判别边界
hard negative mining：让 SVM 学会拒绝最容易混淆的背景区域
bbox regression：修正分类正确但框不准的问题
NMS：删除同一目标上的重复检测框
```

所以 R-CNN 是一个分阶段训练、分模块组合的系统，不是端到端一次训练完成的检测器。

## 3. R-CNN 的训练过程

R-CNN 的训练可以分成三个主要阶段。

### 3.1 第一阶段：CNN 预训练

作者先在 ILSVRC/ImageNet 分类数据上预训练 CNN。这个阶段只有图像级标签，没有检测框。

训练目标是普通图像分类：

```text
输入：整张图
输出：ImageNet 类别
```

这个阶段的作用是让 CNN 学到通用视觉表示，比如边缘、颜色、纹理、形状、部件和高层语义模式。

### 3.2 第二阶段：CNN detection fine-tuning

有了 ImageNet 预训练后，作者把 CNN 迁移到检测任务。

做法是：

```text
把每个 region proposal warp 到 227 x 227
把 ImageNet 的 1000 类分类层替换成 N+1 类
N 是目标类别数，额外 1 类是 background
继续用 SGD 训练 CNN
```

在 VOC 上：

```text
N = 20
输出层 = 21-way softmax
```

fine-tuning 的样本定义比较宽松：

```text
proposal 与某个 ground-truth box 的 IoU >= 0.5 -> 该类别正样本
其他 proposals -> background
```

这样做的原因是 CNN 参数很多，只用 ground-truth boxes 正样本太少，容易过拟合。把 IoU 较高的 proposals 也作为正样本，相当于加入很多 jittered examples，让 CNN 学到更鲁棒的检测领域特征。

### 3.3 第三阶段：训练类别专属 SVM

fine-tuning 之后，R-CNN 不直接使用 softmax 作为最终检测器，而是提取每个 proposal 的 CNN 特征，通常是 `fc7` 的 `4096` 维向量，然后为每个类别训练一个线性 SVM。

对 VOC 来说，就是训练 20 个二分类器：

```text
aeroplane vs not-aeroplane
bicycle vs not-bicycle
...
person vs not-person
...
tvmonitor vs not-tvmonitor
```

SVM 的样本定义更严格：

```text
正样本：该类别 ground-truth boxes
负样本：与该类别所有真实框 IoU < 0.3 的 proposals
灰区样本：忽略
```

并且 SVM 使用 hard negative mining，重点挖掘那些容易被误判为目标的背景区域。

### 3.4 第四阶段：训练 bbox regression

最后，作者为每个类别训练一个 bbox regressor，用来修正 proposal 的坐标。

它的输入是 proposal 的 CNN 特征，输出是四个几何变换量：

```text
tx, ty, tw, th
```

也就是让 proposal 的中心点和宽高更接近对应 ground-truth box。

这个模块解决的是定位问题，不解决分类问题。

## 4. R-CNN 的测试过程

测试时，R-CNN 对一张新图像执行以下步骤：

```text
1. 对整张图运行 Selective Search，生成约 2000 个 proposals
2. 对每个 proposal 加上下文 padding，并 warp 到 227 x 227
3. 把每个 warped proposal 输入 fine-tuned CNN
4. 提取 CNN 特征，例如 fc7 4096 维
5. 对每个类别，用对应 SVM 给这个 proposal 打分
6. 如果使用 bbox regression，则用对应类别回归器修正框
7. 对每个类别分别做 NMS
8. 输出最终检测框、类别和分数
```

不带 bbox regression 时：

```text
最终框 = Selective Search 原始 proposal
```

带 bbox regression 时：

```text
最终框 = 原 proposal 经过类别专属回归器修正后的框
```

测试流程的关键特点是：**每个 proposal 都要单独 warp、单独跑 CNN**。这带来了高精度，也带来了速度瓶颈。

## 5. 这一路在追的问题

我们一直在追一个核心问题：

**CNN 里的某个神经元为什么会对狗脸、猫脸、肉类纹理、文字、窗格这类东西敏感？**

这个问题背后其实包含三层理解：

1. `pool5 = 6 x 6 x 256` 到底表示什么。
2. `pool5 feature (3,3,13)` 的 top activations 是怎么来的。
3. 神经元的“语义敏感性”是如何通过训练形成的。

可以先给一个总答案：

CNN 是层层递进的。`conv1` 更靠近原始像素，常学到边缘、颜色和方向；`conv2/conv3` 会组合出纹理、角点和局部结构；`conv4/conv5` 会形成更复杂的部件或语义相关模式，比如脸部、轮子、文字、肉类纹理等。`pool5` 本身通常没有可学习参数，它主要是对 `conv5` 的响应做 max pooling。因此论文里可视化 `pool5 feature (3,3,13)`，本质上是在观察前面卷积层学出来的某个高层视觉模式。

## 6. `pool5 = 6 x 6 x 256` 应该怎么理解

`pool5` 可以想象成：

```text
256 张 6 x 6 的打分表
```

其中：

```text
256 = 特征通道数
6 x 6 = 空间位置
每个数 = 某个特征在某个位置的激活强度
```

更直观地说：

```text
通道 = 看什么
位置 = 在哪里
数值 = 有多像
```

例如：

```text
pool5[3, 3, 13] = 8.5
```

可以理解为：

```text
第 13 个高层特征检测器，在空间位置 (3,3) 对当前 proposal 的响应强度是 8.5。
```

这里容易误解的一点是：`6 x 6` 不是 36 种不同特征。它表示同一种特征在不同空间位置上的响应。真正更接近“特征种类”的维度是 `256` 个通道。

## 7. `pool5 feature (3,3,13)` 的 top 1-24 是什么

论文图里的 `pool5 feature: (3,3,13)` top 1-24，不是从一张图的 `3 x 3` 区域里找最大值。

它的操作是：

1. 固定住 `pool5` 中的一个神经元：空间位置 `(3,3)`，通道 `13`。
2. 把大量 region proposals 都送进 CNN。
3. 记录这个固定神经元对每个 proposal 的激活值。
4. 按激活值从高到低排序。
5. 展示让它响应最高的前 24 个 proposals。

也就是说，论文不是在问：

```text
一张图里哪个 pool5 位置最大？
```

而是在问：

```text
对固定的某个 pool5 unit，所有候选区域里哪些最能激活它？
```

如果这些 top regions 都像狗脸，就说明这个神经元对狗脸类模式比较敏感。如果都像文字，就说明它对文字类模式敏感。

这就是 R-CNN 论文中非参数可视化方法的核心：让神经元通过它最喜欢的真实图像区域“自己说话”。

## 8. 神经元为什么会变得敏感

这种“敏感”不是人工指定的。训练开始时，卷积核参数通常是随机的，神经元并不知道狗脸、猫脸、猪肉纹理是什么。

它后来变得敏感，是因为训练目标不断奖励那些能帮助分类正确的视觉特征。

例如很多狗样本里反复出现：

- 眼睛
- 鼻子
- 嘴巴
- 耳朵
- 毛发纹理
- 脸部轮廓

如果某些卷积核开始对这些结构有响应，并且这些响应能帮助最终输出 `dog`，那么反向传播就会强化相关参数。经过大量样本之后，这些卷积核就可能从“随机响应”逐渐变成“对狗脸局部模式稳定响应”。

## 9. 训练更新到底发生在哪里

一次训练时，图像或 proposal 先做前向传播：

```text
输入 proposal
-> conv1 / conv2 / conv3 / conv4 / conv5
-> pool5
-> fc6 / fc7
-> 分类层
-> 输出各类别分数
```

假设真实类别是 `dog`，但模型预测成了 `cat`：

```text
dog_score 低
cat_score 高
```

这时损失函数会变大。损失函数的作用是告诉模型：当前输出和正确答案差了多少。

然后反向传播从输出层往前传：

```text
loss
-> 分类层
-> fc 层
-> pool5 / conv5
-> conv4
-> conv3
-> conv2
-> conv1
```

它不是凭语义理解“这是狗脸”，而是通过梯度计算：哪些激活和权重让 `cat` 分数太高、`dog` 分数太低。

可以粗略理解为：

```text
真实是 dog：
强激活且支持 dog 的路径 -> 加强
强激活但错误支持 cat 的路径 -> 削弱
没有激活、没参与这次判断的路径 -> 更新很小或不更新
```

更技术一点，某个权重的更新量和两个东西有关：

```text
更新量 ≈ 误差信号 x 该神经元的激活值
```

如果一个神经元这次激活值很高，说明它参与了这次判断，那么它相关的权重更可能被明显更新。如果一个神经元几乎没激活，比如车轮神经元看到狗脸时输出接近 0，它对这次错误贡献很小，梯度也通常很小。

卷积层也是类似的。一个卷积核的响应可以粗略理解为：

```text
激活值 = 卷积核权重 · 输入局部特征 + bias
```

如果某个卷积核对狗脸局部结构有响应，而且这种响应能帮助降低 loss，反向传播会调整这个卷积核的权重，让它以后更容易匹配这种有用模式。

所以最终形成的是一种分工：

- 有些神经元对狗脸敏感。
- 有些神经元对猫耳朵敏感。
- 有些神经元对毛发纹理敏感。
- 有些神经元对车轮敏感。
- 有些神经元对肉类纹理敏感。
- 有些神经元对文字、窗格、颜色斑块敏感。

但单个神经元通常不决定最终类别。最终判断来自许多特征的组合：

```text
中间层：提取局部、部件、纹理模式
高层和全连接层：组合这些模式
分类器：输出 dog、cat、car 等类别分数
```

一句话总结：

**神经元的敏感性来自训练中反复的前向计算和反向更新。凡是能帮助正确分类的模式会被强化，导致错误分类的连接会被削弱，没参与当前样本判断的路径更新很小。经过大量样本后，某些神经元自然变成狗脸、猫脸、肉类纹理等视觉模式的检测器。**

## 10. 附录 B 精读：为什么 fine-tuning 和 SVM 的样本定义不同

附录 B 主要解释两个问题：

1. 为什么 CNN fine-tuning 和 SVM 训练的正负样本定义不一样。
2. 为什么已经有 fine-tuned CNN 的 softmax 输出了，还要再训练 SVM。

核心思想是：

**R-CNN 里 CNN 和 SVM 承担的任务不完全一样，所以训练样本的定义也不一样。**

### 6.1 CNN fine-tuning 的样本定义

CNN fine-tuning 时，作者把每个 region proposal 和真实框做 IoU 匹配：

```text
如果 proposal 和某个 ground-truth box 的 IoU >= 0.5
-> 当作这个类别的正样本

否则
-> 当作背景
```

例如真实框是一只狗，某个 proposal 框住了狗的大部分身体，IoU 是 `0.62`，那它就被当作 `dog` 正样本。

这时 CNN 学的是：

```text
这个 proposal 大体上是不是某类物体？
```

它不要求 proposal 必须非常精确地贴住物体边界。只要重叠够高，就可以作为正样本。

为什么这么宽松？因为 CNN fine-tuning 要更新整个大网络，参数很多。如果只用 ground-truth boxes 当正样本，样本数量太少，容易过拟合。

所以作者用 `IoU >= 0.5` 的 proposals 扩大正样本数量。这相当于引入了很多 jittered examples，也就是轻微偏移、偏大、偏小、形变但仍然大体包含目标的正样本。这能让 CNN 学到更鲁棒的检测领域特征。

### 6.2 SVM 的样本定义

SVM 训练时规则更严格。

对于某一个类别，比如 `dog` SVM：

```text
正样本：dog 的 ground-truth boxes
负样本：和所有 dog 实例 IoU < 0.3 的 proposals
忽略：IoU >= 0.3 但又不是 ground-truth box 的 proposals
```

这和 fine-tuning 不一样。

fine-tuning 时：

```text
IoU >= 0.5 的 proposal 可以是正样本
```

SVM 时：

```text
正样本更严格，主要使用 ground-truth boxes
```

为什么 SVM 更严格？因为 SVM 是最终检测器，它直接决定一个 proposal 的检测分数。它要学的是：

```text
这个框是不是一个好的、准确的该类检测框？
```

如果把那些框得很松、框得偏的 proposal 都当作正样本，SVM 可能会觉得“只要大概包住狗就算好检测”，这样定位质量会变差。

所以 SVM 训练更像是在教模型：

```text
贴得准的框 -> 高分
明显不是该类的框 -> 低分
中间模糊的框 -> 不参与训练
```

`0.3` 到 `1.0` 之间但不是 ground truth 的区域，很多属于灰区。比如一个框只框住狗头，或者框住狗和背景的一大块。它既不完全是背景，也不是标准正例。作者干脆忽略它，避免给 SVM 传递混乱信号。

### 6.3 为什么不用 softmax 直接做检测

R-CNN 在 fine-tuning CNN 时，最后其实有一个 softmax 分类层。VOC 有 20 个类别，再加一个背景类，所以是：

```text
21-way softmax
```

一个自然问题是：既然 CNN 已经能输出 `dog/cat/car/background` 的概率，为什么不直接用这个输出做检测？

作者试了。结果是：

```text
直接用 softmax：50.9% mAP
使用 SVM：54.2% mAP
```

也就是说，SVM 更好，大约高 `3.3` 个 mAP 点。

原因主要有两个。

第一，softmax 的训练样本定义更宽松。fine-tuning 时 `IoU >= 0.5` 的 proposal 都可能被当作正样本。这有利于训练 CNN，但不一定有利于最终精确检测。

第二，SVM 使用 hard negative mining，也就是专门找那些模型容易误判的负样本来训练。

比如对 `dog` SVM 来说，普通背景很容易判断：

```text
天空、草地、墙面 -> 不像狗
```

但困难负例可能是：

- 猫脸
- 毛绒玩具
- 狐狸
- 狗旁边的背景框
- 只框住狗尾巴的 proposal

这些更容易被误判成狗。SVM 通过 hard negative mining 重点学习这些难例，所以最终检测更稳。

附录 B 的一句话总结：

**fine-tuning 是为了让 CNN 学到更适合检测的特征，所以用更多、更宽松的正样本；SVM 是为了最终判断检测框是否可靠，所以用更严格的正样本和困难负例。softmax 虽然能直接做检测，但在 R-CNN 这套训练方式下，SVM 的标签规则和 hard negative mining 带来了更好的 mAP。**

## 11. 为什么更深的 O-Net/VGG 提高精度，也让速度问题更突出

R-CNN 后续版本中加入了更深的 O-Net/VGG 结果。更深网络带来了更强的表示能力，因此检测精度提高。

但这也让 R-CNN 的速度问题更加突出。

原因在于 R-CNN 的计算方式是：

```text
每张图约 2000 个 proposals
每个 proposal 单独 warp 到固定大小
每个 proposal 单独跑一次 CNN
```

如果 backbone 是 AlexNet，单次前向已经不便宜；如果换成更深的 O-Net/VGG，单次前向更贵。因为 R-CNN 不能在这些 proposals 之间共享大部分卷积计算，所以每张图的总成本近似变成：

```text
总计算量 ≈ proposals 数量 x 单个 CNN 前向成本
```

更深网络让“单个 CNN 前向成本”变大，于是整体速度问题被放大。

这也解释了论文中 R-CNN 与 OverFeat 的差异。OverFeat 大致采用滑动窗口/整图卷积式计算，重叠窗口之间可以共享卷积特征；R-CNN 则对每个 proposal 在图像层面单独 warp，再单独跑 CNN，因此共享计算很少。

所以：

```text
更深 CNN -> 特征更强 -> 精度更高
更深 CNN -> 每个 proposal 前向更慢 -> R-CNN 速度瓶颈更严重
```

这正是后来 Fast R-CNN / Faster R-CNN 要改掉逐 proposal 前向传播的重要原因。

## 12. ILSVRC2013 为什么不能从 train split 挖 hard negatives

在 ILSVRC2013 detection 中，作者不能简单地从 `train` split 挖 hard negatives，关键原因是：

**train split 的标注不完整。**

hard negative mining 的前提是：如果一个 region 没有和某类标注目标重叠，我们可以相对放心地把它当作该类负样本。

但如果数据集标注不完整，会出现危险情况：

```text
图中其实有一只未标注的 dog
某个 proposal 框住了这只 dog
因为标注里没有 dog box
系统误以为它是 dog 的负样本
```

这样会污染 SVM 训练。模型会被迫学习：

```text
真正的 dog 外观 -> 负样本
```

这对 hard negative mining 尤其危险，因为 hard negatives 本来就是模型容易误判为目标的区域。如果其中混入了未标注目标，就会把真正的正例当成困难负例反复强化，损害分类边界。

因此论文中 ILSVRC2013 的数据使用更谨慎：

- `train` 可用于增加正样本，因为有些类别实例被标出来。
- 但不适合用来挖 hard negatives，因为没标出来的目标可能被错当负例。
- hard negative mining 主要在标注更可靠的 `val1` 上进行。

一句话总结：

**hard negative mining 要求“没标注为目标”大致等于“不是目标”；ILSVRC2013 train split 标注不完整，这个前提不成立。**

## 13. 附录 A 为什么最终选择 `warp + p=16`

R-CNN 中每个 region proposal 的形状和大小都不同，但 CNN 要求固定输入：

```text
227 x 227
```

所以必须把任意矩形 proposal 转换成固定大小输入。附录 A 比较了几种方式：

1. 用最紧正方形包住 proposal，再等比例缩放。
2. 不保留上下文，只保留目标区域。
3. 直接把 proposal 的外接矩形 warp 到 `227 x 227`。
4. 给 proposal 周围加入 context padding。

最终选择：

```text
warp + p=16
```

其中 `warp` 是各向异性缩放，会改变宽高比；`p=16` 表示在变换后的输入坐标系中，原 proposal 四周保留 16 像素上下文。

为什么选它？

第一，CNN 必须吃固定尺寸输入，warp 最简单直接。它让 proposal 内容充分占据网络输入，不需要复杂的 mask 或形状处理。

第二，上下文对检测有帮助。很多目标不能只看目标内部，还要看周围环境。例如：

- 车和道路背景有关。
- 船和水面背景有关。
- 人和身体周围轮廓有关。
- 只看局部纹理可能难以区分目标类别。

`p=16` 给网络保留了一圈上下文，有助于分类。

第三，这是实验选择。附录 A 中预实验发现：

```text
warp + p=16 比其他替代方案高约 3-5 mAP 点
```

所以它不是因为理论上最优雅，而是因为：

```text
简单、统一、效果最好
```

一句话总结：

**warp 解决固定输入尺寸问题，`p=16` 提供有用上下文；两者组合在预实验中显著优于替代方案。**

## 14. bbox regression 的详细解释

bbox regression 是 R-CNN 中专门修正定位误差的模块。它不负责判断类别，而是负责把一个已经被 SVM 认为“像某类目标”的 proposal 调整得更贴近真实目标框。

### 14.1 它解决的具体问题

R-CNN 的主体检测流程是：

```text
Selective Search proposals
-> CNN features
-> class-specific SVM scores
-> NMS
```

如果没有 bbox regression，最终检测框就是 Selective Search 原始 proposal。

问题是：Selective Search 负责“提出可能有物体的区域”，但它不是为精确贴合物体边界而优化的。一个 proposal 可能包含目标，但位置偏移、宽高不准、框得太松或太紧。

例如真实目标框是：

```text
GT = (50, 40, 180, 200)
```

Selective Search 给出的高分 proposal 可能是：

```text
P = (45, 35, 170, 190)
```

SVM 可能正确判断它是 `cat`，但框和真实框仍有偏差。VOC 评估依赖 IoU，如果定位不准，即使类别判断正确，也可能因为 IoU 不够而被算作错误检测。

论文的错误分析发现，R-CNN 的一个主要错误类型就是 localization error：

```text
类别对了，但框不够准。
```

bbox regression 正是为这个问题服务的。

### 14.2 它学的不是类别，而是几何修正

bbox regression 的输入和输出可以理解为：

```text
输入：proposal 的 CNN 特征
输出：这个 proposal 应该如何平移、缩放，才能更接近 ground-truth box
```

对每个类别，R-CNN 学一个类别专属回归器：

```text
aeroplane bbox regressor
bicycle bbox regressor
...
person bbox regressor
...
tvmonitor bbox regressor
```

为什么是类别专属？因为不同类别的形状和 proposal 偏差模式不同。比如人、车、瓶子、狗的框形状差异很大，用同一个修正函数可能不够精细。

### 14.3 框的参数化方式

回归器不是直接预测新的 `(xmin, ymin, xmax, ymax)`，而是预测 proposal 到 ground truth 的相对变换。

先把 proposal `P` 和真实框 `G` 都写成中心点和宽高：

```text
P = (Px, Py, Pw, Ph)
G = (Gx, Gy, Gw, Gh)
```

其中：

```text
Px, Py = proposal 中心点
Pw, Ph = proposal 宽高
Gx, Gy = ground-truth 中心点
Gw, Gh = ground-truth 宽高
```

训练目标是四个量：

```text
tx = (Gx - Px) / Pw
ty = (Gy - Py) / Ph
tw = log(Gw / Pw)
th = log(Gh / Ph)
```

这四个量分别表示：

```text
tx：中心点 x 方向应该移动多少，按 proposal 宽度归一化
ty：中心点 y 方向应该移动多少，按 proposal 高度归一化
tw：宽度应该放大或缩小多少，用 log 比例表示
th：高度应该放大或缩小多少，用 log 比例表示
```

这样做比直接预测绝对坐标更稳定。因为不同图片和目标大小不同，直接学像素坐标会让数值尺度变化很大；用相对平移和对数尺度变化，学习问题更统一。

### 14.4 一个具体计算例子

假设 proposal 是：

```text
P = (45, 35, 170, 190)
```

这里用 `(xmin, ymin, xmax, ymax)` 表示。

它的中心点和宽高大约是：

```text
Pw = 170 - 45 = 125
Ph = 190 - 35 = 155
Px = 45 + 125 / 2 = 107.5
Py = 35 + 155 / 2 = 112.5
```

真实框是：

```text
G = (50, 40, 180, 200)
```

真实框的中心点和宽高是：

```text
Gw = 180 - 50 = 130
Gh = 200 - 40 = 160
Gx = 50 + 130 / 2 = 115
Gy = 40 + 160 / 2 = 120
```

那么训练目标是：

```text
tx = (115 - 107.5) / 125 = 0.06
ty = (120 - 112.5) / 155 ≈ 0.048
tw = log(130 / 125) ≈ 0.039
th = log(160 / 155) ≈ 0.032
```

这说明回归器应该学习：

```text
中心点略向右移动
中心点略向下移动
宽度略微放大
高度略微放大
```

测试时，如果回归器对某个 proposal 预测出类似的 `(tx, ty, tw, th)`，就可以把原始 proposal 修正到更接近真实框的位置。

### 14.5 回归器的形式

论文中每个回归函数都是基于 CNN 特征的线性函数。可以把它理解成：

```text
tx = wx · feature + bx
ty = wy · feature + by
tw = ww · feature + bw
th = wh · feature + bh
```

输入特征在论文附录 C 中使用的是 `pool5` 特征 `φ5(P)`。也就是说，回归器不是只看 proposal 的几何坐标，而是看 CNN 从 proposal 中提取出的视觉特征。

训练目标是让预测的四个变换量接近真实变换量。论文使用带 L2 正则的最小二乘，也就是 ridge regression。正则化很重要，论文在验证集上选择了较强的正则参数。

### 14.6 训练样本怎么选

不是所有 proposals 都适合训练 bbox regressor。

如果一个 proposal 离所有真实框都很远，例如只框住背景，那么让它学习如何变成某个真实框是不合理的。这样的训练 pair 会非常混乱。

所以论文只选择与某个 ground-truth box 足够接近的 proposals，例如：

```text
IoU(P, G) >= 0.6
```

这些 proposal 已经大致覆盖目标，回归器只需要学习“微调”：

```text
稍微平移
稍微放大或缩小
让框更贴合目标
```

这也是 bbox regression 的本质：它不是从任意背景框中“创造”目标框，而是在已有较好 proposal 的基础上做定位修正。

### 14.7 测试时怎么用

测试时流程是：

```text
1. Selective Search 生成 proposal P
2. CNN 提取 P 的特征
3. 每个类别的 SVM 给 P 打分
4. 如果某个类别分数高，就使用该类别的 bbox regressor
5. regressor 预测 tx, ty, tw, th
6. 用这四个量把 P 修正成新框
7. 对修正后的框做 NMS 和最终输出
```

所以不带 bbox regression：

```text
输出框 = Selective Search 原始 proposal
```

带 bbox regression：

```text
输出框 = 原始 proposal 经过类别专属回归器修正后的框
```

两者分类分数可能相同，但框坐标不同。

### 14.8 为什么它能提高 mAP

VOC 的 AP/mAP 不只看分类，还看定位。一个检测框必须同时满足：

```text
类别正确
IoU 足够高
```

R-CNN 的 CNN 特征和 SVM 已经让类别判别很强，但 Selective Search 框不一定足够准。bbox regression 修正框后，很多原本因为 IoU 不够而失败的检测，会变成正确检测。

因此 bbox regression 主要提升的是定位质量，尤其减少：

```text
classification correct, localization wrong
```

这种错误。

一句话总结：

**bbox regression 不解决“这是什么类别”，它解决“这个框怎样移动和缩放才更贴住目标”。它把 R-CNN 从“选中一个大概对的 proposal”推进到“输出一个更精确的检测框”。**

## 15. 整篇 R-CNN 的主线

R-CNN 的系统可以压缩成一条主线：

```text
region proposals 负责给出可能位置
CNN 负责提取强视觉特征
SVM 负责最终类别判别边界
hard negative mining 让 SVM 学会拒绝困难背景
bbox regression 修正定位误差
```

它的关键创新不是某一个单点，而是把几个模块组合得很有效：

1. 用 Selective Search 避免穷举滑窗。
2. 用 ImageNet 预训练 CNN 解决检测数据不足。
3. 用 VOC proposals fine-tuning 让 CNN 适应检测任务。
4. 用 `fc7` 深度特征替代 HOG/SIFT 这类手工特征。
5. 用 class-specific SVM 和 hard negative mining 学最终检测边界。
6. 用 bbox regression 补救定位误差。

更深的 O-Net/VGG 进一步证明：CNN 表示越强，检测精度可以越高。但它也暴露了 R-CNN 的根本速度问题：每个 proposal 单独跑 CNN，导致计算量随候选框数量线性放大。

这条瓶颈后来推动了 Fast R-CNN、Faster R-CNN 的出现：

```text
R-CNN：每个 proposal 单独跑 CNN
Fast R-CNN：整图跑 CNN，proposal 在特征图上共享计算
Faster R-CNN：进一步用 RPN 学 proposals
```

## 16. 常见误区

**误区 1：`pool5` 的每个格子都学会一种不同特征。**

更准确地说，通道更接近“看什么”，`6 x 6` 位置表示“在哪里”。同一通道在不同空间位置上共享相同的检测函数。

**误区 2：`pool5` 本身学习了狗脸。**

`pool5` 通常没有可学习参数，它只是对 `conv5` 响应做 max pooling。真正学习的是前面卷积层的卷积核以及后面的全连接权重。

**误区 3：bbox regression 是重新分类。**

bbox regression 不判断类别，它只修正框位置。

**误区 4：ILSVRC train 没有标注的区域都能当负样本。**

不行。因为标注不完整，未标注区域可能是真实目标。直接挖 hard negatives 会污染 SVM。

**误区 5：`warp + p=16` 是理论上必然最优。**

不是。它是简单工程方案，并且在预实验中比替代方案高 `3-5` mAP，因此被采用。

## 17. 可背诵总结

R-CNN 的核心是把“区域候选”和“CNN 特征”结合起来。Selective Search 给出约 2000 个可能位置，CNN 把每个 warped proposal 转成高维语义特征，SVM 用这些特征做类别判别，bbox regression 修正定位误差。

神经元可视化说明 CNN 学到的不是单纯类别标签，而是一组分布式视觉模式，包括形状、纹理、颜色、材质和局部部件。`pool5 feature (3,3,13)` 的 top activations 是固定一个神经元后，在大量 proposals 中找最能激活它的区域。

附录 B 解释了为什么 CNN fine-tuning 和 SVM 的样本定义不同：前者为了学习鲁棒特征，用更宽松的正样本；后者作为最终检测器，用更严格的正样本和 hard negative mining。

Appendix A 说明 `warp + p=16` 是固定输入尺寸和上下文信息之间的有效折中。bbox regression 则针对 R-CNN 的主要错误之一：分类正确但定位不准。

更深的 O-Net/VGG 提高了特征表达能力和检测精度，但也让 R-CNN 每个 proposal 单独跑 CNN 的速度瓶颈更明显。这正是后续 Fast R-CNN 系列要解决的问题。
