# R-CNN VOC 论文复现与神经元可视化系统

这是一个从零实现的 R-CNN 复现项目，目标是尽量按照 Girshick et al. 原始 R-CNN 论文的流程，在 Pascal VOC 2010/2012 数据集上完成目标检测训练、检测、评估，并提供一个本地 Streamlit Web 系统用于演示和多层神经元可视化。

本项目不依赖旧的钢材缺陷检测代码，是一个独立的新工程。

## 项目目标

系统复现的是经典 R-CNN 的三阶段检测范式：

1. 使用 Selective Search 生成类别无关的候选区域。
2. 将每个候选区域直接 warp 到 `227 x 227`。
3. 使用 ImageNet 预训练 AlexNet，并在 VOC region classification 任务上微调。
4. 从 AlexNet `fc7` 层提取 `4096` 维特征。
5. 对 VOC 20 个类别分别训练一个线性 SVM。
6. 对每个类别训练边界框回归器。
7. 测试时执行 `proposals -> CNN features -> SVM scores -> bbox regression -> per-class NMS`。
8. 输出 VOC 格式检测结果，并计算 AP/mAP。
9. 支持类似论文 Figure 4 的神经元 top activating regions 可视化。

## 项目结构

```text
rcnn_voc_system/
  app/
    streamlit_app.py          # 本地 Web 可视化系统
  configs/
    rcnn_voc.yaml             # 完整 VOC 复现配置
    rcnn_voc_tiny.yaml        # 小规模 smoke test 配置
  scripts/
    check_dataset.py          # 检查 VOC 数据集
    generate_proposals.py     # 生成 Selective Search proposal 缓存
    train_full.py             # 完整训练流程：CNN + SVM + bbox regression
    evaluate.py               # 检测并计算 AP/mAP
    detect_image.py           # 单张图片检测
    export_neuron_cache.py    # 导出神经元激活缓存与网格图
  src/rcnn/
    voc.py                    # VOCdevkit 数据解析
    proposals.py              # Selective Search
    preprocess.py             # region crop、warp、ImageNet 均值预处理
    model.py                  # AlexNet R-CNN 模型与多层特征提取
    train.py                  # CNN 微调、SVM、bbox regression
    detect.py                 # 单图/数据集检测与 VOC 结果导出
    evaluate.py               # AP/mAP 评估
    neuron.py                 # 多层神经元 top activation 可视化
    geometry.py               # IoU、NMS、bbox transform
    metrics.py                # VOC AP 计算
    linear.py                 # NumPy 版 Linear SVM 和 Ridge Regressor
  tests/
    test_*.py                 # 单元测试
  outputs/
    checkpoints/              # 训练后的模型、SVM、回归器
    proposals/                # proposal 缓存
    detections/               # 检测结果与 VOC txt 文件
    reports/                  # 评估报告 JSON
    neurons/                  # 神经元 top region 缓存和图片
```

## 环境安装

建议在项目目录中安装为可编辑包：

```powershell
cd C:\Users\wang'q'k\Documents\r-cnn\rcnn_voc_system
python -m pip install -e .
```

主要依赖包括：

- `torch` / `torchvision`
- `opencv-contrib-python`
- `numpy`
- `Pillow`
- `PyYAML`
- `streamlit`
- `matplotlib`

项目内置了 NumPy 版线性 SVM 和 Ridge 回归器，因此不依赖 `scikit-learn`。这是为了避免某些环境中 `scikit-learn/scipy` 与 NumPy 2 的二进制兼容问题。

## Pascal VOC 数据集要求

项目读取标准 VOCdevkit 结构，例如：

```text
VOCdevkit/
  VOC2012/
    Annotations/
    ImageSets/
      Main/
        train.txt
        val.txt
        trainval.txt
    JPEGImages/
```

支持：

- `VOC2010`
- `VOC2012`

如果本机还没有 VOCdevkit，可以先尝试下载：

```powershell
python scripts/download_voc.py --root data --year 2012 --image-set trainval
```

下载完成后，VOC 根目录通常是：

```text
data/VOCdevkit
```

配置文件中默认使用：

```yaml
dataset:
  year: "2012"
  image_set_train: "trainval"
  image_set_eval: "val"
```

如果使用 VOC2012，请注意官方 test 标注通常不可用。本项目不会伪造 test mAP；当 test 标注不存在时，应导出 VOC 官方评测格式的检测结果文件。

## 快速检查数据集

```powershell
python scripts/check_dataset.py --config configs/rcnn_voc.yaml --voc-root C:\path\to\VOCdevkit
```

该命令会输出：

- VOC 年份和 split
- 图片数量
- 有标注图片数量
- 总目标数
- difficult 目标数量
- 每个类别的目标数量
- 实际解析到的 VOC 路径

## 完整训练流程

### 1. 生成候选区域缓存

```powershell
python scripts/generate_proposals.py --config configs/rcnn_voc.yaml --voc-root C:\path\to\VOCdevkit
```

默认使用：

- Selective Search fast mode
- 每张图最多 `2000` 个候选区域
- 最小候选框边长 `16`

缓存会写入：

```text
outputs/proposals/
```

### 2. 运行完整训练

```powershell
python scripts/train_full.py --config configs/rcnn_voc.yaml --voc-root C:\path\to\VOCdevkit
```

该命令依次执行：

1. 如果 proposal 缓存不存在，先生成缓存。
2. 微调 AlexNet。
3. 提取 region 的 `fc7` 特征。
4. 训练每个类别的线性 SVM。
5. 训练每个类别的 bbox regression。

输出文件位于：

```text
outputs/checkpoints/
  alexnet_rcnn_finetuned.pt
  linear_svms.pkl
  bbox_regressors.pkl
```

### 3. 跳过部分训练阶段

如果已经有 CNN 权重，只想重新训练 SVM：

```powershell
python scripts/train_full.py --config configs/rcnn_voc.yaml --voc-root C:\path\to\VOCdevkit --skip-cnn --skip-bbox
```

如果只想训练 bbox regression：

```powershell
python scripts/train_full.py --config configs/rcnn_voc.yaml --voc-root C:\path\to\VOCdevkit --skip-cnn --skip-svm
```

## 检测与评估

### 单张图片检测

```powershell
python scripts/detect_image.py --config configs/rcnn_voc.yaml --voc-root C:\path\to\VOCdevkit C:\path\to\image.jpg --output outputs\detections\detected.jpg
```

该命令会输出检测框，并保存可视化图片。

### 数据集评估

```powershell
python scripts/evaluate.py --config configs/rcnn_voc.yaml --voc-root C:\path\to\VOCdevkit --split val
```

评估会执行：

- 对验证集运行 R-CNN 检测。
- 导出 VOC detection result txt。
- 根据本地标注计算每类 AP。
- 计算整体 mAP。
- 将报告保存到 `outputs/reports/`。

VOC detection result 文件格式类似：

```text
comp3_det_test_cat.txt
comp3_det_test_person.txt
...
```

每行格式为：

```text
image_id score xmin ymin xmax ymax
```

坐标会按 VOC 官方格式导出为 1-based。

## Streamlit Web 系统

启动 Web 系统：

```powershell
streamlit run app/streamlit_app.py
```

默认访问：

[http://localhost:8501](http://localhost:8501)

Web 系统包含四个页面。

### Dataset

用于检查 VOC 数据集是否能正确读取。

显示内容包括：

- 数据集年份
- split 名称
- 图片数量
- 标注数量
- 每类目标统计
- proposal 缓存路径和是否存在

### Train

用于查看完整训练命令。

由于完整 R-CNN 训练耗时较长，Web 页面不会直接在浏览器中启动长训练任务，而是给出可复制的终端命令。这样训练日志更稳定，也方便中断和恢复。

### Detect & Eval

支持两种检测方式：

- 上传任意图片。
- 从 VOC 验证集中选择图片。

点击检测后会展示：

- 检测框可视化图。
- 类别名。
- SVM 分数。
- bbox 坐标。

如果已经运行过 `scripts/evaluate.py`，该页面还会读取 `outputs/reports/*.json` 并展示 mAP 报告。

### Neuron Viewer

用于查看神经元激活最高的图像区域。

Web 页面读取 `outputs/neurons/*.json` 缓存，并展示对应的 top region 网格图。

如果缓存不存在，页面会提示运行类似命令：

```powershell
python scripts/export_neuron_cache.py --config configs/rcnn_voc.yaml --voc-root C:\path\to\VOCdevkit --layer pool5 --channel 0 --top-k 16
```

## 神经元可视化说明

默认神经元可视化遵循 R-CNN 原文的非参数方法：

1. 选择一个网络层。
2. 选择一个通道、空间位置或全连接 unit。
3. 在 held-out proposals 上前向传播。
4. 记录该神经元在每个 proposal 上的激活值。
5. 按激活值从高到低排序。
6. 展示 top-k 个最能激活该神经元的真实图像区域。

支持层：

- `conv1`
- `conv2`
- `conv3`
- `conv4`
- `conv5`
- `pool5`
- `fc6`
- `fc7`

卷积层和 `pool5`：

- 使用 `--channel` 选择通道。
- 可用 `--row`、`--col` 选择空间位置。
- 如果不指定空间位置，默认取特征图中心位置。

全连接层：

- 使用 `--unit` 选择神经元编号。

示例：

```powershell
python scripts/export_neuron_cache.py --config configs/rcnn_voc.yaml --voc-root C:\path\to\VOCdevkit --layer pool5 --channel 42 --row 3 --col 3 --top-k 16
```

```powershell
python scripts/export_neuron_cache.py --config configs/rcnn_voc.yaml --voc-root C:\path\to\VOCdevkit --layer fc7 --unit 100 --top-k 16
```

输出包括：

```text
outputs/neurons/
  pool5_c42_r3_c3_uNone.json
  pool5_c42_r3_c3_uNone.png
```

`.json` 保存 top activation 元数据，`.png` 是类似论文 Figure 4 的 top region 网格图。

## 配置文件说明

完整复现配置位于：

```text
configs/rcnn_voc.yaml
```

关键配置：

```yaml
proposals:
  mode: "fast"
  top_k: 2000

model:
  image_size: 227
  feature_layer: "fc7"

training:
  epochs: 20
  batch_size: 128
  learning_rate: 0.0001
  log_interval: 100
  grad_clip_norm: 10.0
  positive_iou: 0.5
  svm_negative_iou: 0.3
  bbox_iou: 0.6
  hard_negative_iterations: 3

detection:
  score_threshold: 0.0
  nms_iou: 0.3
```

小规模测试配置位于：

```text
configs/rcnn_voc_tiny.yaml
```

它会减少图片数、proposal 数和 epoch 数，只用于验证流程是否能跑通，不用于报告论文复现实验结果。

## 测试

运行单元测试：

```powershell
python -m pytest -q
```

当前测试覆盖：

- VOC XML 解析。
- IoU。
- NMS。
- bbox transform / inverse transform。
- AP 计算。
- AlexNet `fc7` 输出 shape。
- VOC detection result txt 格式。

## 当前实现边界

- 这是完整工程骨架和可运行实现，不包含 VOC 数据集本身。
- 完整 VOC 训练非常耗时，尤其是 Selective Search 和逐 region AlexNet 前向传播。
- 默认使用 AlexNet 和 R-CNN 原始范式，不使用 Fast R-CNN / Faster R-CNN 的 RoI Pooling 或端到端训练。
- VOC2012 test 标注不可本地获得时，只导出官方评测格式结果，不报告本地 test mAP。
- 神经元可视化的主方法是论文中的 top activating regions；热力图只作为辅助理解，不作为原文 Figure 4 的替代。
- 本项目使用 torchvision 的 ImageNet AlexNet 权重，因此输入预处理采用 torchvision 的 RGB、`0-1` 缩放、ImageNet mean/std。原论文 Caffe AlexNet 使用 BGR 减均值，两者权重来源不同，预处理不能混用。

## 推荐运行顺序

```powershell
cd C:\Users\wang'q'k\Documents\r-cnn\rcnn_voc_system

python scripts/check_dataset.py --config configs/rcnn_voc.yaml --voc-root C:\path\to\VOCdevkit

python scripts/generate_proposals.py --config configs/rcnn_voc.yaml --voc-root C:\path\to\VOCdevkit

python scripts/train_full.py --config configs/rcnn_voc.yaml --voc-root C:\path\to\VOCdevkit

python scripts/evaluate.py --config configs/rcnn_voc.yaml --voc-root C:\path\to\VOCdevkit --split val

python scripts/export_neuron_cache.py --config configs/rcnn_voc.yaml --voc-root C:\path\to\VOCdevkit --layer pool5 --channel 0 --top-k 16

streamlit run app/streamlit_app.py
```
