# R-CNN 学习笔记与 VOC 复现项目

这个仓库用于整理 R-CNN 问题驱动阅读笔记，以及一个基于 Pascal VOC 的 R-CNN 复现系统。

## 内容

- `R-CNN问题精读笔记.md`：围绕问题整理的阅读笔记，重点解释 R-CNN 主线、数据集处理、训练流程、速度瓶颈、边界框回归和附录选择。
- `rcnn_voc_system/`：R-CNN VOC 复现系统，包含训练、检测、评估、Streamlit 网页浏览和神经元可视化。

## 运行 VOC 复现系统

```powershell
cd rcnn_voc_system
python -m pip install -e .
streamlit run app/streamlit_app.py
```

网页默认访问：

```text
http://localhost:8501/
```

## 没有上传的内容

仓库不会提交原论文 PDF、详细逐段精读笔记、旧的钢材检测项目、本地 VOC 数据集、训练输出、模型权重、proposal 缓存、日志文件、Python 缓存和 IDE 配置。这些内容体积大、与本机环境相关，或不属于这个 GitHub 仓库的公开范围，已经通过 `.gitignore` 排除。

如需重新生成训练产物，请参考 `rcnn_voc_system/README.md`。
