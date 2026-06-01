# R-CNN 学习笔记与 VOC 复现项目

这个仓库用于整理 R-CNN 论文阅读材料，以及一个基于 Pascal VOC 的 R-CNN 复现系统。

## 内容

- `R-CNN精读.md`：论文逐节精读、翻译与讲解，包含论文原图对照。
- `R-CNN问题精读笔记.md`：围绕问题整理的阅读笔记，重点解释 R-CNN 主线、数据集处理、训练流程、速度瓶颈、边界框回归和附录选择。
- `assets/rcnn-paper/`：从论文中整理出的关键图表。
- `assets/rcnn-notes/`：阅读笔记配套示意图。
- `rcnn_voc_system/`：R-CNN VOC 复现系统，包含训练、检测、评估、Streamlit 网页浏览和神经元可视化。
- `rcnn_steel_detection/`：早期钢材缺陷检测相关代码，作为对比和历史实验材料保留。
- `tools/`：论文图表渲染与整理工具。

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

仓库不会提交本地 VOC 数据集、训练输出、模型权重、proposal 缓存、日志文件、Python 缓存和 IDE 配置。这些内容体积大或与本机环境相关，已经通过 `.gitignore` 排除。

如需重新生成训练产物，请参考 `rcnn_voc_system/README.md`。
