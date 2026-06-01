from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

platform.machine = lambda: "AMD64"
platform.system = lambda: "Windows"

import cv2
import numpy as np
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rcnn.config import load_config, resolve_paths, seed_everything
from rcnn.detect import detect_image, draw_detections, load_pickle
from rcnn.evaluate import evaluate_detections
from rcnn.model import load_model
from rcnn.neuron import load_hits, render_top_regions
from rcnn.proposals import load_proposals, proposal_cache_path
from rcnn.voc import VOCDataset, dataset_summary


st.set_page_config(page_title="R-CNN VOC 系统", layout="wide")
st.title("R-CNN VOC 论文复现系统")

config_path = st.sidebar.text_input("配置文件", str(PROJECT_ROOT / "configs" / "rcnn_voc_top300.yaml"))
voc_root = st.sidebar.text_input("VOCdevkit 根目录", str(PROJECT_ROOT / "data" / "VOCdevkit"))
device = st.sidebar.selectbox("运行设备", ["cpu", "cuda"], index=0)
page = st.sidebar.radio("页面", ["数据集", "训练", "检测与评估", "神经元可视化"])


@st.cache_data(show_spinner=False)
def read_cfg(path: str):
    return load_config(path)


cfg = read_cfg(config_path)
seed_everything(int(cfg.get("training", {}).get("seed", 42)))
paths = resolve_paths(cfg, PROJECT_ROOT)


def dataset(split_key: str):
    root = voc_root or cfg.get("dataset", {}).get("voc_root")
    if not root:
        st.warning("请先在左侧设置 VOCdevkit 根目录。")
        return None
    try:
        return VOCDataset(root, cfg["dataset"].get("year", "2012"), cfg["dataset"].get(split_key, "val"))
    except Exception as exc:
        st.error(str(exc))
        return None


def translate_summary(summary: dict) -> dict:
    labels = {
        "year": "年份",
        "image_set": "数据划分",
        "num_images": "图像数量",
        "num_objects": "目标数量",
        "classes": "类别",
        "class_counts": "各类别目标数量",
    }
    return {labels.get(key, key): value for key, value in summary.items()}


def translate_detections(dets) -> list[dict]:
    rows = []
    for det in dets:
        row = det.__dict__.copy()
        rows.append(
            {
                "图像 ID": row.get("image_id"),
                "类别": row.get("class_name"),
                "分数": row.get("score"),
                "边界框 (x1, y1, x2, y2)": row.get("box"),
            }
        )
    return rows


if page == "数据集":
    ds = dataset("image_set_train")
    if ds:
        st.subheader("数据集概览")
        summary = dataset_summary(ds)
        st.json(translate_summary(summary))
        cache = proposal_cache_path(paths.proposals, ds.year, ds.image_set)
        st.write("候选区域缓存：", str(cache))
        st.write("是否存在：", cache.exists())

if page == "训练":
    st.subheader("训练流水线命令")
    root_arg = f' --voc-root "{voc_root}"' if voc_root else ""
    st.code(f"python scripts/generate_proposals.py --config {config_path}{root_arg}", language="powershell")
    st.code(f"python scripts/train_full.py --config {config_path}{root_arg}", language="powershell")
    st.code(f"python scripts/evaluate.py --config {config_path}{root_arg}", language="powershell")
    st.info("完整 R-CNN 训练耗时较长，建议从终端启动，这样日志可见，任务也更容易中断和恢复。")
    st.write("模型与分类器检查点：", str(paths.checkpoints))
    st.write("评估报告目录：", str(paths.reports))

if page == "检测与评估":
    ds = dataset("image_set_eval")
    st.subheader("单图检测")
    uploaded = st.file_uploader("上传图片", type=["jpg", "jpeg", "png"])
    image_id = "上传图片"
    img = None
    if uploaded:
        data = np.frombuffer(uploaded.read(), np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    elif ds and len(ds.image_ids):
        image_id = st.selectbox("VOC 图像", ds.image_ids[:200])
        item = ds.get(image_id)
        img = cv2.imread(str(item.image_path), cv2.IMREAD_COLOR)
    if st.button("运行检测", disabled=img is None):
        try:
            model = load_model(str(paths.checkpoints / "alexnet_rcnn_finetuned.pt"), pretrained=cfg["model"].get("pretrained", True), device=device)
            svms = load_pickle(paths.checkpoints / "linear_svms.pkl")
            bbox_path = paths.checkpoints / "bbox_regressors.pkl"
            regressors = load_pickle(bbox_path) if bbox_path.exists() else {}
            dets = detect_image(img, image_id, model, svms, regressors=regressors, image_size=cfg["model"]["image_size"], proposal_mode=cfg["proposals"]["mode"], top_k=cfg["proposals"]["top_k"], min_size=cfg["proposals"]["min_size"], score_threshold=cfg["detection"]["score_threshold"], nms_iou=cfg["detection"]["nms_iou"], max_detections_per_class=cfg["detection"]["max_detections_per_class"], batch_size=cfg["training"]["batch_size"], device=device)
            vis = draw_detections(img, dets)
            st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption=f"{len(dets)} 个检测结果")
            st.dataframe(translate_detections(dets))
        except Exception as exc:
            st.error(str(exc))
    st.subheader("评估报告")
    reports = sorted(paths.reports.glob("*.json"))
    if reports:
        report = st.selectbox("报告文件", reports, format_func=lambda p: p.name)
        st.json(json.loads(Path(report).read_text(encoding="utf-8")))
    else:
        st.write("还没有评估报告 JSON 文件。")

if page == "神经元可视化":
    st.subheader("Top 激活区域")
    st.write("先使用 `scripts/export_neuron_cache.py` 扫描候选区域并导出神经元缓存，然后在这里查看缓存的 top 激活区域。")
    caches = sorted(paths.neurons.glob("*.json"))
    if not caches:
        st.code(f"python scripts/export_neuron_cache.py --config {config_path} --layer pool5 --channel 0 --top-k 16", language="powershell")
        st.write("还没有神经元缓存 JSON 文件。")
    else:
        cache = st.selectbox("激活缓存", caches, format_func=lambda p: p.name)
        hits = load_hits(cache)
        png = cache.with_suffix(".png")
        if not png.exists():
            render_top_regions(hits, png)
        st.image(str(png), caption="论文风格的 top 激活区域")
        st.dataframe([h.__dict__ for h in hits])
