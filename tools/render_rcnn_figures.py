from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "rcnn-notes"
OUT.mkdir(parents=True, exist_ok=True)

W, H = 1600, 900
BG = "#F7F8FA"
INK = "#17202A"
SUB = "#566573"
BLUE = "#1877C9"
TEAL = "#008F86"
RED = "#D64545"
GREEN = "#219653"
AMBER = "#D98B16"
PURPLE = "#7856C8"
GRID = "#D9E1E8"
WHITE = "#FFFFFF"


def font(size, bold=False):
    name = "arialbd.ttf" if bold else "arial.ttf"
    return ImageFont.truetype(str(Path("C:/Windows/Fonts") / name), size)


F_TITLE = font(48, True)
F_H2 = font(28, True)
F_BODY = font(23)
F_SMALL = font(19)
F_BOLD = font(23, True)


def canvas(title, subtitle):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.text((70, 48), title, fill=INK, font=F_TITLE)
    d.text((72, 108), subtitle, fill=SUB, font=F_BODY)
    d.line((70, 154, W - 70, 154), fill=GRID, width=3)
    return im, d


def rounded(d, box, fill=WHITE, outline=GRID, radius=14, width=2):
    d.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def centered(d, box, text, fill=INK, f=F_BODY):
    left, top, right, bottom = box
    bb = d.multiline_textbbox((0, 0), text, font=f, spacing=7, align="center")
    x = left + (right - left - (bb[2] - bb[0])) / 2
    y = top + (bottom - top - (bb[3] - bb[1])) / 2
    d.multiline_text((x, y), text, fill=fill, font=f, spacing=7, align="center")


def arrow(d, start, end, color=SUB, width=4):
    d.line((*start, *end), fill=color, width=width)
    x, y = end
    if abs(end[0] - start[0]) > abs(end[1] - start[1]):
        s = -1 if end[0] > start[0] else 1
        d.polygon([(x, y), (x + s * 16, y - 9), (x + s * 16, y + 9)], fill=color)
    else:
        s = -1 if end[1] > start[1] else 1
        d.polygon([(x, y), (x - 9, y + s * 16), (x + 9, y + s * 16)], fill=color)


def save(im, name):
    im.save(OUT / name, dpi=(144, 144))


def overview():
    im, d = canvas("R-CNN detection pipeline", "Inference: proposals first, CNN features second, class decisions last")
    boxes = [
        ("Input image", "pixels", BLUE),
        ("Selective Search", "~2000 proposals", TEAL),
        ("Warp + CNN", "227 x 227 -> 4096-D", PURPLE),
        ("Linear SVMs", "score each class", AMBER),
        ("BB reg. + NMS", "final boxes", GREEN),
    ]
    x, y, bw, bh, gap = 70, 350, 258, 155, 37
    for i, (a, b, color) in enumerate(boxes):
        box = (x + i * (bw + gap), y, x + i * (bw + gap) + bw, y + bh)
        rounded(d, box, fill=WHITE, outline=color, width=4)
        centered(d, (box[0], box[1] + 20, box[2], box[3] - 42), a, f=F_H2)
        centered(d, (box[0], box[3] - 55, box[2], box[3] - 12), b, fill=color, f=F_SMALL)
        if i < len(boxes) - 1:
            arrow(d, (box[2] + 8, y + bh / 2), (box[2] + gap - 8, y + bh / 2), BLUE)
    d.text((70, 650), "Key idea:", font=F_H2, fill=INK)
    d.text((245, 653), "CNN features are computed for each proposed region, then shared across all class SVMs.", font=F_BODY, fill=SUB)
    d.text((70, 730), "Training adds: ImageNet pre-training -> VOC fine-tuning -> SVM training -> class-specific box regressors.", font=F_BODY, fill=SUB)
    save(im, "rcnn-overview.png")


def nms_iou():
    im, d = canvas("Score is not IoU: greedy non-maximum suppression", "One class ('car'): scores rank boxes; IoU removes duplicate boxes")
    panel = (75, 225, 650, 750)
    rounded(d, panel, fill=WHITE)
    d.text((100, 250), "Candidate detections", font=F_H2, fill=INK)
    d.rectangle((155, 360, 490, 600), outline=BLUE, width=6)
    d.text((160, 325), "A  score = 0.95   KEEP", fill=BLUE, font=F_BOLD)
    d.rectangle((190, 380, 515, 615), outline=RED, width=5)
    d.text((200, 625), "B  score = 0.82", fill=RED, font=F_BODY)
    d.rectangle((360, 485, 600, 680), outline=AMBER, width=5)
    d.text((377, 688), "C  score = 0.61", fill=AMBER, font=F_BODY)
    d.text((105, 785), "A, B overlap heavily; C may be another car.", fill=SUB, font=F_SMALL)
    right = (745, 225, 1525, 750)
    rounded(d, right, fill=WHITE)
    d.text((780, 254), "Greedy steps (example threshold = 0.50)", fill=INK, font=F_H2)
    steps = [
        ("1", "Sort by SVM score", "A 0.95 > B 0.82 > C 0.61", BLUE),
        ("2", "Select highest score A", "A becomes a kept detection", GREEN),
        ("3", "Compare overlap", "IoU(A, B) = 0.67 > 0.50", RED),
        ("4", "Suppress duplicate", "Delete B; continue testing C", PURPLE),
    ]
    y = 350
    for num, heading, detail, color in steps:
        d.ellipse((790, y, 832, y + 42), fill=color)
        centered(d, (790, y, 832, y + 42), num, fill=WHITE, f=F_BOLD)
        d.text((858, y - 2), heading, fill=INK, font=F_BOLD)
        d.text((858, y + 33), detail, fill=SUB, font=F_BODY)
        y += 91
    d.text((785, 722), "IoU = area(intersection) / area(union)", fill=TEAL, font=F_BOLD)
    save(im, "nms-iou-score.png")


def training():
    im, d = canvas("R-CNN is trained in stages", "The labels and thresholds are intentionally different across stages")
    stages = [
        ("1  Pre-train CNN", "ILSVRC classification\nimage labels only", "1000-way softmax", BLUE),
        ("2  Fine-tune CNN", "VOC warped proposals\nIoU >= 0.5 positive", "(N + 1)-way softmax", TEAL),
        ("3  Train SVMs", "GT boxes positive\nIoU < 0.3 negative", "one SVM / class", AMBER),
        ("4  Box regression", "nearby proposals\nIoU > 0.6", "one regressor / class", GREEN),
    ]
    x, y, bw, bh, gap = 62, 250, 335, 290, 43
    for i, (title, sample, output, color) in enumerate(stages):
        b = (x + i * (bw + gap), y, x + i * (bw + gap) + bw, y + bh)
        rounded(d, b, outline=color, width=4)
        d.rectangle((b[0], b[1], b[2], b[1] + 67), fill=color)
        centered(d, (b[0] + 8, b[1] + 8, b[2] - 8, b[1] + 60), title, fill=WHITE, f=F_H2)
        centered(d, (b[0] + 12, b[1] + 95, b[2] - 12, b[1] + 190), sample, f=F_BODY)
        d.line((b[0] + 25, b[1] + 207, b[2] - 25, b[1] + 207), fill=GRID, width=2)
        centered(d, (b[0], b[1] + 219, b[2], b[3] - 14), output, fill=color, f=F_BOLD)
        if i < len(stages) - 1:
            arrow(d, (b[2] + 9, y + 145), (b[2] + gap - 8, y + 145), BLUE)
    d.text((70, 655), "Important:", font=F_H2, fill=RED)
    d.text((254, 658), "this is not a single end-to-end loss; features, classifiers, and regressors are learned separately.", font=F_BODY, fill=SUB)
    d.text((70, 730), "The SVM grey zone (0.3 <= IoU < 1, except GT positives) is ignored during SVM training.", font=F_BODY, fill=SUB)
    save(im, "training-stages.png")


def pool5():
    im, d = canvas("AlexNet/Caffe path to pool5", "Spatial sizes use padding; receptive field (RF) grows independently of padding")
    layers = [
        ("input", "227", "RF 1", INK),
        ("conv1\n11/s4 p0", "55", "RF 11", BLUE),
        ("pool1\n3/s2", "27", "RF 19", TEAL),
        ("conv2\n5/s1 p2", "27", "RF 51", BLUE),
        ("pool2\n3/s2", "13", "RF 67", TEAL),
        ("conv3-5\n3/s1 p1", "13", "RF 163", BLUE),
        ("pool5\n3/s2", "6", "RF 195", PURPLE),
    ]
    x, y, bw, bh, gap = 65, 305, 185, 190, 28
    for i, (name, dim, rf, color) in enumerate(layers):
        b = (x + i * (bw + gap), y, x + i * (bw + gap) + bw, y + bh)
        rounded(d, b, fill=WHITE, outline=color, width=4)
        centered(d, (b[0], b[1] + 18, b[2], b[1] + 90), name, f=F_BOLD)
        centered(d, (b[0], b[1] + 96, b[2], b[1] + 140), dim + " x " + dim, fill=color, f=F_H2)
        centered(d, (b[0], b[1] + 147, b[2], b[3] - 10), rf, fill=SUB, f=F_SMALL)
        if i < len(layers) - 1:
            arrow(d, (b[2] + 8, y + bh / 2), (b[2] + gap - 7, y + bh / 2), SUB)
    rounded(d, (370, 610, 1230, 780), fill="#EAF3FB", outline=BLUE)
    centered(d, (390, 628, 1210, 690), "pool5 output = 6 x 6 x 256 = 9216 values", fill=BLUE, f=F_H2)
    centered(d, (390, 704, 1210, 760), "one unit observes 195 x 195 pixels of the 227 x 227 input", fill=INK, f=F_BODY)
    save(im, "pool5-shape-receptive-field.png")


def ablation():
    im, d = canvas("Layer ablation on VOC 2007 test", "mAP from the paper: before and after detection fine-tuning")
    rounded(d, (75, 238, 610, 730), fill=WHITE)
    d.text((112, 275), "CNN representation", font=F_H2, fill=INK)
    nodes = [
        ("pool5", "6 x 6 x 256 = 9216", PURPLE),
        ("fc6", "4096", TEAL),
        ("fc7", "4096", BLUE),
    ]
    y = 365
    for i, (n, dim, c) in enumerate(nodes):
        rounded(d, (160, y, 520, y + 84), fill=WHITE, outline=c, width=4)
        d.text((190, y + 17), n, fill=c, font=F_H2)
        d.text((335, y + 24), dim, fill=SUB, font=F_SMALL)
        if i < 2:
            arrow(d, (340, y + 84), (340, y + 121), c)
        y += 122
    d.text((100, 696), "Read out features at any marked layer", fill=SUB, font=F_SMALL)
    d.text((752, 265), "Feature", font=F_H2, fill=INK)
    d.text((990, 265), "No fine-tune", font=F_H2, fill=INK)
    d.text((1260, 265), "Fine-tuned", font=F_H2, fill=INK)
    rows = [("pool5", 44.2, 47.3, PURPLE), ("fc6", 46.2, 53.1, TEAL), ("fc7", 44.7, 54.2, BLUE), ("fc7 + BB", None, 58.5, GREEN)]
    y = 370
    scale = 5.8
    for label, before, after, color in rows:
        d.text((752, y + 14), label, fill=color, font=F_BOLD)
        if before is not None:
            d.rectangle((982, y + 14, 982 + before * scale, y + 45), fill="#D5DDE5")
            d.text((992, y + 17), f"{before:.1f}", fill=INK, font=F_SMALL)
        d.rectangle((1250, y + 14, 1250 + after * 4.1, y + 45), fill=color)
        d.text((1260, y + 17), f"{after:.1f}", fill=WHITE, font=F_SMALL)
        y += 88
    d.text((753, 723), "Fine-tuning helps fc6/fc7 most; BB fixes localization.", fill=SUB, font=F_BODY)
    save(im, "ablation-layers.png")


def localization():
    im, d = canvas("Localization error and bounding-box regression", "Classification can be right while the predicted coordinates are wrong")
    panels = [("Before regression: localization FP", RED), ("After regression: improved IoU", GREEN)]
    for idx, (label, color) in enumerate(panels):
        left = 86 + idx * 755
        rounded(d, (left, 245, left + 660, 730), fill=WHITE)
        d.text((left + 34, 274), label, fill=color, font=F_H2)
        d.rectangle((left + 110, 390, left + 480, 630), outline=BLUE, width=6)
        d.text((left + 112, 350), "Ground truth G", font=F_BODY, fill=BLUE)
        if idx == 0:
            d.rectangle((left + 55, 430, left + 380, 670), outline=RED, width=6)
            d.text((left + 55, 680), "proposal P: offset, low IoU", font=F_BODY, fill=RED)
            arrow(d, (left + 575, 505), (left + 575, 430), RED)
            d.text((left + 505, 530), "dx, dy", font=F_SMALL, fill=RED)
        else:
            d.rectangle((left + 98, 400, left + 470, 638), outline=GREEN, width=6)
            d.text((left + 100, 680), "predicted box G-hat", font=F_BODY, fill=GREEN)
            d.text((left + 113, 640), "higher overlap", font=F_SMALL, fill=GREEN)
    d.text((92, 797), "Regressor predicts: center shift (dx, dy) and log-scale change (dw, dh) from pool5 features.", fill=SUB, font=F_BODY)
    save(im, "localization-bbox-regression.png")


if __name__ == "__main__":
    overview()
    nms_iou()
    training()
    pool5()
    ablation()
    localization()
    print(f"Rendered six diagrams to {OUT}")
