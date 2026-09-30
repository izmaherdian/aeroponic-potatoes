"""Membuat seluruh gambar di docs/images/ untuk README.

Sumber: folder dataset/ (lokal, tidak di-push), output yang tersimpan di
notebooks/*.ipynb, dan docs/root_label_map_camera2.pdf.

    python scripts/make_readme_figures.py
"""
import base64
import collections
import glob
import json
import os
import re
import subprocess

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(ROOT, "dataset")
OUT = os.path.join(ROOT, "docs", "images")
MONTHS = ["2024-07", "2024-08", "2024-09"]
MONTH_NAMES = {"2024-07": "Juli", "2024-08": "Agustus", "2024-09": "September"}
CLASSES = ["Umbi"] + [f"Akar {i}" for i in range(1, 8)]

# Palet: 8 warna kategorikal (BGR untuk OpenCV)
PALETTE_HEX = ["#e8a33d", "#3987e5", "#d64b4b", "#3fae6a", "#9b59d0",
               "#1bb3b3", "#e36fb5", "#8fae2a"]


def hex_to_bgr(h):
    h = h.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


CLASS_BGR = {c: hex_to_bgr(PALETTE_HEX[i]) for i, c in enumerate(CLASSES)}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.dpi": 120,
})


def save_jpg(path, img, width=None, quality=85):
    if width and img.shape[1] > width:
        h = int(img.shape[0] * width / img.shape[1])
        img = cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA)
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    print("saved", os.path.relpath(path, ROOT))


def put_label(img, text, org=(20, 50), scale=1.4):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 8, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 3, cv2.LINE_AA)


def notebook_images(nb_path, cell):
    nb = json.load(open(nb_path))
    imgs = []
    for o in nb["cells"][cell].get("outputs", []):
        for mt in ("image/png", "image/jpeg"):
            if mt in o.get("data", {}):
                buf = np.frombuffer(base64.b64decode("".join(o["data"][mt])), np.uint8)
                imgs.append(cv2.imdecode(buf, cv2.IMREAD_COLOR))
    return imgs


def first(pattern):
    return sorted(glob.glob(os.path.join(DATASET, pattern)))[0]


# ---------------------------------------------------------------- pipeline
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(14, 6.2))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    ax.grid(False)

    lanes = [
        (5.0, "Akuisisi", "#3987e5"),
        (3.1, "Pra-proses & Anotasi", "#3fae6a"),
        (1.2, "Model & Pengukuran", "#e8a33d"),
    ]
    for y, name, col in lanes:
        ax.add_patch(FancyBboxPatch((0.1, y - 0.75), 13.8, 1.5, boxstyle="round,pad=0.02,rounding_size=0.15",
                                    fc=col, alpha=0.07, ec="none"))
        ax.text(0.45, y, name.replace(" & ", " &\n"), color=col, fontsize=10, fontweight="bold",
                va="center", ha="center", rotation=90)

    def box(x, y, title, sub, col):
        ax.add_patch(FancyBboxPatch((x - 1.15, y - 0.5), 2.3, 1.0, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc="white", ec=col, lw=2))
        ax.text(x, y + 0.14, title, ha="center", va="center", fontsize=10.5, fontweight="bold", color="#222")
        ax.text(x, y - 0.22, sub, ha="center", va="center", fontsize=8.5, color="#555")
        return (x, y)

    def arrow(p, q, rad=0.0):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=16, lw=1.6, color="#666",
                                     connectionstyle=f"arc3,rad={rad}"))

    b, g, o = "#3987e5", "#3fae6a", "#e8a33d"
    a1 = box(2.2, 5.0, "Kamera 2", "ruang akar aeroponik", b)
    a2 = box(5.4, 5.0, "Citra mentah", "2560×1440, tiap ±10 menit", b)
    a3 = box(8.6, 5.0, "Kalibrasi kamera", "checkerboard 8×6, 40 mm", b)
    a4 = box(11.8, 5.0, "Timelapse", "Jul – Sep 2024", b)

    p1 = box(2.2, 3.1, "Seleksi data", "1 frame per jam", g)
    p2 = box(5.4, 3.1, "Undistort", "koreksi distorsi lensa", g)
    p3 = box(8.6, 3.1, "Anotasi poligon", "AnyLabeling: Umbi, Akar 1–7", g)
    p4 = box(11.8, 3.1, "Konversi label", "YOLO txt & mask PNG", g)

    # baris bawah mengalir dari kanan ke kiri (snake layout)
    m1 = box(11.8, 1.2, "YOLO11s / U-Net", "deteksi bbox / segmentasi strip", o)
    m2 = box(8.6, 1.2, "Post-processing", "tinggi bbox / kontur / skeleton", o)
    m3 = box(5.4, 1.2, "Konversi skala", "0,0511 cm/px (kalibrasi)", o)
    m4 = box(2.2, 1.2, "Panjang akar (cm)", "per akar, per waktu", o)

    arrow((a1[0] + 1.15, 5.0), (a2[0] - 1.15, 5.0))
    arrow((a2[0] + 1.15, 5.0), (a3[0] - 1.15, 5.0))
    arrow((a3[0] + 1.15, 5.0), (a4[0] - 1.15, 5.0))
    arrow((a2[0] - 0.4, 4.5), (p1[0] + 0.4, 3.6))
    arrow((a3[0] - 0.4, 4.5), (p2[0] + 0.4, 3.6))
    arrow((p1[0] + 1.15, 3.1), (p2[0] - 1.15, 3.1))
    arrow((p2[0] + 1.15, 3.1), (p3[0] - 1.15, 3.1))
    arrow((p3[0] + 1.15, 3.1), (p4[0] - 1.15, 3.1))
    arrow((p4[0], 2.6), (m1[0], 1.7))
    arrow((m1[0] - 1.15, 1.2), (m2[0] + 1.15, 1.2))
    arrow((m2[0] - 1.15, 1.2), (m3[0] + 1.15, 1.2))
    arrow((m3[0] - 1.15, 1.2), (m4[0] + 1.15, 1.2))

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "pipeline.png"), dpi=150, facecolor="white")
    plt.close(fig)
    print("saved docs/images/pipeline.png")


# ---------------------------------------------------------------- growth
def fig_growth():
    picks = [("2024-07", "2024-07-27"), ("2024-08", "2024-08-15"), ("2024-09", "2024-09-20")]
    rows = []
    for m, day in picks:
        row = []
        for hh, tag in (("12", "siang"), ("00", "malam")):
            f = first(f"{m}/clean/{day}_{hh}-*.jpg")
            img = cv2.resize(cv2.imread(f), (960, 540), interpolation=cv2.INTER_AREA)
            img[0:44, 0:330] = 0
            put_label(img, f"{day} {hh}:00 ({tag})", (15, 32), 0.9)
            row.append(img)
        rows.append(np.hstack([row[0], np.full((540, 8, 3), 255, np.uint8), row[1]]))
    sep = np.full((8, rows[0].shape[1], 3), 255, np.uint8)
    grid = np.vstack([rows[0], sep, rows[1], sep, rows[2]])
    save_jpg(os.path.join(OUT, "growth_by_month.jpg"), grid, width=1600)


# ---------------------------------------------------------------- annotation
def draw_polygons(img, shapes, alpha=0.45):
    overlay = img.copy()
    for s in shapes:
        pts = np.array(s["points"], np.int32)
        cv2.fillPoly(overlay, [pts], CLASS_BGR.get(s["label"], (200, 200, 200)))
    out = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    for s in shapes:
        pts = np.array(s["points"], np.int32)
        cv2.polylines(out, [pts], True, CLASS_BGR.get(s["label"], (200, 200, 200)), 3, cv2.LINE_AA)
    return out


def legend_strip(width, height=70):
    strip = np.full((height, width, 3), 255, np.uint8)
    x = 20
    for c in CLASSES:
        cv2.rectangle(strip, (x, 22), (x + 30, 48), CLASS_BGR[c], -1)
        cv2.putText(strip, c, (x + 40, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 40, 40), 2, cv2.LINE_AA)
        x += 40 + cv2.getTextSize(c, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0][0] + 40
    return strip


def fig_annotation():
    base = "2024-09/clean/2024-09-20_12-00-12"
    img = cv2.imread(os.path.join(DATASET, base + ".jpg"))
    shapes = json.load(open(os.path.join(DATASET, base + ".json")))["shapes"]

    poly = draw_polygons(img, shapes)

    mask = np.zeros(img.shape[:2], np.uint8)
    for s in shapes:
        cv2.fillPoly(mask, [np.array(s["points"], np.int32)], 128 if s["label"].lower() == "umbi" else 255)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # label YOLO (bbox) dari folder labels_yolo
    yolo = img.copy()
    names = open(os.path.join(DATASET, "2024-09/labels_yolo/classes.txt")).read().split("\n")
    names = [n.strip() for n in names if n.strip()]
    h, w = img.shape[:2]
    lbl = os.path.join(DATASET, "2024-09/labels_yolo/labels", os.path.basename(base) + ".txt")
    for line in open(lbl):
        c, cx, cy, bw, bh = line.split()
        cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
        col = CLASS_BGR[names[int(c)]]
        p1 = (int(cx - bw / 2), int(cy - bh / 2))
        cv2.rectangle(yolo, p1, (int(cx + bw / 2), int(cy + bh / 2)), col, 4)
        if names[int(c)] != "Umbi":
            cv2.putText(yolo, names[int(c)], (p1[0], max(30, p1[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        1.1, col, 3, cv2.LINE_AA)

    tiles = []
    for im, t in ((img, "a) Citra bersih"), (poly, "b) Anotasi poligon (JSON)"),
                  (yolo, "c) Label YOLO (bbox)"), (mask, "d) Mask segmentasi (PNG)")):
        im = im.copy()
        im[0:120, 0:1150] = 0  # tutup timestamp OSD kamera
        put_label(im, t, (30, 85), 2.2)
        tiles.append(cv2.resize(im, (1280, 720), interpolation=cv2.INTER_AREA))
    vs = np.full((720, 10, 3), 255, np.uint8)
    top = np.hstack([tiles[0], vs, tiles[1]])
    bot = np.hstack([tiles[2], vs, tiles[3]])
    grid = np.vstack([top, np.full((10, top.shape[1], 3), 255, np.uint8), bot, legend_strip(top.shape[1])])
    save_jpg(os.path.join(OUT, "annotation_formats.jpg"), grid, width=1800)


# ---------------------------------------------------------------- label map
def fig_label_map():
    pdf = os.path.join(ROOT, "docs", "root_label_map_camera2.pdf")
    tmp = os.path.join(OUT, "_label_map")
    subprocess.run(["pdftoppm", "-jpeg", "-r", "80", "-singlefile", pdf, tmp], check=True)
    img = cv2.imread(tmp + ".jpg")
    os.remove(tmp + ".jpg")
    save_jpg(os.path.join(OUT, "root_label_map.jpg"), img, width=1600)


# ---------------------------------------------------------------- dataset stats
def dataset_stats():
    stats = {}
    for m in MONTHS:
        raw = [f for f in os.listdir(os.path.join(DATASET, m, "raw")) if f.startswith(m) and f.endswith(".jpg")]
        clean = [f for f in os.listdir(os.path.join(DATASET, m, "clean")) if f.endswith(".jpg")]
        jsons = glob.glob(os.path.join(DATASET, m, "labels_json", "*.json"))
        counts = collections.Counter()
        for j in jsons:
            for s in json.load(open(j))["shapes"]:
                counts[s["label"]] += 1
        days = sorted({f[:10] for f in raw})
        stats[m] = dict(raw=len(raw), clean=len(clean), labeled=len(jsons), counts=counts,
                        first=days[0], last=days[-1], days=len(days))
    return stats


def fig_dataset(stats):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1, 1.5]})
    x = np.arange(len(MONTHS))
    wbar = 0.26
    series = [("raw", "Citra mentah", "#9aa5b1"), ("clean", "Data bersih (1/jam)", "#3987e5"),
              ("labeled", "Teranotasi", "#e8a33d")]
    for i, (k, name, col) in enumerate(series):
        vals = [stats[m][k] for m in MONTHS]
        bars = ax1.bar(x + (i - 1) * wbar, vals, wbar, label=name, color=col)
        ax1.bar_label(bars, fontsize=8.5, padding=2)
    ax1.set_xticks(x, [MONTH_NAMES[m] for m in MONTHS])
    ax1.set_ylabel("Jumlah citra")
    ax1.set_title("Jumlah citra per bulan", loc="left", fontweight="bold")
    ax1.legend(frameon=False, fontsize=9)
    ax1.grid(axis="x", visible=False)

    bottoms = np.zeros(len(CLASSES))
    month_cols = ["#b7d3f5", "#3987e5", "#1d4f91"]
    for m, col in zip(MONTHS, month_cols):
        vals = np.array([stats[m]["counts"].get(c, 0) for c in CLASSES])
        ax2.bar(CLASSES, vals, bottom=bottoms, color=col, label=MONTH_NAMES[m], width=0.65)
        bottoms += vals
    for i, v in enumerate(bottoms):
        ax2.text(i, v + 150, f"{int(v):,}".replace(",", "."), ha="center", fontsize=9)
    ax2.set_yscale("log")
    ax2.set_ylim(500, 40000)
    ax2.set_ylabel("Jumlah poligon (skala log)")
    ax2.set_title("Jumlah instance anotasi per kelas", loc="left", fontweight="bold")
    ax2.legend(frameon=False, fontsize=9, ncol=3, loc="upper right")
    ax2.grid(axis="x", visible=False)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "dataset_stats.png"), dpi=150, facecolor="white")
    plt.close(fig)
    print("saved docs/images/dataset_stats.png")


# ---------------------------------------------------------------- calibration
def fig_calibration():
    nb = os.path.join(ROOT, "notebooks", "02_camera_calibration.ipynb")
    imgs = notebook_images(nb, 0)
    pick = [cv2.resize(imgs[i], (1280, 720), interpolation=cv2.INTER_AREA) for i in (0, 4)]
    vs = np.full((720, 10, 3), 255, np.uint8)
    save_jpg(os.path.join(OUT, "camera_calibration.jpg"), np.hstack([pick[0], vs, pick[1]]), width=1600)


# ---------------------------------------------------------------- yolo
def parse_yolo_log():
    nb = json.load(open(os.path.join(ROOT, "notebooks", "03_yolo_root_detection.ipynb")))
    txt = "".join("".join(o.get("text", "")) for o in nb["cells"][5]["outputs"])
    txt = re.sub(r"\x1b\[[0-9;]*m", "", txt)
    rows = []
    epoch_re = re.compile(r"^\s*(\d+)/60\s+\S+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\d+\s+640")
    val_re = re.compile(r"^\s*all\s+20\s+20\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)")
    cur = None
    for line in re.split(r"[\r\n]", txt):
        m = epoch_re.match(line)
        if m:
            cur = [int(m.group(1))] + [float(v) for v in m.groups()[1:]]
            continue
        v = val_re.match(line)
        if v and cur is not None:
            if rows and rows[-1][0] == cur[0]:
                continue
            rows.append(cur + [float(g) for g in v.groups()])
            cur = None
    return np.array(rows)  # epoch, box, cls, dfl, P, R, mAP50, mAP50-95


def fig_yolo_training(log):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.3))
    ep = log[:, 0]
    for i, (name, col) in enumerate([("box loss", "#3987e5"), ("cls loss", "#d64b4b"), ("dfl loss", "#3fae6a")]):
        ax1.plot(ep, log[:, 1 + i], color=col, lw=2, label=name)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (train)")
    ax1.set_title("Loss pelatihan YOLO11s", loc="left", fontweight="bold")
    ax1.legend(frameon=False)
    ax1.set_ylim(0, 3)

    for i, (name, col) in enumerate([("Precision", "#9aa5b1"), ("Recall", "#e8a33d"),
                                     ("mAP@50", "#3987e5"), ("mAP@50-95", "#d64b4b")]):
        ax2.plot(ep, log[:, 4 + i], color=col, lw=2, label=name)
    best = int(np.argmax(log[:, 7]))
    ax2.scatter(ep[best], log[best, 7], color="#d64b4b", zorder=5)
    ax2.annotate(f"terbaik: {log[best, 7]:.3f} (epoch {int(ep[best])})", (ep[best], log[best, 7]),
                 xytext=(-150, -45), textcoords="offset points", fontsize=9,
                 arrowprops=dict(arrowstyle="-", color="#888"))
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Skor (validasi)")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Metrik validasi", loc="left", fontweight="bold")
    ax2.legend(frameon=False, loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "yolo_training_curves.png"), dpi=150, facecolor="white")
    plt.close(fig)
    print("saved docs/images/yolo_training_curves.png")


def fig_yolo_predictions():
    nb = os.path.join(ROOT, "notebooks", "03_yolo_root_detection.ipynb")
    det = notebook_images(nb, 7)
    cm = notebook_images(nb, 15)
    size = (1057, 418)
    rows = []
    for i in (0, 3, 6):
        rows.append(np.hstack([cv2.resize(det[i], size), np.full((418, 8, 3), 255, np.uint8),
                               cv2.resize(cm[i], size)]))
    sep = np.full((8, rows[0].shape[1], 3), 255, np.uint8)
    save_jpg(os.path.join(OUT, "yolo_predictions.jpg"), np.vstack([rows[0], sep, rows[1], sep, rows[2]]),
             width=1600)


# ---------------------------------------------------------------- u-net
def crop_panels(img, n):
    """Ambil panel gambar dari figure matplotlib (subplot bersebelahan, latar putih)."""
    nonwhite = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) < 235
    cols = np.where(nonwhite.sum(axis=0) > img.shape[0] * 0.5)[0]
    groups = [g for g in np.split(cols, np.where(np.diff(cols) > 5)[0] + 1) if len(g) > 10]
    panels = []
    for g in groups[:n]:
        sub = img[:, g[0]:g[-1] + 1]
        rows = np.where((cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY) < 235).mean(axis=1) > 0.9)[0]
        panels.append(sub[rows[0]:rows[-1] + 1])
    return panels


def fig_unet():
    nb = os.path.join(ROOT, "notebooks", "05_unet_inference_root_length.ipynb")
    pred = crop_panels(notebook_images(nb, 3)[0], 2)
    skel = crop_panels(notebook_images(nb, 6)[0], 3)
    titles = ["Strip input", "Prediksi U-Net", "Mask akar", "Skeleton", "Overlay"]
    panels = pred + skel
    H = 460
    fig, axes = plt.subplots(1, 5, figsize=(10, 5.2))
    for ax, p, t in zip(axes, panels, titles):
        p = cv2.resize(p, (max(1, int(p.shape[1] * H / p.shape[0])), H), interpolation=cv2.INTER_NEAREST)
        ax.imshow(cv2.cvtColor(p, cv2.COLOR_BGR2RGB))
        ax.set_title(t, fontsize=11)
        ax.axis("off")
    fig.text(0.21, 0.02, "Contoh 1 · segmentasi 1 akar", ha="center", fontsize=9.5, color="#555")
    fig.text(0.70, 0.02, "Contoh 2 · analisis skeleton (24 cabang)", ha="center", fontsize=9.5, color="#555")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(OUT, "unet_inference.png"), dpi=150, facecolor="white")
    plt.close(fig)
    print("saved docs/images/unet_inference.png")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_pipeline()
    fig_growth()
    fig_annotation()
    fig_label_map()
    s = dataset_stats()
    json.dump({m: {k: (dict(v) if isinstance(v, collections.Counter) else v) for k, v in d.items()}
               for m, d in s.items()}, open(os.path.join(OUT, "dataset_stats.json"), "w"), indent=2)
    fig_dataset(s)
    fig_calibration()
    log = parse_yolo_log()
    np.savetxt(os.path.join(OUT, "yolo_training_log.csv"), log, delimiter=",", fmt="%.5g",
               header="epoch,box_loss,cls_loss,dfl_loss,precision,recall,mAP50,mAP50_95", comments="")
    fig_yolo_training(log)
    fig_yolo_predictions()
    fig_unet()
