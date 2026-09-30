"""Potong citra (sudah dikoreksi distorsi) menjadi strip vertikal per akar.

Setiap strip berukuran CROP_W x CROP_H dan berpusat pada koordinat x akar
yang ditentukan manual (lihat docs/root_label_map_camera2.pdf).

Contoh:
    python scripts/crop_root_strips.py koreksi_151.jpg -o crops --prefix crop15
"""
import argparse
import os

import cv2

# Ukuran crop
CROP_W = 160
CROP_H = 1440

# Titik tengah (koordinat x) tiap akar pada citra 2560x1440
DEFAULT_X_CENTERS = [281, 820, 905, 1038, 1387, 1465, 1540, 1889, 2003, 2084]


def crop_strips(image_path, output_dir, x_centers, prefix="crop"):
    os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    h, w = img.shape[:2]
    half_w = CROP_W // 2

    for i, x_center in enumerate(x_centers):
        # Hitung batas crop (tinggi penuh)
        x_start = max(0, x_center - half_w)
        x_end = min(w, x_center + half_w)
        crop = img[0:CROP_H, x_start:x_end]

        out_path = os.path.join(output_dir, f"{prefix}_{i + 1}.jpg")
        cv2.imwrite(out_path, crop)
        print(f"Crop {i + 1} disimpan di {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", help="citra input (hasil undistort)")
    parser.add_argument("-o", "--output-dir", default="crops")
    parser.add_argument("--prefix", default="crop")
    parser.add_argument("--x-centers", type=int, nargs="+", default=DEFAULT_X_CENTERS)
    args = parser.parse_args()
    crop_strips(args.image, args.output_dir, args.x_centers, args.prefix)
