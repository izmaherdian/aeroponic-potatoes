"""Konversi anotasi poligon AnyLabeling (.json) menjadi mask PNG grayscale.

Nilai piksel mask:
    0   = latar belakang
    128 = umbi
    255 = akar (Akar 1 ... Akar 7)

Contoh:
    python scripts/json_to_mask.py dataset/2024-08/labels_json -o masks
"""
import argparse
import json
import os

import cv2
import numpy as np

IMG_WIDTH = 2560
IMG_HEIGHT = 1440

VALUE_UMBI = 128
VALUE_AKAR = 255


def json_to_mask(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    # Kanvas hitam
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)

    # Gambar poligon berdasarkan label
    for shape in data.get("shapes", []):
        label = shape["label"].lower()
        points = np.array(shape["points"], dtype=np.int32)
        color = VALUE_UMBI if label == "umbi" else VALUE_AKAR
        cv2.fillPoly(mask, [points], color)

    return mask


def main(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    json_files = sorted(f for f in os.listdir(input_folder) if f.endswith(".json"))

    for json_file in json_files:
        print(f"Memproses: {json_file}")
        mask = json_to_mask(os.path.join(input_folder, json_file))
        output_name = os.path.splitext(json_file)[0] + ".png"
        cv2.imwrite(os.path.join(output_folder, output_name), mask)

    print(f"Semua file selesai diproses. Hasil disimpan di: {output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input_folder", help="folder berisi file .json")
    parser.add_argument("-o", "--output-folder", default="masks")
    args = parser.parse_args()
    main(args.input_folder, args.output_folder)
