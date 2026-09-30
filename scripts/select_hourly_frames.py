"""Seleksi data bersih: ambil satu citra pertama untuk setiap jam.

Kamera merekam tiap ~10 menit dengan nama file `YYYY-MM-DD_HH-MM-SS.jpg`.
Skrip ini menyalin frame pertama tiap jam ke folder tujuan
(versi skrip dari notebooks/01_data_cleaning.ipynb).

Contoh:
    python scripts/select_hourly_frames.py dataset/2024-08/raw dataset/2024-08/clean
"""
import argparse
import os
import shutil
from collections import defaultdict


def select_hourly(folder_asal, folder_tujuan):
    os.makedirs(folder_tujuan, exist_ok=True)

    # {tanggal: {jam: file}}
    file_terpilih = defaultdict(dict)
    for file in sorted(os.listdir(folder_asal)):
        if not file.endswith(".jpg"):
            continue
        try:
            tanggal, waktu = file.split("_")
            jam, _, _ = waktu.split(".")[0].split("-")
        except ValueError:
            continue
        # simpan file pertama pada jam tersebut
        file_terpilih[tanggal].setdefault(jam, file)

    jumlah = 0
    for jam_dict in file_terpilih.values():
        for f in jam_dict.values():
            shutil.copy2(os.path.join(folder_asal, f), os.path.join(folder_tujuan, f))
            jumlah += 1

    print("Selesai! Jumlah file terpilih:", jumlah)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder_asal")
    parser.add_argument("folder_tujuan")
    args = parser.parse_args()
    select_hourly(args.folder_asal, args.folder_tujuan)
