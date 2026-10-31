"""
Pengaturan bersama untuk skrip eksperimen: alamat berkas, random_state, daftar
model dan kondisi, kandidat hyperparameter, dan fungsi pembagian data.

Diimpor oleh skrip lain, tidak dijalankan langsung.
"""

import os
import platform
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

# ===================== ALAMAT BERKAS =====================
FOLDER_DATA = r"D:\Codes\Skripsi\dashboard_jaringan"
FOLDER_HASIL = "hasil"

BERKAS = {
    "DoS": "DoS1-Thursday-15-02-2018_TrafficForML_CICFlowMeter.parquet",
    "Infiltration": "Infil1-Wednesday-28-02-2018_TrafficForML_CICFlowMeter.parquet",
}

KOLOM_LABEL = "Label"
LABEL_NORMAL = "Benign"

# ===================== PENGATURAN EKSPERIMEN =====================
RANDOM_STATE = 42
PROPORSI_DATA_UJI = 0.3
JUMLAH_LIPATAN = 5
SCORING = "f1"

# Nama panjang model, dipakai pada keluaran dan nama berkas.
NAMA_MODEL = {
    "lr": "Logistic Regression",
    "dt": "Decision Tree",
    "rf": "Random Forest",
}

# Dua kondisi penanganan class imbalance.
# None berarti tanpa penanganan, "balanced" berarti class_weight='balanced'.
KONDISI = {
    "netral": None,
    "balanced": "balanced",
}

# Kandidat hyperparameter sesuai Tabel 3.4 pada naskah.
# Nama parameter untuk Logistic Regression memakai awalan "clf__" karena
# modelnya dibungkus dalam pipeline bersama StandardScaler.
KANDIDAT = {
    "lr": {"clf__C": [0.01, 0.1, 1, 10]},
    "dt": {"max_depth": [None, 5, 10, 15]},
    "rf": {"n_estimators": [100, 200]},
}

# Dinaikkan dari nilai bawaan agar Logistic Regression konvergen.
MAX_ITER_LR = 1000


# ===================== FUNGSI BANTU =====================
class Pencatat:
    """Mencetak teks ke layar sekaligus menyimpannya untuk ditulis ke berkas."""

    def __init__(self):
        self.baris = []

    def __call__(self, teks=""):
        teks = str(teks)
        print(teks)
        self.baris.append(teks)

    def garis(self, judul=""):
        self("")
        self("=" * 70)
        if judul:
            self(judul)
            self("=" * 70)

    def simpan(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as berkas:
            berkas.write("\n".join(self.baris))
        print(f"\nKeluaran disimpan ke: {os.path.abspath(path)}")


def folder(*bagian):
    """Membuat (bila belum ada) lalu mengembalikan alamat folder di dalam hasil/."""
    path = os.path.join(FOLDER_HASIL, *bagian)
    os.makedirs(path, exist_ok=True)
    return path


def angka(n):
    """Memformat bilangan dengan pemisah ribuan gaya Indonesia."""
    return f"{int(n):,}".replace(",", ".")


def persen(x, desimal=2):
    """Memformat pecahan 0-1 menjadi persen gaya Indonesia."""
    return f"{x * 100:.{desimal}f}".replace(".", ",") + "%"


def info_lingkungan():
    """Mengembalikan daftar baris berisi keterangan lingkungan pengerjaan."""
    return [
        f"Waktu mulai    : {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}",
        f"Sistem operasi : {platform.platform()}",
        f"Prosesor       : {platform.processor()}",
        f"Python         : {sys.version.split()[0]}",
        f"pandas         : {pd.__version__}",
        f"numpy          : {np.__version__}",
        f"scikit-learn   : {sklearn.__version__}",
    ]


def muat_data(skenario):
    """Memuat satu berkas parquet beserta pemeriksaan singkat kondisinya."""
    if skenario not in BERKAS:
        raise ValueError(f"Skenario '{skenario}' tidak dikenal. Pilihan: {list(BERKAS)}")

    path = os.path.join(FOLDER_DATA, BERKAS[skenario])
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Berkas tidak ditemukan: {path}\n"
            f"Periksa kembali nilai FOLDER_DATA pada konfigurasi.py."
        )
    return pd.read_parquet(path)


def kolom_fitur(df):
    """Mengembalikan daftar nama kolom fitur (seluruh kolom selain kolom label)."""
    return [k for k in df.columns if k != KOLOM_LABEL]


def label_biner(df):
    """Mengubah kolom Label menjadi 0 untuk Benign dan 1 untuk selain Benign."""
    teks = df[KOLOM_LABEL].astype(str)
    if LABEL_NORMAL not in set(teks.unique()):
        raise ValueError(
            f"Nilai '{LABEL_NORMAL}' tidak ditemukan pada kolom {KOLOM_LABEL}. "
            f"Nilai yang ada: {sorted(teks.unique())}"
        )
    return (teks != LABEL_NORMAL).astype(int)


def siapkan_data(skenario):
    """Memuat data, membentuk label biner, lalu membagi 70:30 secara berstrata."""
    df = muat_data(skenario)
    fitur = kolom_fitur(df)
    X = df[fitur]
    y = label_biner(df)

    X_latih, X_uji, y_latih, y_uji = train_test_split(
        X, y,
        test_size=PROPORSI_DATA_UJI,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return X_latih, X_uji, y_latih, y_uji, fitur
