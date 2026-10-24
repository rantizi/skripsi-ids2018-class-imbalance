"""
Uji coba awal: Random Forest pada dua skenario biner (Normal vs DoS dan
Normal vs Infiltration) dari dataset CSE-CIC-IDS2018.

Menghitung akurasi, precision, recall, F1, confusion matrix, dan tiga
pemeriksaan awal potensi kebocoran data.

Jalankan: python 02_uji_coba_awal_rf.py
Hasil disimpan di folder hasil/
"""

import os
import platform
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# ===================== KONFIGURASI =====================
FOLDER_DATA = r"D:\Codes\Skripsi\dashboard_jaringan"
FOLDER_OUTPUT = "hasil"

BERKAS = {
    "DoS": "DoS1-Thursday-15-02-2018_TrafficForML_CICFlowMeter.parquet",
    "Infiltration": "Infil1-Wednesday-28-02-2018_TrafficForML_CICFlowMeter.parquet",
}

KOLOM_LABEL = "Label"
LABEL_NORMAL = "Benign"

PROPORSI_DATA_UJI = 0.3
RANDOM_STATE = 42
JUMLAH_POHON = 100

# Jumlah inti prosesor yang dipakai Random Forest.
# -1 berarti seluruh inti. Nilai ini hanya memengaruhi kecepatan, bukan hasil.
# Kalau muncul MemoryError, ubah jadi 2.
JUMLAH_INTI = -1
# =======================================================


def buat_pencatat():
    """Membuat fungsi pencatat baru untuk satu skenario (keluaran terpisah per berkas)."""
    baris_keluaran = []

    def catat(teks=""):
        teks = str(teks)
        print(teks)
        baris_keluaran.append(teks)

    return catat, baris_keluaran


def angka(n):
    """Memformat bilangan dengan pemisah ribuan gaya Indonesia. 223050 -> 223.050"""
    return f"{int(n):,}".replace(",", ".")


def persen(x):
    """Memformat persentase gaya Indonesia. 99.9874 -> 99,99%"""
    return f"{x * 100:.2f}".replace(".", ",") + "%"


def jalankan_skenario(nama_skenario, nama_berkas):
    """Menjalankan seluruh tahapan uji coba awal untuk satu skenario."""
    catat, baris_keluaran = buat_pencatat()
    waktu_total = time.perf_counter()
    path = os.path.join(FOLDER_DATA, nama_berkas)

    catat("=" * 70)
    catat(f"STUDI KELAYAKAN AWAL - SKENARIO {nama_skenario.upper()}")
    catat("Model: Random Forest tanpa penanganan class imbalance")
    catat("=" * 70)

    # ---------- Informasi lingkungan ----------
    catat()
    catat("--- Informasi lingkungan pengerjaan ---")
    catat(f"Waktu mulai    : {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")
    catat(f"Sistem operasi : {platform.platform()}")
    catat(f"Prosesor       : {platform.processor()}")
    catat(f"Python         : {sys.version.split()[0]}")
    catat(f"pandas         : {pd.__version__}")
    catat(f"numpy          : {np.__version__}")
    catat(f"scikit-learn   : {sklearn.__version__}")
    catat(f"Berkas data    : {nama_berkas}")

    # ---------- 1. Memuat data ----------
    catat()
    catat("--- 1. Memuat data ---")
    waktu = time.perf_counter()
    df = pd.read_parquet(path)
    catat(f"Waktu muat     : {time.perf_counter() - waktu:.2f} detik")
    catat(f"Jumlah baris   : {angka(df.shape[0])}")
    catat(f"Jumlah kolom   : {df.shape[1]}")

    # ---------- 2. Pemeriksaan ulang kondisi data ----------
    catat()
    catat("--- 2. Pemeriksaan ulang kondisi data ---")
    kolom_fitur = [k for k in df.columns if k != KOLOM_LABEL]

    jumlah_duplikat = int(df.duplicated().sum())
    jumlah_kosong = int(df.isna().sum().sum())
    jumlah_infinity = 0
    for kolom in kolom_fitur:
        if pd.api.types.is_float_dtype(df[kolom]):
            jumlah_infinity += int(np.isinf(df[kolom].to_numpy()).sum())

    catat(f"Duplikasi baris: {angka(jumlah_duplikat)}")
    catat(f"Missing value  : {angka(jumlah_kosong)}")
    catat(f"Nilai infinity : {angka(jumlah_infinity)}")

    if jumlah_duplikat > 0 or jumlah_kosong > 0 or jumlah_infinity > 0:
        catat()
        catat("PROSES DIHENTIKAN.")
        catat("Ditemukan nilai bermasalah pada data. Tidak ada baris yang dibuang oleh")
        catat("skrip ini, karena penanganannya perlu ditentukan terlebih dahulu agar")
        catat("jumlah data tetap dapat dipertanggungjawabkan.")
        simpan(nama_skenario, baris_keluaran)
        return None

    catat("Kondisi data   : bersih, seluruh baris dipakai tanpa pembuangan.")

    # ---------- 3. Pembentukan label biner ----------
    catat()
    catat(f"--- 3. Pembentukan label biner (0 = {LABEL_NORMAL}, 1 = selain {LABEL_NORMAL}) ---")
    label_teks = df[KOLOM_LABEL].astype(str)
    catat(f"Nilai label yang ada: {sorted(label_teks.unique())}")

    if LABEL_NORMAL not in set(label_teks.unique()):
        catat(f"PROSES DIHENTIKAN: nilai '{LABEL_NORMAL}' tidak ditemukan pada kolom {KOLOM_LABEL}.")
        simpan(nama_skenario, baris_keluaran)
        return None

    y = (label_teks != LABEL_NORMAL).astype(int)
    catat(f"Kelas 0 (normal)   : {angka((y == 0).sum())}")
    catat(f"Kelas 1 (serangan) : {angka((y == 1).sum())}  ({persen((y == 1).mean())} dari seluruh data)")

    # Seluruh fitur pada berkas ini sudah numerik. Pemeriksaan berikut dicatat
    # sebagai bukti bahwa tidak ada kolom kategorikal yang perlu dikodekan.
    kolom_non_numerik = [k for k in kolom_fitur if not pd.api.types.is_numeric_dtype(df[k])]
    catat(f"Jumlah fitur       : {len(kolom_fitur)}")
    if kolom_non_numerik:
        catat(f"Fitur non-numerik  : {kolom_non_numerik} (perlu dikodekan)")
    else:
        catat("Fitur non-numerik  : tidak ada, pengodean kategorikal tidak diperlukan.")

    X = df[kolom_fitur]

    # ---------- 4. Pembagian data latih dan data uji ----------
    catat()
    catat("--- 4. Pembagian data latih dan data uji ---")
    X_latih, X_uji, y_latih, y_uji = train_test_split(
        X, y,
        test_size=PROPORSI_DATA_UJI,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    catat(f"Metode         : pembagian berstrata, random_state={RANDOM_STATE}")
    catat(f"Data latih     : {angka(X_latih.shape[0])} baris "
          f"(normal {angka((y_latih == 0).sum())}, serangan {angka((y_latih == 1).sum())})")
    catat(f"Data uji       : {angka(X_uji.shape[0])} baris "
          f"(normal {angka((y_uji == 0).sum())}, serangan {angka((y_uji == 1).sum())})")

    # ---------- 5. Pelatihan Random Forest ----------
    catat()
    catat("--- 5. Pelatihan Random Forest ---")
    catat(f"Konfigurasi    : n_estimators={JUMLAH_POHON}, random_state={RANDOM_STATE}, "
          f"tanpa penanganan class imbalance")

    model = RandomForestClassifier(
        n_estimators=JUMLAH_POHON,
        random_state=RANDOM_STATE,
        n_jobs=JUMLAH_INTI,
    )

    waktu = time.perf_counter()
    model.fit(X_latih, y_latih)
    waktu_latih = time.perf_counter() - waktu
    catat(f"Waktu pelatihan: {waktu_latih:.2f} detik")

    waktu = time.perf_counter()
    prediksi = model.predict(X_uji)
    waktu_prediksi = time.perf_counter() - waktu
    catat(f"Waktu prediksi : {waktu_prediksi:.2f} detik")

    # ---------- 6. Evaluasi ----------
    akurasi = accuracy_score(y_uji, prediksi)
    presisi = precision_score(y_uji, prediksi, zero_division=0)
    recall = recall_score(y_uji, prediksi, zero_division=0)
    f1 = f1_score(y_uji, prediksi, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_uji, prediksi).ravel()

    catat()
    catat("--- 6. Hasil evaluasi pada data uji ---")
    catat(f"Akurasi   : {akurasi:.4f}  ({persen(akurasi)})")
    catat(f"Precision : {presisi:.4f}  ({persen(presisi)})")
    catat(f"Recall    : {recall:.4f}  ({persen(recall)})")
    catat(f"F1-score  : {f1:.4f}  ({persen(f1)})")
    catat()
    catat("Confusion matrix:")
    catat(f"  True Negative  (normal diprediksi normal)     : {angka(tn)}")
    catat(f"  False Positive (normal diprediksi serangan)   : {angka(fp)}")
    catat(f"  False Negative (serangan diprediksi normal)   : {angka(fn)}")
    catat(f"  True Positive  (serangan diprediksi serangan) : {angka(tp)}")
    catat()
    catat(f"  Dari {angka(tn + fp)} trafik normal pada data uji, {angka(fp)} salah diklasifikasikan sebagai serangan.")
    catat(f"  Dari {angka(fn + tp)} serangan pada data uji, {angka(fn)} gagal dikenali.")

    # ---------- 7. Diagnostik 1: Decision Tree dangkal ----------
    catat()
    catat("--- 7. Pemeriksaan kebocoran data 1: Decision Tree dangkal ---")
    catat("Pohon dilatih pada data latih, akurasinya diukur pada data uji.")
    catat("Akurasi yang sudah sangat tinggi pada kedalaman 1 sampai 2 menandakan")
    catat("kemungkinan adanya satu fitur yang memisahkan kelas secara berlebihan.")
    for kedalaman in [1, 2, 3]:
        pohon = DecisionTreeClassifier(max_depth=kedalaman, random_state=RANDOM_STATE)
        pohon.fit(X_latih, y_latih)
        akurasi_pohon = accuracy_score(y_uji, pohon.predict(X_uji))
        catat(f"  Kedalaman {kedalaman}: akurasi = {akurasi_pohon:.4f}  ({persen(akurasi_pohon)})")

    # ---------- 8. Diagnostik 2: feature importance ----------
    catat()
    catat("--- 8. Pemeriksaan kebocoran data 2: sepuluh fitur terpenting ---")
    importance = pd.Series(model.feature_importances_, index=kolom_fitur)
    importance = importance.sort_values(ascending=False)
    for urutan, (nama_fitur, nilai) in enumerate(importance.head(10).items(), start=1):
        catat(f"  {urutan:>2}. {nama_fitur:<30} {nilai:.6f}")

    # ---------- 9. Diagnostik 3: sebaran lima fitur teratas ----------
    catat()
    catat("--- 9. Pemeriksaan kebocoran data 3: sebaran lima fitur teratas per kelas ---")
    catat("Sebaran dihitung pada DATA LATIH saja, agar data uji tetap tidak tersentuh.")
    catat("Fitur yang rentang nilainya sama sekali tidak bertumpang tindih antara")
    catat("kelas normal dan kelas serangan perlu diperiksa lebih lanjut.")
    lima_teratas = importance.head(5).index.tolist()
    for nama_fitur in lima_teratas:
        catat()
        catat(f"  Fitur: {nama_fitur}")
        catat("  Kelas 0 (normal):")
        catat("    " + X_latih.loc[y_latih == 0, nama_fitur].describe().to_string().replace("\n", "\n    "))
        catat("  Kelas 1 (serangan):")
        catat("    " + X_latih.loc[y_latih == 1, nama_fitur].describe().to_string().replace("\n", "\n    "))

    # ---------- 10. Ringkasan ----------
    waktu_keseluruhan = time.perf_counter() - waktu_total
    catat()
    catat("=" * 70)
    catat(f"RINGKASAN SKENARIO {nama_skenario.upper()}")
    catat("=" * 70)
    catat(f"Akurasi        : {persen(akurasi)}")
    catat(f"Precision      : {persen(presisi)}")
    catat(f"Recall         : {persen(recall)}")
    catat(f"F1-score       : {persen(f1)}")
    catat(f"False Positive : {angka(fp)} dari {angka(tn + fp)} trafik normal")
    catat(f"False Negative : {angka(fn)} dari {angka(fn + tp)} serangan")
    catat(f"Waktu pelatihan: {waktu_latih:.2f} detik")
    catat(f"Waktu seluruh tahapan skenario ini: {waktu_keseluruhan:.2f} detik")
    catat()
    catat("Hasil di atas merupakan studi kelayakan awal. Angka ini belum melalui")
    catat("validasi silang, pemilihan hyperparameter, maupun penanganan class")
    catat("imbalance, sehingga belum dapat dianggap sebagai hasil akhir penelitian.")

    simpan(nama_skenario, baris_keluaran)

    return {
        "skenario": nama_skenario,
        "akurasi": akurasi,
        "precision": presisi,
        "recall": recall,
        "f1": f1,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "waktu_latih": waktu_latih,
    }


def simpan(nama_skenario, baris_keluaran):
    os.makedirs(FOLDER_OUTPUT, exist_ok=True)
    path_keluaran = os.path.join(FOLDER_OUTPUT, f"hasil_02_uji_coba_{nama_skenario}.txt")
    with open(path_keluaran, "w", encoding="utf-8") as berkas:
        berkas.write("\n".join(baris_keluaran))
    print(f"\nHasil skenario {nama_skenario} disimpan ke: {os.path.abspath(path_keluaran)}")


def main():
    waktu_mulai = time.perf_counter()
    hasil = []

    for nama_skenario, nama_berkas in BERKAS.items():
        ringkasan = jalankan_skenario(nama_skenario, nama_berkas)
        if ringkasan is not None:
            hasil.append(ringkasan)
        print("\n")

    if hasil:
        print("=" * 70)
        print("PERBANDINGAN ANTARSKENARIO")
        print("=" * 70)
        tabel = pd.DataFrame(hasil)
        print(tabel.to_string(index=False))

    print(f"\nTotal waktu eksekusi seluruh skrip: {time.perf_counter() - waktu_mulai:.2f} detik")


if __name__ == "__main__":
    main()
