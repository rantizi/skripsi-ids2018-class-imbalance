"""
Pemeriksaan kondisi berkas dataset untuk penelitian:
"Analisis Pengaruh Penanganan Class Imbalance terhadap Deteksi Serangan DoS dan
Infiltration pada Trafik Jaringan CSE-CIC-IDS2018"

Catatan: nilai yang ditemukan hanya dihitung dan dilaporkan, tidak dibuang.
Tujuan pemeriksaan ini adalah mengonfirmasi kondisi data sebelum digunakan,
bukan membersihkannya kembali.

Keluaran tercetak di layar dan tersimpan di: hasil/hasil_01_cek_dataset.txt
"""

import os
import platform
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import sklearn

# ===================== KONFIGURASI =====================
# Folder tempat berkas parquet disimpan.
FOLDER_DATA = r"D:\Codes\Skripsi\dashboard_jaringan"

# Folder tempat berkas hasil disimpan (dibuat otomatis kalau belum ada).
FOLDER_OUTPUT = "hasil"

# Dua berkas yang dipakai pada penelitian, satu berkas per skenario.
BERKAS = {
    "DoS": "DoS1-Thursday-15-02-2018_TrafficForML_CICFlowMeter.parquet",
    "Infiltration": "Infil1-Wednesday-28-02-2018_TrafficForML_CICFlowMeter.parquet",
}

KOLOM_LABEL = "Label"
LABEL_NORMAL = "Benign"
# =======================================================


# Semua keluaran dikumpulkan di sini supaya bisa ditulis ke berkas di akhir.
baris_keluaran = []


def catat(teks=""):
    """Mencetak teks ke layar sekaligus menyimpannya untuk ditulis ke berkas."""
    teks = str(teks)
    print(teks)
    baris_keluaran.append(teks)


def angka(n):
    """Memformat bilangan dengan pemisah ribuan gaya Indonesia. 743498 -> 743.498"""
    return f"{int(n):,}".replace(",", ".")


def persen(x):
    """Memformat persentase gaya Indonesia. 93.5478 -> 93,55%"""
    return f"{x:.2f}".replace(".", ",") + "%"


def garis(judul=""):
    catat()
    catat("=" * 70)
    if judul:
        catat(judul)
        catat("=" * 70)


def tulis_informasi_lingkungan():
    garis("INFORMASI LINGKUNGAN PENGERJAAN")
    catat(f"Waktu mulai      : {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")
    catat(f"Sistem operasi   : {platform.platform()}")
    catat(f"Prosesor         : {platform.processor()}")
    catat(f"Python           : {sys.version.split()[0]}")
    catat(f"pandas           : {pd.__version__}")
    catat(f"numpy            : {np.__version__}")
    catat(f"scikit-learn     : {sklearn.__version__}")
    try:
        import pyarrow
        catat(f"pyarrow          : {pyarrow.__version__}")
    except ImportError:
        catat("pyarrow          : tidak terpasang")
    catat(f"Folder data      : {FOLDER_DATA}")


def ringkasan_semua_berkas():
    """Membaca kolom Label saja dari setiap berkas parquet, untuk gambaran umum."""
    garis("BAGIAN 1 - RINGKASAN SELURUH BERKAS PARQUET DI FOLDER DATA")
    catat("Hanya kolom Label yang dibaca, sehingga proses ini ringan.")
    catat("Bagian ini menjadi dasar pemilihan dua berkas yang dipakai pada penelitian.")
    catat()

    daftar = []
    for nama in sorted(os.listdir(FOLDER_DATA)):
        if not nama.endswith(".parquet"):
            continue

        path = os.path.join(FOLDER_DATA, nama)
        label = pd.read_parquet(path, columns=[KOLOM_LABEL])[KOLOM_LABEL].astype(str)

        total = len(label)
        normal = int((label == LABEL_NORMAL).sum())
        serangan = total - normal

        daftar.append({
            "berkas": nama.split("_")[0],
            "ukuran_MB": round(os.path.getsize(path) / 1024 / 1024, 1),
            "total_baris": total,
            "normal": normal,
            "serangan": serangan,
            "persen_serangan": round(serangan / total * 100, 2),
            "label_unik": ", ".join(sorted(label.unique())),
        })

    ringkasan = pd.DataFrame(daftar)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 80)
    catat(ringkasan.to_string(index=False))


def periksa_satu_berkas(nama_skenario, nama_berkas):
    """Pemeriksaan detail satu berkas: distribusi kelas dan kualitas data."""
    path = os.path.join(FOLDER_DATA, nama_berkas)

    garis(f"BAGIAN 2 - PEMERIKSAAN DETAIL SKENARIO {nama_skenario.upper()}")
    catat(f"Berkas      : {nama_berkas}")
    catat(f"Ukuran      : {round(os.path.getsize(path) / 1024 / 1024, 1)} MB")

    waktu_mulai = time.perf_counter()
    df = pd.read_parquet(path)
    catat(f"Waktu muat  : {time.perf_counter() - waktu_mulai:.2f} detik")

    jumlah_baris, jumlah_kolom = df.shape
    catat(f"Jumlah baris: {angka(jumlah_baris)}")
    catat(f"Jumlah kolom: {jumlah_kolom} (terdiri atas {jumlah_kolom - 1} kolom fitur + 1 kolom {KOLOM_LABEL})")

    # Jumlah kolom bertipe teks/kategori selain kolom label.
    kolom_fitur = [k for k in df.columns if k != KOLOM_LABEL]
    kolom_non_numerik = [k for k in kolom_fitur if not pd.api.types.is_numeric_dtype(df[k])]
    if kolom_non_numerik:
        catat(f"Kolom fitur non-numerik: {len(kolom_non_numerik)} -> {kolom_non_numerik}")
    else:
        catat("Kolom fitur non-numerik: 0 (seluruh fitur sudah numerik, encoding tidak diperlukan)")

    # ---------- Distribusi kelas asli (pendukung Tabel 3.2) ----------
    catat()
    catat("--- Jumlah dan persentase data per kelas ---")
    label_teks = df[KOLOM_LABEL].astype(str)
    jumlah_per_kelas = label_teks.value_counts()
    for nama_kelas, jumlah in jumlah_per_kelas.items():
        catat(f"  {nama_kelas:<28} : {angka(jumlah):>12}  ({persen(jumlah / jumlah_baris * 100)})")
    catat(f"  {'TOTAL':<28} : {angka(jumlah_baris):>12}  (100,00%)")

    # ---------- Distribusi label biner ----------
    catat()
    catat(f"--- Distribusi label biner (0 = {LABEL_NORMAL}, 1 = selain {LABEL_NORMAL}) ---")
    if LABEL_NORMAL not in set(label_teks.unique()):
        catat(f"  PERINGATAN: nilai '{LABEL_NORMAL}' tidak ditemukan pada kolom {KOLOM_LABEL}.")
        catat(f"  Nilai yang tersedia: {sorted(label_teks.unique())}")
        return

    label_biner = (label_teks != LABEL_NORMAL).astype(int)
    jumlah_normal = int((label_biner == 0).sum())
    jumlah_serangan = int((label_biner == 1).sum())
    catat(f"  0 (normal)   : {angka(jumlah_normal):>12}  ({persen(jumlah_normal / jumlah_baris * 100)})")
    catat(f"  1 (serangan) : {angka(jumlah_serangan):>12}  ({persen(jumlah_serangan / jumlah_baris * 100)})")

    # ---------- Pemeriksaan kualitas data (pendukung Tabel 3.3) ----------
    catat()
    catat("--- Pemeriksaan kualitas data ---")

    waktu_mulai = time.perf_counter()
    jumlah_duplikat = int(df.duplicated().sum())
    jumlah_kosong = int(df.isna().sum().sum())

    # Nilai infinity diperiksa per kolom numerik agar hemat memori.
    jumlah_infinity = 0
    for kolom in kolom_fitur:
        if pd.api.types.is_float_dtype(df[kolom]):
            jumlah_infinity += int(np.isinf(df[kolom].to_numpy()).sum())

    catat(f"  Duplikasi baris : {angka(jumlah_duplikat)}")
    catat(f"  Missing value   : {angka(jumlah_kosong)}")
    catat(f"  Nilai infinity  : {angka(jumlah_infinity)}")
    catat(f"  Waktu periksa   : {time.perf_counter() - waktu_mulai:.2f} detik")

    if jumlah_duplikat == 0 and jumlah_kosong == 0 and jumlah_infinity == 0:
        catat("  Kesimpulan      : data bersih, tidak ada baris yang perlu dibuang.")
    else:
        catat("  Kesimpulan      : DITEMUKAN nilai bermasalah. Tidak ada baris yang dibuang")
        catat("                    oleh skrip ini. Temuan ini perlu ditinjau terlebih dahulu")
        catat("                    sebelum data digunakan untuk pelatihan model.")

    del df


def main():
    os.makedirs(FOLDER_OUTPUT, exist_ok=True)
    waktu_total = time.perf_counter()

    catat("PEMERIKSAAN KONDISI DATASET - STUDI KELAYAKAN AWAL")
    catat("Penelitian: Analisis Pengaruh Penanganan Class Imbalance terhadap Deteksi")
    catat("            Serangan DoS dan Infiltration pada Trafik Jaringan CSE-CIC-IDS2018")

    tulis_informasi_lingkungan()
    ringkasan_semua_berkas()

    for nama_skenario, nama_berkas in BERKAS.items():
        periksa_satu_berkas(nama_skenario, nama_berkas)

    garis("SELESAI")
    catat(f"Total waktu eksekusi: {time.perf_counter() - waktu_total:.2f} detik")

    path_keluaran = os.path.join(FOLDER_OUTPUT, "hasil_01_cek_dataset.txt")
    with open(path_keluaran, "w", encoding="utf-8") as berkas:
        berkas.write("\n".join(baris_keluaran))
    print(f"\nHasil disimpan ke: {os.path.abspath(path_keluaran)}")


if __name__ == "__main__":
    main()
