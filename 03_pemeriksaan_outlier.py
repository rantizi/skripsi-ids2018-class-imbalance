"""
Pemeriksaan nilai pencilan pada seluruh fitur kedua berkas data dengan metode IQR.
Tidak ada baris atau nilai yang dihapus, keluarannya hanya laporan.

Jalankan : python 03_pemeriksaan_outlier.py [skenario]
Keluaran : hasil/outlier/
"""

import sys
import time

import numpy as np
import pandas as pd

import konfigurasi as kf

# Jumlah fitur dengan persentase pencilan tertinggi yang ditampilkan di ringkasan.
JUMLAH_DITAMPILKAN = 15


def periksa_satu_fitur(nilai):
    """Menghitung statistik IQR untuk satu kolom fitur."""
    # Ketiga kuartil dihitung sekaligus agar kolom tidak perlu diurutkan tiga kali.
    kuartil = nilai.quantile([0.25, 0.5, 0.75])
    q1 = float(kuartil.loc[0.25])
    median = float(kuartil.loc[0.50])
    q3 = float(kuartil.loc[0.75])
    iqr = q3 - q1
    batas_bawah = q1 - 1.5 * iqr
    batas_atas = q3 + 1.5 * iqr

    jumlah_bawah = int((nilai < batas_bawah).sum())
    jumlah_atas = int((nilai > batas_atas).sum())
    jumlah_negatif = int((nilai < 0).sum())

    return {
        "minimum": float(nilai.min()),
        "q1": q1,
        "median": median,
        "q3": q3,
        "maksimum": float(nilai.max()),
        "iqr": iqr,
        "batas_bawah": batas_bawah,
        "batas_atas": batas_atas,
        "jumlah_bawah": jumlah_bawah,
        "jumlah_atas": jumlah_atas,
        "jumlah_outlier": jumlah_bawah + jumlah_atas,
        "jumlah_negatif": jumlah_negatif,
        "iqr_nol": iqr == 0,
    }


def periksa_skenario(skenario, catat):
    catat.garis(f"SKENARIO {skenario.upper()}")

    waktu = time.perf_counter()
    df = kf.muat_data(skenario)
    fitur = kf.kolom_fitur(df)
    jumlah_baris = len(df)

    catat(f"Berkas       : {kf.BERKAS[skenario]}")
    catat(f"Jumlah baris : {kf.angka(jumlah_baris)}")
    catat(f"Jumlah fitur : {len(fitur)}")
    catat(f"Waktu muat   : {time.perf_counter() - waktu:.2f} detik")

    waktu = time.perf_counter()
    baris_hasil = []
    for nama in fitur:
        nilai = df[nama]
        if not pd.api.types.is_numeric_dtype(nilai):
            continue
        hasil = {"fitur": nama}
        hasil.update(periksa_satu_fitur(nilai))
        hasil["persen_outlier"] = hasil["jumlah_outlier"] / jumlah_baris * 100
        baris_hasil.append(hasil)

    tabel = pd.DataFrame(baris_hasil)
    catat(f"Waktu hitung : {time.perf_counter() - waktu:.2f} detik")

    # Simpan rincian seluruh fitur ke CSV.
    path_csv = f"{kf.folder('outlier')}/outlier_{skenario}.csv"
    tabel.to_csv(path_csv, index=False)
    catat(f"Rincian seluruh fitur disimpan ke: {path_csv}")

    # ---------- Ringkasan ----------
    iqr_nol = tabel[tabel["iqr_nol"]]
    iqr_wajar = tabel[~tabel["iqr_nol"]]

    catat("")
    catat("--- Ringkasan ---")
    catat(f"Fitur numerik yang diperiksa      : {len(tabel)}")
    catat(f"Fitur dengan IQR bernilai nol     : {len(iqr_nol)}")
    catat(f"Fitur dengan IQR lebih dari nol   : {len(iqr_wajar)}")
    catat(f"Fitur tanpa nilai pencilan        : {int((tabel['jumlah_outlier'] == 0).sum())}")
    catat(f"Fitur yang memuat nilai negatif   : {int((tabel['jumlah_negatif'] > 0).sum())}")

    catat("")
    catat("Catatan mengenai fitur dengan IQR bernilai nol:")
    catat("Pada fitur tersebut, kuartil 1 dan kuartil 3 bernilai sama, sehingga batas")
    catat("bawah dan batas atas berimpit. Akibatnya setiap nilai yang berbeda dari")
    catat("nilai tengah langsung terhitung sebagai pencilan. Persentase pencilan pada")
    catat("kelompok fitur ini tidak dapat ditafsirkan dengan cara yang sama seperti")
    catat("fitur lain, sehingga dilaporkan secara terpisah.")

    # Fitur dengan IQR nol
    if len(iqr_nol) > 0:
        catat("")
        catat(f"--- Fitur dengan IQR bernilai nol ({len(iqr_nol)} fitur) ---")
        urut = iqr_nol.sort_values("persen_outlier", ascending=False)
        for _, b in urut.head(JUMLAH_DITAMPILKAN).iterrows():
            catat(f"  {b['fitur']:<28} nilai tengah = {b['median']:<14.4g} "
                  f"pencilan = {kf.angka(b['jumlah_outlier']):>10} "
                  f"({kf.persen(b['persen_outlier'] / 100)})")
        if len(urut) > JUMLAH_DITAMPILKAN:
            catat(f"  ... dan {len(urut) - JUMLAH_DITAMPILKAN} fitur lain, "
                  f"selengkapnya pada berkas CSV.")

    # Fitur dengan IQR wajar
    catat("")
    catat(f"--- {JUMLAH_DITAMPILKAN} fitur dengan persentase pencilan tertinggi "
          f"(IQR lebih dari nol) ---")
    urut = iqr_wajar.sort_values("persen_outlier", ascending=False)
    for _, b in urut.head(JUMLAH_DITAMPILKAN).iterrows():
        catat(f"  {b['fitur']:<28} "
              f"bawah = {kf.angka(b['jumlah_bawah']):>9}  "
              f"atas = {kf.angka(b['jumlah_atas']):>10}  "
              f"total = {kf.persen(b['persen_outlier'] / 100)}")

    # Nilai negatif
    negatif = tabel[tabel["jumlah_negatif"] > 0].sort_values("jumlah_negatif", ascending=False)
    catat("")
    catat("--- Fitur yang memuat nilai negatif ---")
    if len(negatif) == 0:
        catat("  Tidak ada fitur yang memuat nilai negatif.")
    else:
        catat("  Nilai negatif perlu ditinjau satu per satu. Pada sebagian fitur,")
        catat("  nilai negatif memang tidak mungkin terjadi dan menandakan kesalahan")
        catat("  pengukuran. Pada fitur lain, nilai negatif dipakai sebagai penanda")
        catat("  bahwa nilainya tidak teramati, sehingga bukan kesalahan.")
        catat("")
        for _, b in negatif.head(25).iterrows():
            catat(f"  {b['fitur']:<28} jumlah = {kf.angka(b['jumlah_negatif']):>10} "
                  f"({kf.persen(b['jumlah_negatif'] / jumlah_baris)})  "
                  f"nilai minimum = {b['minimum']:.4g}")
        if len(negatif) > 25:
            catat(f"  ... dan {len(negatif) - 25} fitur lain, selengkapnya pada berkas CSV.")

    del df
    return tabel


def main():
    pilihan = sys.argv[1:]
    daftar = pilihan if pilihan else list(kf.BERKAS)

    for skenario in daftar:
        if skenario not in kf.BERKAS:
            print(f"Skenario '{skenario}' tidak dikenal. Pilihan: {list(kf.BERKAS)}")
            return

    catat = kf.Pencatat()
    waktu_total = time.perf_counter()

    catat("PEMERIKSAAN NILAI PENCILAN DENGAN METODE INTERQUARTILE RANGE (IQR)")
    catat("")
    catat("Batas bawah = Q1 - 1,5 x IQR")
    catat("Batas atas  = Q3 + 1,5 x IQR")
    catat("Tidak ada baris maupun nilai yang dihapus oleh skrip ini.")
    catat("")
    for baris in kf.info_lingkungan():
        catat(baris)

    for skenario in daftar:
        periksa_skenario(skenario, catat)

    catat.garis("SELESAI")
    catat(f"Total waktu eksekusi: {time.perf_counter() - waktu_total:.2f} detik")
    catat.simpan(f"{kf.folder('outlier')}/hasil_03_pemeriksaan_outlier.txt")


if __name__ == "__main__":
    main()
