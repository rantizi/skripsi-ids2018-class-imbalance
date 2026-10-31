"""
Pemilihan hyperparameter lewat validasi silang berstrata pada data latih, untuk enam
pasangan skenario dan model. Dijalankan pada kondisi netral, dan hasil yang sudah ada
akan dilewati.

Jalankan : python 04_validasi_silang.py <lr|dt|rf> [skenario] [--ulang]
Keluaran : hasil/cv/
"""

import json
import os
import sys
import time
import warnings

from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import konfigurasi as kf


def buat_model(kode_model, class_weight=None):
    """
    Membuat objek model sesuai kode yang diberikan. Logistic Regression dibungkus
    pipeline bersama StandardScaler agar penskalaan dihitung ulang per lipatan.
    """
    if kode_model == "lr":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=kf.MAX_ITER_LR,
                class_weight=class_weight,
                random_state=kf.RANDOM_STATE,
            )),
        ])

    if kode_model == "dt":
        return DecisionTreeClassifier(
            class_weight=class_weight,
            random_state=kf.RANDOM_STATE,
        )

    if kode_model == "rf":
        return RandomForestClassifier(
            class_weight=class_weight,
            random_state=kf.RANDOM_STATE,
            n_jobs=-1,
        )

    raise ValueError(f"Kode model '{kode_model}' tidak dikenal. Pilihan: {list(kf.NAMA_MODEL)}")


def jalankan(skenario, kode_model, ulang=False):
    path_json = f"{kf.folder('cv')}/cv_{skenario}_{kode_model}.json"

    if os.path.exists(path_json) and not ulang:
        print(f"[lewati] Hasil untuk {skenario} + {kode_model} sudah ada di {path_json}.")
        print("         Tambahkan --ulang bila ingin menjalankannya kembali.\n")
        return

    catat = kf.Pencatat()
    waktu_total = time.perf_counter()

    catat("=" * 70)
    catat(f"VALIDASI SILANG - SKENARIO {skenario.upper()} - {kf.NAMA_MODEL[kode_model].upper()}")
    catat("=" * 70)
    catat("")
    for baris in kf.info_lingkungan():
        catat(baris)

    # ---------- Menyiapkan data ----------
    catat("")
    catat("--- Menyiapkan data ---")
    waktu = time.perf_counter()
    X_latih, X_uji, y_latih, y_uji, fitur = kf.siapkan_data(skenario)
    catat(f"Waktu siapkan  : {time.perf_counter() - waktu:.2f} detik")
    catat(f"Data latih     : {kf.angka(len(X_latih))} baris, {len(fitur)} fitur")
    catat(f"  kelas normal   : {kf.angka((y_latih == 0).sum())}")
    catat(f"  kelas serangan : {kf.angka((y_latih == 1).sum())} "
          f"({kf.persen((y_latih == 1).mean())})")
    catat(f"Data uji       : {kf.angka(len(X_uji))} baris (tidak dipakai pada tahap ini)")

    # Data uji tidak diperlukan di sini, dilepas agar memori lebih longgar.
    del X_uji, y_uji

    # ---------- Pengaturan validasi silang ----------
    kandidat = kf.KANDIDAT[kode_model]
    jumlah_kandidat = 1
    for nilai in kandidat.values():
        jumlah_kandidat *= len(nilai)

    catat("")
    catat("--- Pengaturan validasi silang ---")
    catat(f"Model          : {kf.NAMA_MODEL[kode_model]}")
    catat(f"Kondisi        : netral, tanpa class_weight")
    catat(f"Pembagian      : StratifiedKFold({kf.JUMLAH_LIPATAN}, shuffle=True, "
          f"random_state={kf.RANDOM_STATE})")
    catat(f"Kriteria       : {kf.SCORING} pada kelas positif")
    catat(f"Kandidat       : {kandidat}")
    catat(f"Jumlah pelatihan yang dijalankan: {jumlah_kandidat} kandidat x "
          f"{kf.JUMLAH_LIPATAN} lipatan = {jumlah_kandidat * kf.JUMLAH_LIPATAN} kali")
    catat("")
    catat("Proses ini memakan waktu. Kemajuannya tercetak di layar selama berjalan.")

    lipatan = StratifiedKFold(
        n_splits=kf.JUMLAH_LIPATAN,
        shuffle=True,
        random_state=kf.RANDOM_STATE,
    )

    pencarian = GridSearchCV(
        estimator=buat_model(kode_model, class_weight=None),
        param_grid=kandidat,
        scoring=kf.SCORING,
        cv=lipatan,
        n_jobs=1,          # satu pelatihan pada satu waktu, agar hemat memori
        refit=False,       # model akhir dilatih ulang pada skrip berikutnya
        return_train_score=False,
        error_score="raise",
        verbose=2,
    )

    # ---------- Menjalankan pencarian ----------
    print()
    waktu = time.perf_counter()
    with warnings.catch_warnings(record=True) as tercatat:
        warnings.simplefilter("always", ConvergenceWarning)
        pencarian.fit(X_latih, y_latih)
    waktu_cari = time.perf_counter() - waktu

    jumlah_peringatan = sum(1 for w in tercatat if issubclass(w.category, ConvergenceWarning))

    # ---------- Hasil ----------
    hasil = pencarian.cv_results_
    daftar_kandidat = []
    for i in range(len(hasil["params"])):
        daftar_kandidat.append({
            "parameter": {k: (None if v is None else v) for k, v in hasil["params"][i].items()},
            "skor_rata_rata": float(hasil["mean_test_score"][i]),
            "skor_simpangan_baku": float(hasil["std_test_score"][i]),
            "waktu_latih_rata_rata_detik": float(hasil["mean_fit_time"][i]),
            "peringkat": int(hasil["rank_test_score"][i]),
        })
    daftar_kandidat.sort(key=lambda b: b["peringkat"])

    terpilih = hasil["params"][int(pencarian.best_index_)]
    skor_terbaik = float(pencarian.best_score_)

    catat("")
    catat("--- Hasil validasi silang ---")
    catat(f"Waktu pencarian: {waktu_cari:.2f} detik ({waktu_cari / 60:.1f} menit)")
    catat("")
    catat("Skor F1 rata-rata untuk setiap kandidat, diurutkan dari yang terbaik:")
    for b in daftar_kandidat:
        nilai = ", ".join(f"{k} = {v}" for k, v in b["parameter"].items())
        catat(f"  {b['peringkat']}. {nilai:<28} "
              f"F1 = {b['skor_rata_rata']:.4f} "
              f"(simpangan baku {b['skor_simpangan_baku']:.4f}, "
              f"rata-rata {b['waktu_latih_rata_rata_detik']:.1f} detik per pelatihan)")

    catat("")
    catat(f"Hyperparameter terpilih : {terpilih}")
    catat(f"Skor F1 terbaik         : {skor_terbaik:.4f}")

    if jumlah_peringatan > 0:
        catat("")
        catat(f"Peringatan non-konvergensi: {jumlah_peringatan} kali.")
        catat(f"Nilai max_iter saat ini adalah {kf.MAX_ITER_LR}. Bila peringatan masih")
        catat("muncul, nilai tersebut perlu dinaikkan dan prosedur dijalankan ulang.")
    elif kode_model == "lr":
        catat("")
        catat(f"Tidak ada peringatan non-konvergensi dengan max_iter = {kf.MAX_ITER_LR}.")

    # ---------- Simpan ----------
    ringkasan = {
        "skenario": skenario,
        "kode_model": kode_model,
        "nama_model": kf.NAMA_MODEL[kode_model],
        "kondisi_validasi": "netral, tanpa class_weight",
        "kriteria_pemilihan": kf.SCORING,
        "jumlah_lipatan": kf.JUMLAH_LIPATAN,
        "random_state": kf.RANDOM_STATE,
        "jumlah_data_latih": int(len(X_latih)),
        "jumlah_fitur": len(fitur),
        "kandidat_diuji": {k: list(v) for k, v in kandidat.items()},
        "hasil_setiap_kandidat": daftar_kandidat,
        "hyperparameter_terpilih": terpilih,
        "skor_f1_terbaik": skor_terbaik,
        "waktu_pencarian_detik": round(waktu_cari, 2),
        "jumlah_peringatan_konvergensi": jumlah_peringatan,
        "max_iter_logistic_regression": kf.MAX_ITER_LR if kode_model == "lr" else None,
        "lingkungan": kf.info_lingkungan(),
    }

    with open(path_json, "w", encoding="utf-8") as berkas:
        json.dump(ringkasan, berkas, indent=2, ensure_ascii=False)

    catat("")
    catat(f"Hyperparameter terpilih disimpan ke: {path_json}")
    catat("")
    catat("=" * 70)
    catat(f"Total waktu: {time.perf_counter() - waktu_total:.2f} detik")

    catat.simpan(f"{kf.folder('cv')}/hasil_04_validasi_silang_{skenario}_{kode_model}.txt")
    print()


def main():
    argumen = [a for a in sys.argv[1:] if a != "--ulang"]
    ulang = "--ulang" in sys.argv

    if not argumen:
        print(__doc__)
        print(f"Kode model yang tersedia : {list(kf.NAMA_MODEL)}")
        print(f"Skenario yang tersedia   : {list(kf.BERKAS)}")
        return

    kode_model = argumen[0]
    if kode_model not in kf.NAMA_MODEL:
        print(f"Kode model '{kode_model}' tidak dikenal. Pilihan: {list(kf.NAMA_MODEL)}")
        return

    daftar_skenario = argumen[1:] if len(argumen) > 1 else list(kf.BERKAS)
    for skenario in daftar_skenario:
        if skenario not in kf.BERKAS:
            print(f"Skenario '{skenario}' tidak dikenal. Pilihan: {list(kf.BERKAS)}")
            return

    for skenario in daftar_skenario:
        jalankan(skenario, kode_model, ulang=ulang)


if __name__ == "__main__":
    main()
