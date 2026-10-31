"""
Pelatihan dan evaluasi 12 kombinasi eksperimen memakai hyperparameter hasil tahap
sebelumnya. Menyimpan metrik, probabilitas prediksi, dan feature importance.

Jalankan : python 05_latih_dan_evaluasi.py [model] [skenario] [--ulang]
Keluaran : hasil/model/
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)

import konfigurasi as kf
from importlib import import_module

# Fungsi buat_model dipakai ulang dari skrip validasi silang agar definisi
# model hanya ditulis di satu tempat.
buat_model = import_module("04_validasi_silang").buat_model


def ambil_hyperparameter(skenario, kode_model):
    """Membaca hyperparameter terpilih dari hasil validasi silang."""
    path = os.path.join(kf.FOLDER_HASIL, "cv", f"cv_{skenario}_{kode_model}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Hasil validasi silang belum ada: {path}\n"
            f"Jalankan terlebih dahulu: python 04_validasi_silang.py {kode_model} {skenario}"
        )
    with open(path, encoding="utf-8") as berkas:
        isi = json.load(berkas)
    return isi["hyperparameter_terpilih"], isi.get("skor_f1_terbaik")


def jalankan_kombinasi(skenario, kode_model, nama_kondisi, data, parameter, skor_cv):
    X_latih, X_uji, y_latih, y_uji, fitur = data
    class_weight = kf.KONDISI[nama_kondisi]

    catat = kf.Pencatat()
    waktu_total = time.perf_counter()

    catat("=" * 70)
    catat(f"SKENARIO {skenario.upper()} - {kf.NAMA_MODEL[kode_model].upper()} - "
          f"KONDISI {nama_kondisi.upper()}")
    catat("=" * 70)
    catat("")
    for baris in kf.info_lingkungan():
        catat(baris)

    catat("")
    catat("--- Konfigurasi ---")
    catat(f"Model              : {kf.NAMA_MODEL[kode_model]}")
    catat(f"Penanganan imbalance: "
          f"{'tanpa penanganan' if class_weight is None else f'class_weight = {class_weight}'}")
    catat(f"Hyperparameter     : {parameter}")
    catat(f"  diambil dari hasil validasi silang, F1 = {skor_cv:.4f}")
    catat(f"Data latih         : {kf.angka(len(X_latih))} baris")
    catat(f"Data uji           : {kf.angka(len(X_uji))} baris "
          f"(normal {kf.angka((y_uji == 0).sum())}, "
          f"serangan {kf.angka((y_uji == 1).sum())})")

    # ---------- Pelatihan ----------
    model = buat_model(kode_model, class_weight=class_weight)
    model.set_params(**parameter)

    catat("")
    catat("--- Pelatihan ---")
    waktu = time.perf_counter()
    model.fit(X_latih, y_latih)
    waktu_latih = time.perf_counter() - waktu
    catat(f"Waktu pelatihan    : {waktu_latih:.2f} detik")

    # ---------- Prediksi ----------
    waktu = time.perf_counter()
    prediksi = model.predict(X_uji)
    probabilitas = model.predict_proba(X_uji)[:, 1].astype(np.float32)
    waktu_prediksi = time.perf_counter() - waktu
    catat(f"Waktu prediksi     : {waktu_prediksi:.2f} detik")

    # ---------- Metrik ----------
    presisi = precision_score(y_uji, prediksi, zero_division=0)
    recall = recall_score(y_uji, prediksi, zero_division=0)
    f1 = f1_score(y_uji, prediksi, zero_division=0)
    pr_auc = average_precision_score(y_uji, probabilitas)
    akurasi = accuracy_score(y_uji, prediksi)
    tn, fp, fn, tp = confusion_matrix(y_uji, prediksi, labels=[0, 1]).ravel()

    # Nilai pembanding PR-AUC: model yang menebak secara acak akan memperoleh
    # PR-AUC kira-kira sebesar proporsi kelas serangan pada data uji.
    dasar_pr_auc = float((y_uji == 1).mean())

    catat("")
    catat("--- Metrik pada data uji ---")
    catat(f"Precision          : {presisi:.4f}  ({kf.persen(presisi)})")
    catat(f"Recall             : {recall:.4f}  ({kf.persen(recall)})")
    catat(f"F1-score           : {f1:.4f}  ({kf.persen(f1)})")
    catat(f"PR-AUC             : {pr_auc:.4f}")
    catat(f"  pembanding acak    : {dasar_pr_auc:.4f} "
          f"(proporsi kelas serangan pada data uji)")
    catat(f"Akurasi (pembanding): {akurasi:.4f}  ({kf.persen(akurasi)})")

    catat("")
    catat("Confusion matrix:")
    catat(f"  True Negative  (normal diprediksi normal)     : {kf.angka(tn)}")
    catat(f"  False Positive (normal diprediksi serangan)   : {kf.angka(fp)}")
    catat(f"  False Negative (serangan diprediksi normal)   : {kf.angka(fn)}")
    catat(f"  True Positive  (serangan diprediksi serangan) : {kf.angka(tp)}")

    # ---------- Menyimpan probabilitas ----------
    folder_model = kf.folder("model")
    path_proba = f"{folder_model}/proba_{skenario}_{kode_model}_{nama_kondisi}.npy"
    np.save(path_proba, probabilitas)

    path_label = f"{folder_model}/label_uji_{skenario}.npy"
    if not os.path.exists(path_label):
        np.save(path_label, y_uji.to_numpy().astype(np.int8))

    catat("")
    catat(f"Probabilitas prediksi disimpan ke: {path_proba}")

    # ---------- Feature importance (khusus Random Forest) ----------
    path_importance = None
    if kode_model == "rf":
        importance = pd.Series(model.feature_importances_, index=fitur)
        importance = importance.sort_values(ascending=False)
        path_importance = f"{folder_model}/importance_{skenario}_rf_{nama_kondisi}.csv"
        importance.rename("importance").rename_axis("fitur").to_csv(path_importance)

        catat("")
        catat("--- Sepuluh fitur dengan importance tertinggi ---")
        for urutan, (nama_fitur, nilai) in enumerate(importance.head(10).items(), start=1):
            catat(f"  {urutan:>2}. {nama_fitur:<30} {nilai:.6f}")
        catat("")
        catat(f"Peringkat seluruh fitur disimpan ke: {path_importance}")

    # ---------- Menyimpan metrik ----------
    ringkasan = {
        "skenario": skenario,
        "kode_model": kode_model,
        "nama_model": kf.NAMA_MODEL[kode_model],
        "kondisi": nama_kondisi,
        "class_weight": class_weight,
        "hyperparameter": parameter,
        "skor_f1_validasi_silang": skor_cv,
        "jumlah_data_latih": int(len(X_latih)),
        "jumlah_data_uji": int(len(X_uji)),
        "metrik": {
            "precision": float(presisi),
            "recall": float(recall),
            "f1": float(f1),
            "pr_auc": float(pr_auc),
            "pr_auc_pembanding_acak": dasar_pr_auc,
            "akurasi": float(akurasi),
        },
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "waktu_pelatihan_detik": round(waktu_latih, 2),
        "waktu_prediksi_detik": round(waktu_prediksi, 2),
        "berkas_probabilitas": os.path.basename(path_proba),
        "berkas_importance": os.path.basename(path_importance) if path_importance else None,
        "catatan_pr_auc": "dihitung dengan average_precision_score",
        "lingkungan": kf.info_lingkungan(),
    }

    path_json = f"{folder_model}/metrik_{skenario}_{kode_model}_{nama_kondisi}.json"
    with open(path_json, "w", encoding="utf-8") as berkas:
        json.dump(ringkasan, berkas, indent=2, ensure_ascii=False)

    catat("")
    catat(f"Metrik disimpan ke: {path_json}")
    catat("")
    catat("=" * 70)
    catat(f"Total waktu: {time.perf_counter() - waktu_total:.2f} detik")

    catat.simpan(f"{folder_model}/hasil_05_{skenario}_{kode_model}_{nama_kondisi}.txt")
    print()


def main():
    argumen = [a for a in sys.argv[1:] if a != "--ulang"]
    ulang = "--ulang" in sys.argv

    daftar_model = [argumen[0]] if argumen else list(kf.NAMA_MODEL)
    for kode in daftar_model:
        if kode not in kf.NAMA_MODEL:
            print(f"Kode model '{kode}' tidak dikenal. Pilihan: {list(kf.NAMA_MODEL)}")
            return

    daftar_skenario = argumen[1:] if len(argumen) > 1 else list(kf.BERKAS)
    for skenario in daftar_skenario:
        if skenario not in kf.BERKAS:
            print(f"Skenario '{skenario}' tidak dikenal. Pilihan: {list(kf.BERKAS)}")
            return

    folder_model = kf.folder("model")
    waktu_awal = time.perf_counter()
    jumlah_dikerjakan = 0

    for skenario in daftar_skenario:
        for kode_model in daftar_model:
            # Periksa dulu kondisi mana yang masih perlu dikerjakan, agar data
            # tidak dimuat bila kedua kondisi sudah selesai sebelumnya.
            perlu = []
            for nama_kondisi in kf.KONDISI:
                path_json = (f"{folder_model}/metrik_{skenario}_{kode_model}_"
                             f"{nama_kondisi}.json")
                if os.path.exists(path_json) and not ulang:
                    print(f"[lewati] {skenario} + {kode_model} + {nama_kondisi} sudah ada.")
                else:
                    perlu.append(nama_kondisi)

            if not perlu:
                print()
                continue

            parameter, skor_cv = ambil_hyperparameter(skenario, kode_model)

            print(f"\nMenyiapkan data untuk {skenario} + {kode_model} ...")
            data = kf.siapkan_data(skenario)

            for nama_kondisi in perlu:
                jalankan_kombinasi(skenario, kode_model, nama_kondisi,
                                   data, parameter, skor_cv)
                jumlah_dikerjakan += 1

            del data

    print("=" * 70)
    print(f"Selesai. {jumlah_dikerjakan} kombinasi dikerjakan pada jalan ini.")
    print(f"Total waktu: {time.perf_counter() - waktu_awal:.2f} detik "
          f"({(time.perf_counter() - waktu_awal) / 60:.1f} menit)")


if __name__ == "__main__":
    main()
