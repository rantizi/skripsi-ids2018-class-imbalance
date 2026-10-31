"""
Menggabungkan keluaran tahap sebelumnya menjadi berkas CSV yang siap dipakai sebagai
tabel pada BAB IV. Hanya membaca berkas, tidak menghitung ulang apa pun.

Jalankan : python 06_ringkas_hasil.py
Keluaran : hasil/ringkasan/
"""

import json
import os

import pandas as pd

import konfigurasi as kf

# Banyaknya fitur teratas yang dibandingkan antar kombinasi Random Forest.
JUMLAH_FITUR_DIBANDINGKAN = 20


def baca_json(path):
    with open(path, encoding="utf-8") as berkas:
        return json.load(berkas)


def kumpulkan_hyperparameter(catat):
    folder_cv = os.path.join(kf.FOLDER_HASIL, "cv")
    baris = []
    hilang = []

    for skenario in kf.BERKAS:
        for kode_model in kf.NAMA_MODEL:
            path = os.path.join(folder_cv, f"cv_{skenario}_{kode_model}.json")
            if not os.path.exists(path):
                hilang.append(f"cv_{skenario}_{kode_model}.json")
                continue

            isi = baca_json(path)
            terbaik = isi["hasil_setiap_kandidat"][0]
            kedua = (isi["hasil_setiap_kandidat"][1]
                     if len(isi["hasil_setiap_kandidat"]) > 1 else None)

            baris.append({
                "skenario": skenario,
                "model": isi["nama_model"],
                "hyperparameter_terpilih": ", ".join(
                    f"{k.replace('clf__', '')} = {v}"
                    for k, v in isi["hyperparameter_terpilih"].items()),
                "f1_validasi_silang": round(isi["skor_f1_terbaik"], 6),
                "simpangan_baku": round(terbaik["skor_simpangan_baku"], 6),
                "kandidat_peringkat_2": (", ".join(
                    f"{k.replace('clf__', '')} = {v}"
                    for k, v in kedua["parameter"].items()) if kedua else ""),
                "f1_peringkat_2": round(kedua["skor_rata_rata"], 6) if kedua else "",
                "selisih_dengan_peringkat_2": (
                    round(isi["skor_f1_terbaik"] - kedua["skor_rata_rata"], 6)
                    if kedua else ""),
                "waktu_pencarian_detik": isi["waktu_pencarian_detik"],
            })

    if hilang:
        catat(f"Peringatan: berkas validasi silang belum ada -> {', '.join(hilang)}")
    return pd.DataFrame(baris)


def kumpulkan_metrik(catat):
    folder_model = os.path.join(kf.FOLDER_HASIL, "model")
    metrik, matriks, mentah = [], [], {}
    hilang = []

    for skenario in kf.BERKAS:
        for kode_model in kf.NAMA_MODEL:
            for nama_kondisi in kf.KONDISI:
                nama = f"metrik_{skenario}_{kode_model}_{nama_kondisi}.json"
                path = os.path.join(folder_model, nama)
                if not os.path.exists(path):
                    hilang.append(nama)
                    continue

                isi = baca_json(path)
                m = isi["metrik"]
                c = isi["confusion_matrix"]
                mentah[(skenario, kode_model, nama_kondisi)] = isi

                metrik.append({
                    "skenario": skenario,
                    "model": isi["nama_model"],
                    "kondisi": nama_kondisi,
                    "hyperparameter": ", ".join(
                        f"{k.replace('clf__', '')} = {v}"
                        for k, v in isi["hyperparameter"].items()),
                    "precision": round(m["precision"], 6),
                    "recall": round(m["recall"], 6),
                    "f1": round(m["f1"], 6),
                    "pr_auc": round(m["pr_auc"], 6),
                    "pr_auc_pembanding_acak": round(m["pr_auc_pembanding_acak"], 6),
                    "akurasi": round(m["akurasi"], 6),
                    "waktu_pelatihan_detik": isi["waktu_pelatihan_detik"],
                })

                matriks.append({
                    "skenario": skenario,
                    "model": isi["nama_model"],
                    "kondisi": nama_kondisi,
                    "true_negative": c["true_negative"],
                    "false_positive": c["false_positive"],
                    "false_negative": c["false_negative"],
                    "true_positive": c["true_positive"],
                    "jumlah_normal": c["true_negative"] + c["false_positive"],
                    "jumlah_serangan": c["false_negative"] + c["true_positive"],
                })

    if hilang:
        catat(f"Peringatan: berkas metrik belum ada -> {', '.join(hilang)}")
    return pd.DataFrame(metrik), pd.DataFrame(matriks), mentah


def bandingkan_kondisi(mentah, catat):
    """Menyandingkan kondisi netral dan balanced untuk tiap pasangan skenario-model."""
    baris = []
    for skenario in kf.BERKAS:
        for kode_model in kf.NAMA_MODEL:
            netral = mentah.get((skenario, kode_model, "netral"))
            balanced = mentah.get((skenario, kode_model, "balanced"))
            if netral is None or balanced is None:
                continue

            data = {
                "skenario": skenario,
                "model": netral["nama_model"],
                "hyperparameter": ", ".join(
                    f"{k.replace('clf__', '')} = {v}"
                    for k, v in netral["hyperparameter"].items()),
            }
            for nama_metrik in ["precision", "recall", "f1", "pr_auc", "akurasi"]:
                nilai_n = netral["metrik"][nama_metrik]
                nilai_b = balanced["metrik"][nama_metrik]
                data[f"{nama_metrik}_netral"] = round(nilai_n, 6)
                data[f"{nama_metrik}_balanced"] = round(nilai_b, 6)
                data[f"{nama_metrik}_selisih"] = round(nilai_b - nilai_n, 6)

            data["pr_auc_pembanding_acak"] = round(
                netral["metrik"]["pr_auc_pembanding_acak"], 6)
            data["false_positive_netral"] = netral["confusion_matrix"]["false_positive"]
            data["false_positive_balanced"] = balanced["confusion_matrix"]["false_positive"]
            data["false_negative_netral"] = netral["confusion_matrix"]["false_negative"]
            data["false_negative_balanced"] = balanced["confusion_matrix"]["false_negative"]
            baris.append(data)

    return pd.DataFrame(baris)


def bandingkan_importance(catat):
    """Menyandingkan peringkat fitur dari keempat kombinasi Random Forest."""
    folder_model = os.path.join(kf.FOLDER_HASIL, "model")
    kolom = {}
    hilang = []

    for skenario in kf.BERKAS:
        for nama_kondisi in kf.KONDISI:
            nama = f"importance_{skenario}_rf_{nama_kondisi}.csv"
            path = os.path.join(folder_model, nama)
            if not os.path.exists(path):
                hilang.append(nama)
                continue

            tabel = pd.read_csv(path).set_index("fitur")["importance"]
            label = f"{skenario}_{nama_kondisi}"
            kolom[f"{label}_importance"] = tabel.round(6)
            kolom[f"{label}_peringkat"] = tabel.rank(ascending=False, method="min").astype(int)

    if hilang:
        catat(f"Peringatan: berkas feature importance belum ada -> {', '.join(hilang)}")
    if not kolom:
        return pd.DataFrame()

    gabungan = pd.DataFrame(kolom)
    kolom_peringkat = [k for k in gabungan.columns if k.endswith("_peringkat")]
    gabungan["peringkat_terbaik"] = gabungan[kolom_peringkat].min(axis=1)
    gabungan = gabungan.sort_values("peringkat_terbaik")
    return gabungan.head(JUMLAH_FITUR_DIBANDINGKAN).drop(columns="peringkat_terbaik")


def main():
    catat = kf.Pencatat()
    folder_keluaran = kf.folder("ringkasan")

    catat("RINGKASAN HASIL EKSPERIMEN UTAMA")
    catat("")
    for baris in kf.info_lingkungan():
        catat(baris)

    # ---------- Hyperparameter ----------
    catat.garis("1. HYPERPARAMETER TERPILIH")
    tabel_hp = kumpulkan_hyperparameter(catat)
    if not tabel_hp.empty:
        tabel_hp.to_csv(f"{folder_keluaran}/ringkasan_hyperparameter.csv", index=False)
        catat("")
        catat(tabel_hp[["skenario", "model", "hyperparameter_terpilih",
                        "f1_validasi_silang", "simpangan_baku",
                        "selisih_dengan_peringkat_2"]].to_string(index=False))
        catat("")
        catat("Kolom selisih_dengan_peringkat_2 menunjukkan jarak skor antara kandidat")
        catat("terpilih dan kandidat terbaik berikutnya. Bila nilainya lebih kecil")
        catat("daripada simpangan baku antarlipatan, pemilihan tersebut tidak dapat")
        catat("ditafsirkan sebagai keunggulan yang bermakna.")

    # ---------- Metrik ----------
    catat.garis("2. METRIK 12 KOMBINASI")
    tabel_metrik, tabel_matriks, mentah = kumpulkan_metrik(catat)
    if not tabel_metrik.empty:
        tabel_metrik.to_csv(f"{folder_keluaran}/ringkasan_metrik.csv", index=False)
        tabel_matriks.to_csv(f"{folder_keluaran}/ringkasan_confusion_matrix.csv", index=False)
        catat("")
        catat(tabel_metrik[["skenario", "model", "kondisi", "precision", "recall",
                            "f1", "pr_auc", "akurasi"]].to_string(index=False))

        catat("")
        catat("Confusion matrix:")
        catat(tabel_matriks.to_string(index=False))

    # ---------- Perbandingan kondisi ----------
    catat.garis("3. PERBANDINGAN KONDISI NETRAL DAN BALANCED")
    tabel_banding = bandingkan_kondisi(mentah, catat)
    if not tabel_banding.empty:
        tabel_banding.to_csv(f"{folder_keluaran}/perbandingan_kondisi.csv", index=False)
        catat("")
        catat(tabel_banding[["skenario", "model", "recall_netral", "recall_balanced",
                             "recall_selisih", "pr_auc_netral", "pr_auc_balanced",
                             "pr_auc_selisih"]].to_string(index=False))
        catat("")
        catat("Recall dan PR-AUC disandingkan karena keduanya menjawab pertanyaan yang")
        catat("berbeda. Recall diukur pada satu ambang keputusan, sedangkan PR-AUC")
        catat("mengukur kualitas pemeringkatan pada seluruh ambang. Kenaikan recall yang")
        catat("tidak disertai kenaikan PR-AUC menunjukkan pergeseran ambang keputusan,")
        catat("bukan perbaikan kemampuan model membedakan kedua kelas.")

    # ---------- Feature importance ----------
    catat.garis("4. PERBANDINGAN FEATURE IMPORTANCE (RANDOM FOREST)")
    tabel_importance = bandingkan_importance(catat)
    if not tabel_importance.empty:
        tabel_importance.to_csv(f"{folder_keluaran}/perbandingan_importance.csv")
        catat("")
        kolom_peringkat = [k for k in tabel_importance.columns if k.endswith("_peringkat")]
        catat(tabel_importance[kolom_peringkat].to_string())
        catat("")
        catat(f"Ditampilkan {JUMLAH_FITUR_DIBANDINGKAN} fitur yang menempati peringkat")
        catat("tertinggi pada setidaknya satu dari empat kombinasi Random Forest.")
        catat("Nilai importance selengkapnya ada pada berkas CSV.")

    catat.garis("SELESAI")
    catat(f"Seluruh berkas ringkasan disimpan di: {os.path.abspath(folder_keluaran)}")
    catat.simpan(f"{folder_keluaran}/hasil_06_ringkasan.txt")


if __name__ == "__main__":
    main()
