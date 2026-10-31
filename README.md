# Analisis Pengaruh Penanganan Class Imbalance terhadap Deteksi Serangan DoS dan Infiltration pada Trafik Jaringan CSE-CIC-IDS2018

Repository ini berisi kode program untuk skripsi saya di Program Studi Teknik
Informatika, Fakultas MIPA, Universitas Padjadjaran.

Penelitian ini membandingkan performa deteksi tiga model klasifikasi (Logistic
Regression, Decision Tree, dan Random Forest) pada dua skenario biner yang
dijalankan secara terpisah, yaitu Normal vs DoS dan Normal vs Infiltration.
Masing-masing skenario diuji dalam dua kondisi: tanpa penanganan class imbalance
dan dengan class weighting, sehingga seluruhnya ada 12 kombinasi eksperimen.
Fokus penelitiannya adalah menganalisis pengaruh penanganan class imbalance
terhadap performa deteksi kedua kategori serangan, bukan mencari model dengan
performa tertinggi.

## Status pengerjaan

- [x] Uji coba awal (studi kelayakan)
- [x] Eksperimen utama (12 kombinasi)
- [ ] Dashboard visualisasi

## Dataset

Berkas dataset tidak disertakan di repository ini karena ukurannya besar dan
harus diunduh langsung dari sumbernya.

- Dataset asli: CSE-CIC-IDS2018, Canadian Institute for Cybersecurity (CIC),
  University of New Brunswick.
- Versi yang dipakai: versi yang sudah dibersihkan dan disimpan dalam format
  parquet, diunduh dari Kaggle: <https://www.kaggle.com/datasets/dhoogla/csecicids2018>
- Dua berkas yang dipakai:
  - `DoS1-Thursday-15-02-2018_TrafficForML_CICFlowMeter.parquet` (794.812 baris)
  - `Infil1-Wednesday-28-02-2018_TrafficForML_CICFlowMeter.parquet` (456.873 baris)

Pada versi ini, kolom metadata (Flow ID, alamat IP, port, dan timestamp) sudah
dihapus, baris duplikat dan baris bernilai kosong atau tak terhingga sudah
dibuang, serta tipe data sudah diperkecil. Setiap berkas memiliki 78 kolom,
yaitu 77 kolom fitur dan 1 kolom Label.

## Cara menjalankan

1. Unduh kedua berkas parquet di atas, lalu letakkan pada satu folder.
2. Sesuaikan nilai `FOLDER_DATA` pada `konfigurasi.py` dan pada skrip `01` dan `02`
   agar menunjuk ke folder tersebut.
3. Pasang pustaka yang dibutuhkan:

   ```
   pip install -r requirements.txt
   ```

4. Jalankan skrip secara berurutan sesuai nomornya:

   ```
   python 01_cek_dataset.py
   python 02_uji_coba_awal_rf.py
   python 03_pemeriksaan_outlier.py
   python 04_validasi_silang.py dt
   python 04_validasi_silang.py rf
   python 04_validasi_silang.py lr
   python 05_latih_dan_evaluasi.py
   python 06_ringkas_hasil.py
   ```

Skrip `04` dan `05` melewati bagian yang hasilnya sudah tersimpan, sehingga aman
dijalankan ulang bila prosesnya terputus. Keluaran seluruh skrip tersimpan di
dalam folder `hasil/`.

Perkiraan waktu pada perangkat yang dipakai: skrip `04` sekitar 29 menit untuk
keenam prosedur, dan skrip `05` sekitar 10 menit untuk kedua belas kombinasi.

## Isi repository

```
.
├── 01_cek_dataset.py           pemeriksaan kondisi dan distribusi kelas
├── 02_uji_coba_awal_rf.py      uji coba awal Random Forest (studi kelayakan)
├── konfigurasi.py              pengaturan bersama skrip 03 sampai 06
├── 03_pemeriksaan_outlier.py   pemeriksaan nilai pencilan dengan metode IQR
├── 04_validasi_silang.py       pemilihan hyperparameter, 6 pasangan
├── 05_latih_dan_evaluasi.py    pelatihan dan evaluasi 12 kombinasi
├── 06_ringkas_hasil.py         peringkasan hasil menjadi tabel CSV
├── requirements.txt            versi pustaka yang digunakan
└── hasil/
    ├── outlier/                hasil pemeriksaan IQR
    ├── cv/                     hyperparameter terpilih dan skor kandidat
    ├── model/                  metrik, probabilitas prediksi, feature importance
    └── ringkasan/              tabel gabungan seluruh kombinasi
```

## Hasil

### Studi kelayakan awal

Random Forest (`n_estimators=100`, `random_state=42`) tanpa penanganan class
imbalance, pembagian data 70:30 secara stratified.

| Skenario | Akurasi | Precision | Recall | F1-score |
|---|---|---|---|---|
| DoS | 99,99% | 99,88% | 99,92% | 99,90% |
| Infiltration | 79,88% | 5,30% | 3,73% | 4,38% |

### Eksperimen utama

Hyperparameter ditentukan melalui validasi silang berstrata 5 lipatan pada kondisi
netral, lalu diterapkan sama pada kedua kondisi.

Skenario DoS (pembanding acak PR-AUC = 0,0646):

| Model | Kondisi | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|---|
| Logistic Regression | netral | 0,9863 | 0,9968 | 0,9915 | 0,9961 |
| Logistic Regression | balanced | 0,9707 | 0,9994 | 0,9848 | 0,9948 |
| Decision Tree | netral | 0,9987 | 0,9991 | 0,9989 | 0,9980 |
| Decision Tree | balanced | 0,9985 | 0,9994 | 0,9990 | 0,9982 |
| Random Forest | netral | 0,9988 | 0,9993 | 0,9991 | 0,9999 |
| Random Forest | balanced | 0,9986 | 0,9997 | 0,9992 | 0,9992 |

Skenario Infiltration (pembanding acak PR-AUC = 0,1236):

| Model | Kondisi | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|---|
| Logistic Regression | netral | 0,0000 | 0,0000 | 0,0000 | 0,1513 |
| Logistic Regression | balanced | 0,1392 | 0,6491 | 0,2293 | 0,1515 |
| Decision Tree | netral | 0,0887 | 0,0785 | 0,0833 | 0,1195 |
| Decision Tree | balanced | 0,0824 | 0,1056 | 0,0926 | 0,1192 |
| Random Forest | netral | 0,0530 | 0,0373 | 0,0438 | 0,1306 |
| Random Forest | balanced | 0,0996 | 0,1307 | 0,1130 | 0,1286 |

Nilai pembanding acak dicantumkan karena PR-AUC tidak memiliki titik nol yang
berlaku umum. Model yang menebak secara acak akan memperoleh PR-AUC kira-kira
sebesar proporsi kelas serangan pada data uji.

Catatan yang perlu dibaca bersama tabel di atas: pada skenario Infiltration,
kenaikan recall akibat class weighting tidak disertai kenaikan PR-AUC. Pada
Logistic Regression, recall naik dari 0% menjadi 64,9% sementara PR-AUC hanya
bergerak dari 0,1513 menjadi 0,1515. Pada Decision Tree dan Random Forest, recall
naik tetapi PR-AUC justru turun tipis. Pembahasan lengkapnya ada di naskah skripsi.

## Lingkungan pengujian

- Laptop ASUS Vivobook Go E1404FA, AMD Ryzen 3 7320U, RAM 8 GB, tanpa GPU
- Windows 11 Home Single Language 25H2 (64-bit)
- Python 3.13.7

Versi pustaka yang dipakai tercantum pada `requirements.txt` dan juga ikut
tercetak pada setiap berkas keluaran di folder `hasil/`.

## Catatan

Seluruh proses menggunakan `random_state=42` agar hasilnya dapat direproduksi.
Probabilitas prediksi dan label data uji disimpan dalam format `.npy` pada
`hasil/model/`, sehingga nilai PR-AUC dapat dihitung ulang tanpa melatih model
kembali.
