# Analisis Pengaruh Penanganan Class Imbalance terhadap Deteksi Serangan DoS dan Infiltration pada Trafik Jaringan CSE-CIC-IDS2018

Repository ini berisi kode program untuk skripsi saya di Program Studi Teknik
Informatika, Fakultas MIPA, Universitas Padjadjaran.

Penelitian ini membandingkan performa deteksi tiga model klasifikasi (Logistic
Regression, Decision Tree, dan Random Forest) pada dua skenario biner yang
dijalankan secara terpisah, yaitu Normal vs DoS dan Normal vs Infiltration.
Masing-masing skenario diuji dalam dua kondisi: tanpa penanganan class imbalance
dan dengan class weighting. Fokus penelitiannya adalah menganalisis pengaruh
penanganan class imbalance terhadap performa deteksi kedua kategori serangan,
bukan mencari model dengan performa tertinggi.

## Status pengerjaan

- [x] Uji coba awal (studi kelayakan)
- [ ] Eksperimen utama (12 kombinasi)
- [ ] Dashboard visualisasi

## Dataset

Berkas dataset tidak disertakan di repository ini karena ukurannya besar dan
harus diunduh langsung dari sumbernya.

- Dataset asli: CSE-CIC-IDS2018, Canadian Institute for Cybersecurity (CIC),
  University of New Brunswick.
- Versi yang dipakai: versi yang sudah dibersihkan dan disimpan dalam format
  parquet, diunduh dari Kaggle: <ISI TAUTAN KAGGLE>
- Dua berkas yang dipakai:
  - `DoS1-Thursday-15-02-2018_TrafficForML_CICFlowMeter.parquet` (794.812 baris)
  - `Infil1-Wednesday-28-02-2018_TrafficForML_CICFlowMeter.parquet` (456.873 baris)

Pada versi ini, kolom metadata (Flow ID, alamat IP, port, dan timestamp) sudah
dihapus, baris duplikat dan baris bernilai kosong atau tak terhingga sudah
dibuang, serta tipe data sudah diperkecil. Setiap berkas memiliki 78 kolom,
yaitu 77 kolom fitur dan 1 kolom Label.

## Cara menjalankan

1. Unduh kedua berkas parquet di atas, lalu letakkan pada satu folder.
2. Sesuaikan nilai `FOLDER_DATA` di bagian konfigurasi pada masing-masing skrip
   agar menunjuk ke folder tersebut.
3. Pasang pustaka yang dibutuhkan:

   ```
   pip install -r requirements.txt
   ```

4. Jalankan kedua skrip secara berurutan:

   ```
   python 01_cek_dataset.py
   python 02_uji_coba_awal_rf.py
   ```

Keluaran kedua skrip tercetak di layar sekaligus tersimpan sebagai berkas teks
di dalam folder `hasil/`.

## Isi repository

```
.
├── 01_cek_dataset.py          pemeriksaan kondisi dan distribusi kelas kedua berkas
├── 02_uji_coba_awal_rf.py     uji coba awal Random Forest pada dua skenario
├── requirements.txt           versi pustaka yang digunakan
└── hasil/                     keluaran kedua skrip dalam bentuk berkas teks
```

## Hasil uji coba awal

Random Forest (`n_estimators=100`, `random_state=42`) tanpa penanganan class
imbalance, pembagian data 70:30 secara stratified.

| Skenario | Akurasi | Precision | Recall | F1-score |
|---|---|---|---|---|
| DoS | 99,99% | 99,88% | 99,92% | 99,90% |
| Infiltration | 79,88% | 5,30% | 3,73% | 4,38% |

Angka di atas merupakan hasil studi kelayakan awal. Hasil ini belum melalui
validasi silang, pemilihan hyperparameter, maupun penanganan class imbalance,
sehingga belum dapat dianggap sebagai hasil akhir penelitian.

Catatan dari pemeriksaan awal potensi kebocoran data: pada skenario
Infiltration, Decision Tree dengan kedalaman 1 sampai 3 menghasilkan akurasi
87,64%, yaitu persis sama dengan proporsi kelas normal pada data uji. Artinya
pohon dangkal tidak menemukan fitur yang membedakan kedua kelas. Sebaliknya,
pada skenario DoS, pohon dengan kedalaman 1 sudah mencapai akurasi 99,30%.

## Lingkungan pengujian

- Laptop ASUS Vivobook Go E1404FA, AMD Ryzen 3 7320U, RAM 8 GB, tanpa GPU
- Windows 11 Home Single Language 25H2 (64-bit)
- Python 3.13.7

Versi pustaka yang dipakai tercantum pada `requirements.txt` dan juga ikut
tercetak pada setiap berkas keluaran di folder `hasil/`.

## Catatan

Seluruh proses menggunakan `random_state=42` agar hasilnya dapat direproduksi.
Berkas keluaran di folder `hasil/` dapat dibandingkan langsung dengan hasil
menjalankan ulang skrip pada komputer lain.
