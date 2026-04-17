# Benchmark Sintetis

Dokumentasi ini menyimpan riwayat benchmark sintetis per versi agar setiap peningkatan performa bisa dibandingkan terhadap baseline yang jelas.

## Tujuan

- Menyediakan baseline performa untuk setiap versi.
- Menyimpan konteks pengukuran agar hasil antar versi bisa dibandingkan dengan adil.
- Mencatat benchmark yang berhasil, benchmark yang gagal, dan catatan interpretasinya.

## Struktur

- `README.md`: index utama dan aturan pengisian.
- `TEMPLATE.md`: template untuk entri versi baru.
- `v<version>.md`: snapshot benchmark untuk satu versi tertentu.

## Aturan Pengisian

1. Buat satu file baru untuk setiap versi.
2. Gunakan format dari `TEMPLATE.md` agar konsisten.
3. Isi metadata minimum:
   - tanggal benchmark
   - versi aplikasi
   - commit acuan
   - environment singkat
4. Catat command yang dipakai untuk menjalankan benchmark.
5. Jika ada benchmark gagal, tetap tulis hasilnya sebagai `failed` beserta error ringkas.
6. Jangan menimpa file versi lama. Riwayat harus append-only.

## Daftar Versi

| Versi | Tanggal | Commit | Ringkasan |
| --- | --- | --- | --- |
| [v1.0.0](./v1.0.0.md) | 2026-04-17 | `47e7734` | Baseline awal synthetic benchmark untuk versi sekarang |

## Cara Menambah Versi Baru

1. Salin `TEMPLATE.md` menjadi file versi baru, misalnya `v1.1.0.md`.
2. Jalankan benchmark sintetis yang ingin dijadikan acuan.
3. Isi hasil, status, dan catatan interpretasi.
4. Tambahkan entri baru ke tabel pada file ini.

## Catatan Pembacaan

- Hasil benchmark hanya valid jika konfigurasi model dan environment pengujian dicatat dengan jelas.
- Perbandingan antar versi sebaiknya fokus pada benchmark yang sama, command yang sama, dan kondisi backend yang sama.
- Benchmark yang gagal tetap penting karena bisa menunjukkan regresi, mismatch konfigurasi, atau masalah validitas harness.
