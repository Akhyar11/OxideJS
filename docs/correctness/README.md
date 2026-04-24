# Correctness Snapshots

Dokumentasi ini menyimpan snapshot correctness per versi agar perubahan benchmark selalu dibaca bersama bukti bahwa model masih belajar dan kontrak utama tidak rusak.

## Tujuan

- Menyimpan hasil correctness suite yang dijalankan pada versi tertentu.
- Membuat histori pass/fail untuk jalur training dan benchmark yang sensitif.
- Menjadi companion untuk `docs/benchmark-sintetis`.

## Struktur

- `README.md`: index utama correctness snapshot.
- `v<version>.md`: snapshot correctness untuk satu versi tertentu.

## Command Acuan

Command correctness resmi repo saat ini:

```bash
node -r ts-node/register test/correctness/index.ts
```

Command suite gabungan:

```bash
npm test
```

## Aturan Pengisian

1. Buat satu file versi baru untuk setiap snapshot correctness yang ingin dibekukan.
2. Catat command yang dipakai.
3. Catat status pass/fail.
4. Catat cakupan suite yang relevan.
5. Jika ada failure, tetap tulis penyebab singkat dan area yang terdampak.

## Daftar Versi

| Versi | Tanggal | Commit | Ringkasan |
| --- | --- | --- | --- |
| [v2.2.2](./v2.2.2.md) | 2026-04-24 | `7a0728f` + local patch | Snapshot correctness learning suite untuk recurrent dan transformer, termasuk `trimPad` |

## Versioning

Versi aktif proyek saat ini adalah `2.2.2`.

Proyek ini memakai format versi `MAJOR.MINOR.PATCH` seperti `2.2.2`.

- Angka paling depan (`MAJOR`): perubahan besar yang biasanya membawa breaking change atau perubahan arsitektur utama.
- Angka tengah (`MINOR`): penambahan fitur baru atau peningkatan yang tetap kompatibel dengan versi sebelumnya.
- Angka paling belakang (`PATCH`): perbaikan bug, optimasi kecil, cleanup, atau perubahan minor yang tidak mengubah API utama.
