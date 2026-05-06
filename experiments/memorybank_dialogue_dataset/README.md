# MemoryBank Dialogue Experiment

Eksperimen ini melatih alur:

`Encoder Embedding -> MemoryBank (sequential per token) -> Decoder Embedding -> LSTM -> Shared Dense`

Target training diambil dari setiap giliran `user`, dengan label berupa giliran `assistant` berikutnya. Dengan begitu model belajar:

- menulis fakta ke memori saat user memberi informasi
- membaca memori saat user bertanya
- menghasilkan respons token demi token secara autoregressive

## Jalankan Training

```bash
npx ts-node experiments/memorybank_dialogue_dataset/train.ts
```

Contoh override ringan:

```bash
EPOCHS=10 VOCAB_SIZE=512 EMBEDDING_DIM=96 MEMORY_SLOTS=24 npx ts-node experiments/memorybank_dialogue_dataset/train.ts
```

Artefak akan disimpan ke:

- `experiments/memorybank_dialogue_dataset/artifacts/dialogue_memory_model.json`
- `experiments/memorybank_dialogue_dataset/artifacts/dialogue_tokenizer.json`

## Jalankan Chat

Setelah training selesai:

```bash
npx ts-node experiments/memorybank_dialogue_dataset/chat.ts
```

Command interaktif:

- `/reset` untuk reset memori episode
- `exit` untuk keluar
