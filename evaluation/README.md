# Evaluation

Mengukur kualitas pipeline RAG HealthTruth-AI (`app/rag_engine.py`) menggunakan
[DeepEval](https://github.com/confident-ai/deepeval) dan [RAGAS](https://github.com/explodinggradients/ragas),
plus akurasi klasifikasi HOAX / BENAR / TIDAK LENGKAP terhadap `dataset.json`.

Folder ini **self-contained**: berisi dataset uji sekaligus corpus sumber sendiri, terpisah dari
`data/raw/` (tempat dokumen produksi WHO/Kemenkes yang sebenarnya akan ditaruh user). Clone repo,
install dependency, isi API key — evaluasi langsung bisa jalan tanpa setup data tambahan.

## Isi folder

- `dataset.json` — 20 klaim kesehatan ala pesan WhatsApp (hoax & fakta umum: COVID-19, vaksin, nutrisi,
  gaya hidup, obat), masing-masing dengan `expected_status` dan `reference_answer` (ground truth).
- `corpus/` — corpus sumber contoh (gaya WHO/Kemenkes/CDC) yang melandasi jawaban ke-20 klaim di atas,
  plus `hoax_samples.json`. Ini yang membuat folder ini reproducible — **tetap tracked di git**,
  beda dari `data/raw/` produksi yang di-gitignore.
- `build_eval_index.py` — menjalankan `app/preprocess.py` + `app/build_index.py` dengan path di-arahkan
  ke `corpus/` (input) dan `index/` (output), lewat override env var di `app/config.py`. Tidak pernah
  menyentuh `data/processed/` atau `vectorstore/` milik aplikasi produksi.
- `index/` — hasil build (`chunks.json`, `embeddings.pkl`, `index.faiss`) — gitignored, dibuat otomatis.
- `common.py` — memastikan index evaluasi sudah dibangun, lalu menjalankan `RAGEngine` sekali atas
  seluruh dataset dan menyimpan hasilnya ke `results/predictions.json`, supaya `run_deepeval.py` dan
  `run_ragas.py` tidak memanggil LLM dua kali.
- `llm_judge.py` — LLM judge (default `openai/gpt-4o-mini` via OpenRouter) yang dipakai kedua framework,
  jadi cukup pakai `OPENROUTER_API_KEY` yang sudah ada di `env/.env`, tanpa perlu API key terpisah.
- `run_deepeval.py` — metrik Faithfulness, Answer Relevancy, Contextual Precision/Recall, Hallucination,
  ditambah akurasi klasifikasi status.
- `run_ragas.py` — metrik Faithfulness, Answer Relevancy, Context Precision, Context Recall dari RAGAS.
- `results/` — output evaluasi (predictions, laporan) — gitignored karena berisi hasil run, bukan kode.

## Prasyarat

Install dependency evaluasi (di atas `requirements.txt` utama) dan siapkan `OPENROUTER_API_KEY` di
`env/.env` (dipakai juga oleh aplikasi utama):

```bash
pip install -r evaluation/requirements.txt
```

Tidak perlu menjalankan `app/preprocess.py` / `app/build_index.py` manual — `common.py` otomatis
memanggil `build_eval_index.py` saat index evaluasi belum ada.

## Menjalankan

Jalankan dari root proyek (path data di `config.py` relatif terhadap root):

```bash
# (Opsional) build/rebuild index evaluasi secara eksplisit
python evaluation/build_eval_index.py

# Generate prediksi (dipakai bersama oleh kedua script)
python evaluation/common.py

# Skor DeepEval + akurasi klasifikasi
python evaluation/run_deepeval.py

# Skor RAGAS
python evaluation/run_ragas.py
```

Kedua script otomatis men-generate `results/predictions.json` jika belum ada. Tambahkan `--regenerate`
untuk memaksa menjalankan ulang pipeline RAG (misalnya setelah mengubah prompt atau `dataset.json`):

```bash
python evaluation/run_deepeval.py --regenerate
```

Setelah mengubah isi `corpus/` atau `VECTOR_BACKEND`, hapus `evaluation/index/` (atau jalankan
`build_eval_index.py` lagi) supaya index lama tidak dipakai:

```bash
rm -rf evaluation/index
python evaluation/run_deepeval.py --regenerate
```

Ganti model judge lewat env var bila perlu:

```bash
export EVAL_JUDGE_MODEL="openai/gpt-4o-mini"   # model apa pun yang tersedia di OpenRouter
```

## Evaluasi terhadap Qdrant (staging/production)

`build_eval_index.py` dan `common.py` mengambil backend vector store dari `VECTOR_BACKEND`
(lihat [`app/vector_store.py`](../app/vector_store.py) dan bagian "Vector Backend" di README root).
Untuk mengevaluasi backend Qdrant, set env var sebelum menjalankan — tidak perlu mengubah kode:

```bash
export VECTOR_BACKEND=qdrant
export QDRANT_URL=https://xxxx.qdrant.io   # atau instance staging; QDRANT_COLLECTION default
export QDRANT_API_KEY=xxxx                 # "healthtruth_chunks_eval" biar tidak bentrok dgn produksi

rm -rf evaluation/index                    # index lama (faiss) tidak relevan lagi
python evaluation/build_eval_index.py      # build corpus contoh ke koleksi Qdrant
python evaluation/run_deepeval.py --regenerate
python evaluation/run_ragas.py --regenerate
```

Corpus yang dipakai tetap `evaluation/corpus/` (bukan data produksi), jadi run ini tidak pernah
menyentuh koleksi Qdrant produksi kecuali `QDRANT_COLLECTION` sengaja diarahkan ke sana. Disarankan
menembak Qdrant **staging** untuk run rutin, bukan produksi langsung — supaya tidak menambah beban ke
server yang dipakai user asli dan tidak perlu menaruh kredensial produksi di CI.

## Catatan

- `context` dan `retrieval_context` pada test case DeepEval memakai chunk yang sama (hasil retrieval
  aktual), karena corpus contoh ini belum punya "ideal context" terpisah dari korpus retrieval — jadi
  Hallucination metric pada dasarnya menilai konsistensi jawaban terhadap potongan yang benar-benar
  diambil sistem.
- `reference_answer` di `dataset.json` dipakai sebagai `expected_output` (DeepEval) dan `ground_truth`
  (RAGAS) — tulisan tangan, merujuk gaya WHO/CDC/Kemenkes, bukan hasil scraping dokumen sumber.
- File di `corpus/` berlabel jelas sebagai data contoh (bukan dokumen resmi asli) — dibuat khusus untuk
  membuat evaluasi ini reproducible, bukan untuk dipakai sebagai basis pengetahuan produksi.
