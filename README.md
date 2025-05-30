# Cyber Bullying Classification By Group 6

## Deskripsi Proyek
Proyek ini bertujuan untuk membangun model machine learning yang dapat mengklasifikasikan tweet mengandung cyberbullying berdasarkan kategori tertentu (gender, agama, ras, dll.). Dataset yang digunakan terdiri dari 47.692 tweet yang sudah dilabeli.

## Daftar Isi📌
- [Dataset]
- [Pra-pemrosesan]
- [Eksperimen Model]
- [Hasil]

## Dataset📂
Dataset berasal dari file cyberbullying_tweets.csv dengan kolom:
- tweet_text: Teks tweet.
- cyberbullying_type: Label kategori (e.g., "gender", "religion", "not_cyberbullying").

## Pra-pemrosesan🛠️
Langkah-langkah pra-pemrosesan teks:
1. Pembersihan Teks: Menghapus URL, mention (@), dan karakter khusus.
2. Tokenisasi: Memecah teks menjadi kata per kata.
3. Stopword Removal: Menghilangkan kata umum (contoh: "yang", "di").
4. Lemmatisasi: Mengubah kata ke bentuk dasar.
5. TF-IDF Vectorization: Konversi teks ke vektor numerik.

## Eksperimen Model🤖
Model yang diuji:
1. Logistic Regression
2. Random Forest
3. SVM
4. XGBoost
5. Naive Bayes

Metrik Evaluasi:
1. Akurasi
2. Presisi
3. Recall
4. F1-Score

## Hasil📊
1. Model terbaik adalah SVM dimana mencapai akurasi 81.42% pada data validasi.
2. Deploy Model Menggunakan Streamlit

## Run steamlit app
```
streamlit run cyber bullying_streamlit.py
```

