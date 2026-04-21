# =============================================================================
# TF-IDF Engine: Implementasi Manual & Scikit-learn
# Library: scikit-learn, numpy, pandas
# =============================================================================

import re
import math
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
from dataset import DOCUMENTS

# =============================================================================
# Daftar Stopwords Bahasa Indonesia
# =============================================================================
STOPWORDS_ID = {
    'yang', 'dan', 'di', 'ke', 'dari', 'ini', 'itu', 'dengan', 'untuk',
    'pada', 'adalah', 'sebagai', 'dalam', 'tidak', 'akan', 'juga', 'atau',
    'telah', 'oleh', 'ada', 'mereka', 'sudah', 'saat', 'bisa', 'serta',
    'hanya', 'setelah', 'tentang', 'secara', 'lebih', 'tersebut', 'karena',
    'dapat', 'hal', 'agar', 'antara', 'lain', 'namun', 'bahwa', 'banyak',
    'menjadi', 'seorang', 'seperti', 'ketika', 'dia', 'ia', 'kita', 'kami',
    'anda', 'saya', 'apa', 'siapa', 'mana', 'kapan', 'mengapa', 'bagaimana',
    'ya', 'tak', 'belum', 'masih', 'sangat', 'paling', 'lagi', 'harus',
    'bagi', 'hingga', 'semakin', 'begitu', 'meski', 'meskipun', 'tetap',
    'para', 'demi', 'sejak', 'tanpa', 'bahkan', 'pun', 'bila', 'jika',
    'apabila', 'sedang', 'sedangkan', 'sementara', 'selama', 'setiap',
    'semua', 'beberapa', 'berbagai', 'terhadap', 'melalui', 'sama',
    'sebuah', 'suatu', 'satu', 'dua', 'tiga', 'pernah', 'kembali',
    'sendiri', 'orang', 'tahun', 'atas', 'bawah', 'luar', 'antara',
    'baik', 'hampir', 'seluruh', 'terus', 'maupun', 'sejumlah', 'demikian',
    'merupakan', 'yaitu', 'yakni', 'dimana', 'kemudian', 'terkait',
    'termasuk', 'salah', 'perlu', 'mampu', 'mulai', 'sering', 'pihak',
    'upaya', 'rata', 'cukup', 'turut', 'ikut', 'bukan', 'justru',
}


# =============================================================================
# Preprocessing Teks
# =============================================================================
def preprocess_text(text):
    """
    Membersihkan dan memproses teks:
    1. Mengubah ke huruf kecil (lowercase)
    2. Menghapus tanda baca dan karakter khusus
    3. Tokenisasi (memecah menjadi kata-kata)
    4. Menghapus stopwords Bahasa Indonesia
    5. Menghapus kata dengan panjang < 3 karakter
    """
    # Lowercase
    text = text.lower()
    # Hapus angka dan tanda baca
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenisasi
    tokens = text.split()
    # Hapus stopwords dan kata pendek
    tokens = [t for t in tokens if t not in STOPWORDS_ID and len(t) >= 3]
    return tokens


# =============================================================================
# IMPLEMENTASI MANUAL TF-IDF
# =============================================================================

def compute_tf(tokens):
    """
    Menghitung Term Frequency (TF) untuk setiap kata dalam dokumen.
    TF(t,d) = Jumlah kemunculan term t dalam dokumen d / Total kata dalam dokumen d
    
    Args:
        tokens: list of strings (kata-kata hasil preprocessing)
    
    Returns:
        dict: {term: tf_value}
    """
    tf_dict = {}
    total_terms = len(tokens)
    if total_terms == 0:
        return tf_dict

    # Hitung frekuensi setiap term
    for token in tokens:
        tf_dict[token] = tf_dict.get(token, 0) + 1

    # Normalisasi dengan total kata
    for term in tf_dict:
        tf_dict[term] = tf_dict[term] / total_terms

    return tf_dict


def compute_df(tokenized_docs):
    """
    Menghitung Document Frequency (DF) untuk setiap term.
    DF(t) = Jumlah dokumen yang mengandung term t
    
    Args:
        tokenized_docs: list of list of strings
    
    Returns:
        dict: {term: df_value}
    """
    df_dict = {}
    for tokens in tokenized_docs:
        unique_terms = set(tokens)
        for term in unique_terms:
            df_dict[term] = df_dict.get(term, 0) + 1
    return df_dict


def compute_idf(tokenized_docs):
    """
    Menghitung Inverse Document Frequency (IDF) untuk setiap term.
    IDF(t) = log10(N / DF(t))   dimana N = jumlah total dokumen
    
    Args:
        tokenized_docs: list of list of strings
    
    Returns:
        dict: {term: idf_value}
    """
    N = len(tokenized_docs)
    df_dict = compute_df(tokenized_docs)
    idf_dict = {}

    for term, df in df_dict.items():
        idf_dict[term] = math.log10(N / df)

    return idf_dict


def compute_tfidf(tokenized_docs):
    """
    Menghitung TF-IDF untuk setiap term di setiap dokumen.
    TF-IDF(t,d) = TF(t,d) × IDF(t)
    
    Args:
        tokenized_docs: list of list of strings
    
    Returns:
        list of dict: [{term: tfidf_value}, ...] satu dict per dokumen
        dict: idf values
    """
    idf_dict = compute_idf(tokenized_docs)
    tfidf_list = []

    for tokens in tokenized_docs:
        tf = compute_tf(tokens)
        tfidf = {}
        for term, tf_val in tf.items():
            tfidf[term] = tf_val * idf_dict.get(term, 0)
        tfidf_list.append(tfidf)

    return tfidf_list, idf_dict


def cosine_similarity_manual(vec1, vec2):
    """
    Menghitung Cosine Similarity antara dua vektor.
    cos(θ) = (A · B) / (||A|| × ||B||)
    
    Args:
        vec1: dict {term: value}
        vec2: dict {term: value}
    
    Returns:
        float: nilai cosine similarity (0-1)
    """
    # Gabungkan semua term
    all_terms = set(vec1.keys()) | set(vec2.keys())

    # Hitung dot product
    dot_product = sum(vec1.get(t, 0) * vec2.get(t, 0) for t in all_terms)

    # Hitung magnitude
    mag1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
    mag2 = math.sqrt(sum(v ** 2 for v in vec2.values()))

    if mag1 == 0 or mag2 == 0:
        return 0.0

    return dot_product / (mag1 * mag2)


# =============================================================================
# KELAS UTAMA: TF-IDF Search Engine
# =============================================================================

class TFIDFSearchEngine:
    """
    Mesin pencari dokumen menggunakan TF-IDF dan Cosine Similarity.
    Mendukung implementasi manual dan scikit-learn.
    """

    def __init__(self):
        self.documents = DOCUMENTS
        self.tokenized_docs = []
        self.tfidf_manual = []
        self.idf_values = {}
        self.vocabulary = []

        # Scikit-learn components
        self.vectorizer = TfidfVectorizer(
            tokenizer=preprocess_text,
            token_pattern=None,
            lowercase=False  # preprocessing sudah handle lowercase
        )
        self.tfidf_matrix_sklearn = None

        # Build index
        self._build_index()

    def _build_index(self):
        """Membangun index TF-IDF untuk semua dokumen."""
        # Preprocessing semua dokumen
        corpus = [doc['content'] for doc in self.documents]
        self.tokenized_docs = [preprocess_text(content) for content in corpus]

        # ===== Manual TF-IDF =====
        self.tfidf_manual, self.idf_values = compute_tfidf(self.tokenized_docs)

        # Kumpulkan semua vocabulary
        vocab_set = set()
        for tokens in self.tokenized_docs:
            vocab_set.update(tokens)
        self.vocabulary = sorted(vocab_set)

        # ===== Scikit-learn TF-IDF =====
        self.tfidf_matrix_sklearn = self.vectorizer.fit_transform(corpus)

    def search_manual(self, query, top_k=5):
        """
        Pencarian menggunakan implementasi manual TF-IDF + Cosine Similarity.
        
        Args:
            query: string query pencarian
            top_k: jumlah hasil teratas yang ditampilkan
        
        Returns:
            list of dict: hasil pencarian dengan skor
        """
        # Preprocess query
        query_tokens = preprocess_text(query)
        if not query_tokens:
            return []

        # Hitung TF query
        query_tf = compute_tf(query_tokens)

        # Hitung TF-IDF query menggunakan IDF dari korpus
        query_tfidf = {}
        for term, tf_val in query_tf.items():
            query_tfidf[term] = tf_val * self.idf_values.get(term, 0)

        # Hitung cosine similarity dengan setiap dokumen
        results = []
        for i, doc_tfidf in enumerate(self.tfidf_manual):
            score = cosine_similarity_manual(query_tfidf, doc_tfidf)
            if score > 0:
                results.append({
                    'id': self.documents[i]['id'],
                    'title': self.documents[i]['title'],
                    'content': self.documents[i]['content'],
                    'score': round(score, 6),
                    'method': 'manual'
                })

        # Urutkan berdasarkan skor (descending)
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]

    def search_sklearn(self, query, top_k=5):
        """
        Pencarian menggunakan scikit-learn TF-IDF + Cosine Similarity.
        
        Args:
            query: string query pencarian
            top_k: jumlah hasil teratas yang ditampilkan
        
        Returns:
            list of dict: hasil pencarian dengan skor
        """
        # Transform query menggunakan vectorizer yang sudah di-fit
        query_vec = self.vectorizer.transform([query])

        # Hitung cosine similarity
        similarities = sklearn_cosine_similarity(query_vec, self.tfidf_matrix_sklearn).flatten()

        # Ambil top-k hasil
        top_indices = similarities.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = similarities[idx]
            if score > 0:
                results.append({
                    'id': self.documents[idx]['id'],
                    'title': self.documents[idx]['title'],
                    'content': self.documents[idx]['content'],
                    'score': round(float(score), 6),
                    'method': 'sklearn'
                })

        return results

    def search(self, query, top_k=5):
        """
        Melakukan pencarian dengan kedua metode dan mengembalikan hasil.
        
        Returns:
            dict: {manual: [...], sklearn: [...]}
        """
        return {
            'manual': self.search_manual(query, top_k),
            'sklearn': self.search_sklearn(query, top_k),
            'query': query,
            'query_tokens': preprocess_text(query),
            'top_k': top_k
        }

    def get_tf_matrix(self):
        """
        Mengembalikan matriks Term Frequency sebagai pandas DataFrame.
        """
        tf_data = []
        for tokens in self.tokenized_docs:
            tf = compute_tf(tokens)
            tf_data.append(tf)

        df = pd.DataFrame(tf_data, index=[f"Doc {d['id']:02d}" for d in self.documents])
        df = df.fillna(0)
        # Ambil top terms berdasarkan rata-rata TF
        top_terms = df.mean().nlargest(30).index.tolist()
        return df[top_terms].round(4).to_dict(orient='index')

    def get_idf_values(self):
        """
        Mengembalikan nilai IDF untuk setiap term sebagai dict.
        Diurutkan berdasarkan kata yang paling sering muncul di berbagai dokumen.
        """
        df_dict = compute_df(self.tokenized_docs)
        # Sort berdasarkan jumlah dokumen tempat term muncul (DF) dari yang terbanyak
        sorted_df = sorted(df_dict.items(), key=lambda x: x[1], reverse=True)
        return {term: round(self.idf_values[term], 4) for term, _ in sorted_df[:50]}

    def get_tfidf_matrix(self):
        """
        Mengembalikan matriks TF-IDF sebagai pandas DataFrame.
        """
        df = pd.DataFrame(self.tfidf_manual, index=[f"Doc {d['id']:02d}" for d in self.documents])
        df = df.fillna(0)
        # Ambil top terms berdasarkan rata-rata TF-IDF
        top_terms = df.mean().nlargest(30).index.tolist()
        return df[top_terms].round(4).to_dict(orient='index')

    def get_document_detail(self, doc_id):
        """
        Mengembalikan detail dokumen termasuk TF, IDF, dan TF-IDF.
        """
        idx = doc_id - 1
        if idx < 0 or idx >= len(self.documents):
            return None

        doc = self.documents[idx]
        tokens = self.tokenized_docs[idx]
        tf = compute_tf(tokens)
        tfidf = self.tfidf_manual[idx]

        # Buat tabel term detail
        term_details = []
        for term in sorted(tf.keys(), key=lambda t: tfidf.get(t, 0), reverse=True):
            term_details.append({
                'term': term,
                'tf': round(tf[term], 4),
                'idf': round(self.idf_values.get(term, 0), 4),
                'tfidf': round(tfidf.get(term, 0), 6)
            })

        return {
            'id': doc['id'],
            'title': doc['title'],
            'content': doc['content'],
            'total_tokens': len(tokens),
            'unique_terms': len(set(tokens)),
            'term_details': term_details
        }

    def get_all_documents(self):
        """Mengembalikan semua dokumen dengan statistik dasar."""
        docs = []
        for i, doc in enumerate(self.documents):
            tokens = self.tokenized_docs[i]
            docs.append({
                'id': doc['id'],
                'title': doc['title'],
                'content': doc['content'],
                'word_count': len(doc['content'].split()),
                'token_count': len(tokens),
                'unique_terms': len(set(tokens))
            })
        return docs

    def get_stats(self):
        """Mengembalikan statistik keseluruhan korpus."""
        all_tokens = [t for tokens in self.tokenized_docs for t in tokens]
        return {
            'total_documents': len(self.documents),
            'total_tokens': len(all_tokens),
            'unique_terms': len(self.vocabulary),
            'avg_doc_length': round(np.mean([len(t) for t in self.tokenized_docs]), 1),
            'vocabulary_sample': self.vocabulary[:20]
        }
