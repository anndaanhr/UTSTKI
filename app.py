# =============================================================================
# Flask Web Server — API & Frontend
# =============================================================================

from flask import Flask, render_template, jsonify, request
from tfidf_engine import TFIDFSearchEngine

app = Flask(__name__)
engine = TFIDFSearchEngine()


@app.route('/')
def index():
    """Halaman utama aplikasi."""
    return render_template('index.html')


@app.route('/api/search', methods=['POST'])
def search():
    """
    API Pencarian Dokumen.
    Body: { "query": "...", "top_k": 5 }
    """
    data = request.get_json()
    query = data.get('query', '')
    top_k = data.get('top_k', 5)

    if not query.strip():
        return jsonify({'error': 'Query tidak boleh kosong'}), 400

    results = engine.search(query, top_k=top_k)
    return jsonify(results)


@app.route('/api/documents')
def get_documents():
    """API untuk mendapatkan semua dokumen."""
    docs = engine.get_all_documents()
    return jsonify(docs)


@app.route('/api/document/<int:doc_id>')
def get_document(doc_id):
    """API untuk mendapatkan detail satu dokumen."""
    detail = engine.get_document_detail(doc_id)
    if detail is None:
        return jsonify({'error': 'Dokumen tidak ditemukan'}), 404
    return jsonify(detail)


@app.route('/api/tfidf-matrix')
def get_tfidf_matrix():
    """API untuk mendapatkan matriks TF-IDF."""
    return jsonify({
        'tf_matrix': engine.get_tf_matrix(),
        'idf_values': engine.get_idf_values(),
        'tfidf_matrix': engine.get_tfidf_matrix()
    })


@app.route('/api/stats')
def get_stats():
    """API untuk mendapatkan statistik korpus."""
    return jsonify(engine.get_stats())


if __name__ == '__main__':
    print("=" * 60)
    print("  TF-IDF Information Retrieval System")
    print("  Buka browser: http://localhost:5000")
    print("=" * 60)
    app.run(debug=True, port=5000)
