from flask import Flask, render_template, jsonify, request
from tfidf_engine import TFIDFSearchEngine

app = Flask(__name__)
engine = TFIDFSearchEngine()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/search', methods=['POST'])
def search():
    data = request.get_json()
    query = data.get('query', '')
    top_k = data.get('top_k', 5)

    if not query.strip():
        return jsonify({'error': 'Query tidak boleh kosong'}), 400

    results = engine.search(query, top_k=top_k)
    return jsonify(results)

@app.route('/api/documents')
def get_documents():
    docs = engine.get_all_documents()
    return jsonify(docs)

@app.route('/api/document/<int:doc_id>')
def get_document(doc_id):
    detail = engine.get_document_detail(doc_id)
    if detail is None:
        return jsonify({'error': 'Dokumen tidak ditemukan'}), 404
    return jsonify(detail)

@app.route('/api/tfidf-matrix')
def get_tfidf_matrix():
    return jsonify({
        'tf_matrix': engine.get_tf_matrix(),
        'idf_values': engine.get_idf_values(),
        'tfidf_matrix': engine.get_tfidf_matrix()
    })

@app.route('/api/stats')
def get_stats():
    return jsonify(engine.get_stats())

if __name__ == '__main__':
    app.run(debug=True, port=5000)
