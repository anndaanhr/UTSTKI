// =============================================================================
// SearchLab — TF-IDF IR System Frontend
// =============================================================================

document.addEventListener('DOMContentLoaded', () => {
    loadStats();
    loadDocuments();
    setupTabs();
    setupSearch();
    setupModal();
    setupMatrixTabs();
});

// ─── Tab Navigation ───
function setupTabs() {
    document.querySelectorAll('.nav-item').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.nav-item').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById(`content-${btn.dataset.tab}`).classList.add('active');
            if (btn.dataset.tab === 'matrix') loadMatrix('tfidf');
        });
    });
}

function setupMatrixTabs() {
    document.querySelectorAll('.msw-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.msw-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            loadMatrix(btn.dataset.matrix);
        });
    });
}

// ─── Stats ───
async function loadStats() {
    try {
        const r = await fetch('/api/stats');
        const d = await r.json();
        animateNum('stat-docs', d.total_documents);
        animateNum('stat-tokens', d.total_tokens);
        animateNum('stat-vocab', d.unique_terms);
        document.getElementById('stat-avg').textContent = d.avg_doc_length;
    } catch (e) { console.error(e); }
}

function animateNum(id, target) {
    const el = document.getElementById(id);
    const duration = 600;
    const start = performance.now();
    const from = 0;
    function tick(now) {
        const t = Math.min((now - start) / duration, 1);
        const ease = 1 - Math.pow(1 - t, 3);
        el.textContent = Math.round(from + (target - from) * ease).toLocaleString();
        if (t < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
}

// ─── Search ───
function setupSearch() {
    document.getElementById('btn-search').addEventListener('click', doSearch);
    document.getElementById('search-input').addEventListener('keydown', e => {
        if (e.key === 'Enter') doSearch();
    });
}

function quickSearch(q) {
    document.getElementById('search-input').value = q;
    doSearch();
}

async function doSearch() {
    const query = document.getElementById('search-input').value.trim();
    const topK = parseInt(document.getElementById('topk-select').value);
    if (!query) return;

    document.getElementById('empty-state').style.display = 'none';
    const area = document.getElementById('results-area');
    area.style.display = 'block';
    document.getElementById('results-manual').innerHTML = loading();
    document.getElementById('results-sklearn').innerHTML = loading();

    try {
        const res = await fetch('/api/search', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, top_k: topK })
        });
        const data = await res.json();

        document.getElementById('results-title').textContent = `"${query}"`;
        document.getElementById('query-tokens-badge').textContent =
            `tokens → [ ${data.query_tokens.join(' · ')} ]`;

        renderResults('results-manual', data.manual, 'manual');
        renderResults('results-sklearn', data.sklearn, 'sklearn');
    } catch (e) {
        console.error(e);
        document.getElementById('results-manual').innerHTML = errMsg();
        document.getElementById('results-sklearn').innerHTML = errMsg();
    }
}

function renderResults(id, results, method) {
    const el = document.getElementById(id);
    if (!results || results.length === 0) {
        el.innerHTML = '<div class="no-results">Tidak ada dokumen relevan</div>';
        return;
    }
    const max = results[0].score;
    el.innerHTML = results.map((r, i) => {
        const pct = max > 0 ? (r.score / max * 100) : 0;
        return `
        <div class="r-card ${method}-r" onclick="showDoc(${r.id})">
            <div class="r-rank">${i + 1}</div>
            <div class="r-body">
                <div class="r-title">${r.title}</div>
                <div class="r-snippet">${r.content.substring(0, 140)}…</div>
                <div class="r-footer">
                    <div class="r-score-bar"><div class="r-score-fill" style="width:${pct}%"></div></div>
                    <span class="r-score-val">${r.score.toFixed(4)}</span>
                    <span class="r-docid">DOC-${String(r.id).padStart(2,'0')}</span>
                </div>
            </div>
        </div>`;
    }).join('');
}

// ─── Documents ───
async function loadDocuments() {
    try {
        const res = await fetch('/api/documents');
        const docs = await res.json();
        document.getElementById('documents-grid').innerHTML = docs.map(d => `
            <div class="d-card" onclick="showDoc(${d.id})">
                <div class="d-num">${String(d.id).padStart(2, '0')}</div>
                <div class="d-body">
                    <div class="d-title">${d.title}</div>
                    <div class="d-snippet">${d.content}</div>
                </div>
                <div class="d-meta">
                    <span>${d.word_count} kata</span>
                    <span>${d.token_count} tokens</span>
                    <span>${d.unique_terms} unik</span>
                </div>
            </div>
        `).join('');
    } catch (e) { console.error(e); }
}

// ─── Modal ───
function setupModal() {
    document.getElementById('modal-close').addEventListener('click', closeModal);
    document.getElementById('modal-overlay').addEventListener('click', e => {
        if (e.target === e.currentTarget) closeModal();
    });
    document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });
}

async function showDoc(docId) {
    const overlay = document.getElementById('modal-overlay');
    const body = document.getElementById('modal-body');
    overlay.classList.add('show');
    body.innerHTML = `<div class="loader"><div class="loader-bar"></div></div>`;

    try {
        const res = await fetch(`/api/document/${docId}`);
        const d = await res.json();
        const terms = d.term_details.slice(0, 20);
        body.innerHTML = `
            <div class="modal-label">DOKUMEN #${String(d.id).padStart(2,'0')}</div>
            <div class="modal-doctitle">${d.title}</div>
            <div class="modal-text">${d.content}</div>
            <div class="modal-stats">
                <div class="modal-stat-box"><div class="modal-stat-v">${d.total_tokens}</div><div class="modal-stat-l">Tokens</div></div>
                <div class="modal-stat-box"><div class="modal-stat-v">${d.unique_terms}</div><div class="modal-stat-l">Unique Terms</div></div>
                <div class="modal-stat-box"><div class="modal-stat-v">${terms[0]?.tfidf.toFixed(4) || '—'}</div><div class="modal-stat-l">Max TF-IDF</div></div>
            </div>
            <div class="modal-section-title">Top 20 Terms by TF-IDF</div>
            <table class="term-table">
                <thead><tr><th>#</th><th>TERM</th><th>TF</th><th>IDF</th><th>TF-IDF</th></tr></thead>
                <tbody>${terms.map((t, i) => `
                    <tr>
                        <td>${i + 1}</td>
                        <td class="term-name">${t.term}</td>
                        <td>${t.tf.toFixed(4)}</td>
                        <td>${t.idf.toFixed(4)}</td>
                        <td class="term-highlight">${t.tfidf.toFixed(6)}</td>
                    </tr>`).join('')}
                </tbody>
            </table>`;
    } catch (e) { body.innerHTML = errMsg(); }
}

function closeModal() { document.getElementById('modal-overlay').classList.remove('show'); }

// ─── Matrix ───
let mCache = null;
async function loadMatrix(type) {
    const box = document.getElementById('matrix-container');
    box.innerHTML = `<div class="loader"><div class="loader-bar"></div></div>`;

    try {
        if (!mCache) {
            const r = await fetch('/api/tfidf-matrix');
            mCache = await r.json();
        }
        if (type === 'idf') renderIdf(mCache.idf_values);
        else renderMatrix(type === 'tf' ? mCache.tf_matrix : mCache.tfidf_matrix);
    } catch (e) { box.innerHTML = errMsg(); }
}

function renderMatrix(data) {
    const box = document.getElementById('matrix-container');
    const docs = Object.keys(data);
    const termSet = new Set();
    docs.forEach(d => Object.keys(data[d]).forEach(t => { if (data[d][t] > 0) termSet.add(t); }));
    const terms = [...termSet].slice(0, 25);

    let mx = 0;
    docs.forEach(d => terms.forEach(t => { const v = data[d][t] || 0; if (v > mx) mx = v; }));

    let h = '<table class="m-table"><thead><tr><th>DOC</th>';
    terms.forEach(t => h += `<th>${t}</th>`);
    h += '</tr></thead><tbody>';
    docs.forEach(d => {
        h += `<tr><td class="rh">${d}</td>`;
        terms.forEach(t => {
            const v = data[d][t] || 0;
            const lvl = mx > 0 ? Math.min(Math.floor((v / mx) * 4), 4) : 0;
            h += `<td class="heat-${lvl}">${v > 0 ? v.toFixed(4) : '·'}</td>`;
        });
        h += '</tr>';
    });
    h += '</tbody></table>';
    box.innerHTML = h;
}

function renderIdf(data) {
    const box = document.getElementById('matrix-container');
    let h = '<table class="m-table" style="max-width:560px"><thead><tr><th>#</th><th>TERM</th><th>IDF</th></tr></thead><tbody>';
    Object.entries(data).forEach(([term, val], i) => {
        const lvl = val > 1.0 ? 4 : val > 0.7 ? 3 : val > 0.4 ? 2 : val > 0.1 ? 1 : 0;
        h += `<tr><td>${i + 1}</td><td class="rh" style="position:static">${term}</td><td class="heat-${lvl}">${val.toFixed(4)}</td></tr>`;
    });
    h += '</tbody></table>';
    box.innerHTML = h;
}

// ─── Helpers ───
function loading() { return '<div class="loader"><div class="loader-bar"></div></div>'; }
function errMsg() { return '<div class="no-results">Terjadi kesalahan</div>'; }
