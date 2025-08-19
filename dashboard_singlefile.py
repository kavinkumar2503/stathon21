import requests

# === OpenAI API Config ===
OPENAI_API_KEY = "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"  # <-- User's real OpenAI API key
OPENAI_MODEL = "gpt-3.5-turbo"  # or "gpt-4"
import os
import io
import json
import pandas as pd
import plotly.express as px
import plotly.io as pio
from flask import Flask, request, session, send_file, redirect, url_for, jsonify
from datetime import datetime
from sklearn.impute import KNNImputer

app = Flask(__name__)
app.secret_key = "your_secret_key"
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Inline CSS and JS
INLINE_CSS = '''
body { font-family: Arial, sans-serif; margin: 40px; background: #f6f9ff; min-height: 100vh; }
.dashboard-card { background: #fff; border-radius: 16px; box-shadow: 0 4px 24px rgba(0,0,0,0.08); padding: 2rem; margin-top: 2rem; }
.logo { width: 80px; margin-bottom: 10px; }
h1 { font-weight: 700; color: #2a5298; }
.table-bordered th, .table-bordered td { background: #f8fafc; }
#charts { background: #f6f9ff; border-radius: 12px; padding: 1rem; margin-bottom: 1rem; box-shadow: 0 2px 8px rgba(0,0,0,0.04); }
#ai-chat-toggle { position: fixed; right: 20px; bottom: 20px; z-index: 1001; }
#ai-chat-widget { position: fixed; right: 20px; bottom: 80px; width: 320px; max-height: 60vh; background: #fff; border: 1px solid #ddd; border-radius: 8px; box-shadow: 0 6px 24px rgba(0,0,0,.15); display: none; flex-direction: column; overflow: hidden; z-index: 1000; }
#ai-chat-header { background: #0d6efd; color: #fff; padding: 10px 12px; display: flex; justify-content: space-between; align-items: center; }
#ai-chat-messages { padding: 10px; overflow-y: auto; height: 280px; font-size: 0.95rem; }
.msg { margin: 6px 0; }
.msg.user { text-align: right; }
.msg.bot { text-align: left; }
#ai-chat-input { display: flex; gap: 6px; padding: 10px; border-top: 1px solid #eee; }
#ai-chat-input input { flex: 1; }
'''
INLINE_JS = '''
const toggleBtn = document.getElementById('ai-chat-toggle');
const widget = document.getElementById('ai-chat-widget');
const closeBtn = document.getElementById('ai-chat-close');
const sendBtn = document.getElementById('ai-chat-send');
const input = document.getElementById('ai-chat-text');
const messages = document.getElementById('ai-chat-messages');
function appendMsg(text, who) {
  const div = document.createElement('div');
  div.className = 'msg ' + who;
  div.textContent = text;
  messages.appendChild(div);
  messages.scrollTop = messages.scrollHeight;
}
function send() {
  const text = input.value.trim();
  if (!text) return;
  appendMsg(text, 'user');
  input.value = '';
  fetch('/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message: text })
  })
  .then(r => r.json())
  .then(j => appendMsg(j.reply || j.error || '(no reply)', 'bot'))
  .catch(e => appendMsg('Error: ' + e, 'bot'));
}
toggleBtn.onclick = () => { widget.style.display = widget.style.display === 'flex' ? 'none' : 'flex'; widget.style.display === 'flex' && input.focus(); };
closeBtn.onclick = () => { widget.style.display = 'none'; };
sendBtn.onclick = send;
input.addEventListener('keydown', (e) => { if (e.key === 'Enter') send(); });
'''

# Helper: render the main dashboard HTML
def render_dashboard(summary_html='', plot_div='', cleaned_file=None, login_message=None, logged_in=False, ai_enabled=True):
        html = f"""
<!DOCTYPE html>
<html><head>
<title>Survey Data Dashboard</title>
<style>{INLINE_CSS}</style>
<link rel='stylesheet' href='https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css'>
</head><body>
<div class='container'>
    <div class='dashboard-card'>
        <div class='text-center'>
            <img src='https://cdn-icons-png.flaticon.com/512/3135/3135715.png' class='logo' alt='Logo'>
            <h1 class='mb-4'>Survey Data Dashboard</h1>
        </div>
        {('<div class="alert alert-info">' + str(login_message) + '</div>') if login_message else ''}
        {'' if logged_in else """
        <form method='POST' action='/login' class='mb-4'>
            <button type='submit' class='btn btn-primary w-100'>Login</button>
        </form>"""}
        {'' if not logged_in else """
        <a href='/logout' class='btn btn-secondary mb-4'>Logout</a>
        <form method='POST' enctype='multipart/form-data' class='mb-4'>
            <label class='form-label'>Select a survey file (.csv or .xlsx):</label>
            <input type='file' name='survey_file' accept='.csv,.xlsx' class='form-control mb-2' required>
            <label class='form-label'>Schema Mapping (JSON):</label>
            <input type='file' name='schema_file' accept='.json' class='form-control mb-2'>
            <label class='form-label'>Imputation Method:</label>
            <select name='impute_method' class='form-control mb-2'>
                <option value='median'>Median</option>
                <option value='mean'>Mean</option>
                <option value='knn'>KNN</option>
            </select>
            <label class='form-label'>Outlier Detection:</label>
            <select name='outlier_method' class='form-control mb-2'>
                <option value='none'>None</option>
                <option value='iqr'>IQR</option>
                <option value='zscore'>Z-Score</option>
                <option value='winsor'>Winsorize</option>
            </select>
            <label class='form-label'>Design Weights Column (optional):</label>
            <input type='text' name='weight_column' class='form-control mb-2' placeholder='Column name for weights'>
            <label class='form-label'>Rule-based Validation (JSON):</label>
            <input type='file' name='rules_file' accept='.json' class='form-control mb-2'>
            <button type='submit' class='btn btn-success w-100'>Upload & Process</button>
        </form>"""}
        {(f'<h2 class="mt-4">Summary of Processed Data</h2><div class="table-responsive">{summary_html}</div>') if summary_html else ''}
        {(f'<h2 class="mt-4">Charts</h2><div id="charts">{plot_div}</div>') if plot_div else ''}
        {(f'<a href="/download/{cleaned_file}" class="btn btn-info mt-3 w-100">Download Cleaned File</a>') if cleaned_file else ''}
    </div>
</div>
{'' if not ai_enabled else """
<div id='ai-chat-widget'>
    <div id='ai-chat-header'>
        <div>AI Assistant</div>
        <button class='btn btn-sm btn-light' id='ai-chat-close'>×</button>
    </div>
    <div id='ai-chat-messages'></div>
    <div id='ai-chat-input'>
        <input type='text' id='ai-chat-text' class='form-control' placeholder='Ask something...' />
        <button id='ai-chat-send' class='btn btn-primary'>Send</button>
    </div>
</div>
<button id='ai-chat-toggle' class='btn btn-primary'>Ask AI</button>
<script>{INLINE_JS}</script>
"""}
</body></html>
"""
        return html

@app.route('/', methods=['GET', 'POST'])
def index():
    summary_html = ''
    plot_div = ''
    cleaned_file = None
    login_message = None
    logged_in = session.get('logged_in', False)
    ai_enabled = True
    if request.method == 'POST' and logged_in:
        file = request.files['survey_file']
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)
        # Schema mapping
        schema_file = request.files.get('schema_file')
        if schema_file and schema_file.filename:
            schema = json.load(schema_file)
            df.rename(columns=schema, inplace=True)
            df.columns = [col if not isinstance(col, dict) else str(col) for col in df.columns]
        # Outlier detection
        outlier_method = request.form.get('outlier_method', 'none')
        df = df.loc[:, [col for col in df.columns if not isinstance(col, dict)]]
        numeric_cols = df.select_dtypes(include=[float, int]).columns
        if outlier_method == 'iqr':
            Q1 = df[numeric_cols].quantile(0.25)
            Q3 = df[numeric_cols].quantile(0.75)
            IQR = Q3 - Q1
            mask = ~((df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
            df = df[mask]
        elif outlier_method == 'zscore':
            z_scores = abs((df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std())
            mask = (z_scores < 3).all(axis=1)
            df = df[mask]
        elif outlier_method == 'winsor':
            for col in numeric_cols:
                df[col] = df[col].clip(lower=df[col].quantile(0.05), upper=df[col].quantile(0.95))
        # Drop columns with >40% missing values
        threshold = 0.4
        missing_fraction = df.isnull().mean()
        cols_to_drop = missing_fraction[missing_fraction > threshold].index
        df.drop(columns=cols_to_drop, inplace=True)
        # Imputation
        impute_method = request.form.get('impute_method', 'median')
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                if impute_method == 'mean':
                    df[col] = df[col].fillna(df[col].mean())
                elif impute_method == 'median':
                    df[col] = df[col].fillna(df[col].median())
                elif impute_method == 'knn':
                    imputer = KNNImputer(n_neighbors=3)
                    df[[col]] = imputer.fit_transform(df[[col]])
            else:
                if not df[col].mode().empty:
                    df[col] = df[col].fillna(df[col].mode()[0])
        # Rule-based validation
        rules_file = request.files.get('rules_file')
        rule_warnings = []
        if rules_file and rules_file.filename:
            rules = json.load(rules_file)
            for col, rule in rules.items():
                if col not in df.columns:
                    rule_warnings.append(f"Warning: Column '{col}' not found in data, rule skipped.")
                    continue
                if 'min' in rule:
                    invalid = df[df[col] < rule['min']]
                    if not invalid.empty:
                        rule_warnings.append(f"{col}: {len(invalid)} values below {rule['min']}")
                if 'max' in rule:
                    invalid = df[df[col] > rule['max']]
                    if not invalid.empty:
                        rule_warnings.append(f"{col}: {len(invalid)} values above {rule['max']}")
        # Weight application
        weight_column = request.form.get('weight_column')
        weighted_mean = None
        if weight_column and weight_column in df.columns:
            df[weight_column] = pd.to_numeric(df[weight_column], errors='coerce')
            valid_weights = df[weight_column].notnull()
            weights = df.loc[valid_weights, weight_column]
            numeric_cols = df.select_dtypes(include='number').columns.drop(weight_column, errors='ignore')
            data_for_weight = df.loc[valid_weights, numeric_cols]
            denom = weights.sum()
            if pd.notnull(denom) and denom != 0:
                weighted_mean = (data_for_weight.multiply(weights, axis=0).sum() / denom)
            else:
                weighted_mean = None
        cleaned_filename = "cleaned_" + file.filename
        cleaned_filepath = os.path.join(UPLOAD_FOLDER, cleaned_filename)
        df.to_csv(cleaned_filepath, index=False)
        cleaned_file = cleaned_filename
        summary_html = df.describe().to_html(classes="table table-bordered")
        if weighted_mean is not None:
            summary_html += "<br><b>Weighted Means:</b><br>" + weighted_mean.to_frame("Weighted Mean").to_html(classes="table table-bordered")
        if rule_warnings:
            summary_html += "<br><b>Rule Warnings:</b><ul>" + "".join(f"<li>{w}</li>" for w in rule_warnings) + "</ul>"
        if not df.empty:
            fig = px.bar(df.describe().transpose(), title="Summary Bar Chart")
            plot_div = pio.to_html(fig, full_html=False)
    return render_dashboard(summary_html, plot_div, cleaned_file, login_message, logged_in, ai_enabled)

@app.route('/login', methods=['POST'])
def login():
    session['logged_in'] = True
    session['login_message'] = 'Login successful!'
    return redirect(url_for('index'))

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session['login_message'] = 'Logged out.'
    return redirect(url_for('index'))

@app.route('/download/<filename>')
def download_file(filename):
    path = os.path.join(UPLOAD_FOLDER, filename)
    return send_file(path, as_attachment=True)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json(silent=True) or {}
    user_msg = (data.get('message') or '').strip()
    print("[DEBUG] /chat endpoint called. user_msg:", user_msg)
    if not user_msg:
        print("[DEBUG] No message provided.")
        return jsonify({'error': 'message is required'}), 400
    # --- OpenAI API call ---
    try:
        payload = {
            "model": OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant for the Survey Data Dashboard."},
                {"role": "user", "content": user_msg},
            ],
            "temperature": 0.3,
            "max_tokens": 300,
        }
        print("[DEBUG] Sending request to OpenAI API...")
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=20,
        )
        print(f"[DEBUG] OpenAI response status: {resp.status_code}")
        print(f"[DEBUG] OpenAI response text: {resp.text[:300]}")
        if resp.status_code == 200:
            j = resp.json()
            reply = j.get("choices", [{}])[0].get("message", {}).get("content", "")
            print("[DEBUG] AI reply:", reply)
            return jsonify({"reply": reply or "(no content)"})
        if resp.status_code in (401, 403):
            print("[DEBUG] AI auth error.")
            return jsonify({"reply": "AI auth error: Invalid API key or model. Check your OpenAI API key and model name."})
        if resp.status_code == 429:
            print("[DEBUG] AI quota/rate limit error.")
            return jsonify({"reply": "AI quota or rate limit exceeded. Please check your OpenAI plan or try again later."})
        print(f"[DEBUG] AI error: {resp.status_code} {resp.text[:200]}")
        return jsonify({"reply": f"AI error: {resp.status_code} {resp.text[:200]}"})
    except Exception as e:
        print(f"[DEBUG] Exception in /chat: {e}")
        return jsonify({"reply": f"AI backend error: {e}"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
