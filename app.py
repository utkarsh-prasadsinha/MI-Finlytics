import os
import re
import io
import pandas as pd
import numpy as np

from flask import Flask, request, render_template_string, redirect, url_for
from sklearn.ensemble import IsolationForest
from prophet import Prophet
from transformers import (
    pipeline as hf_pipeline,
    AutoModelForCausalLM,
    AutoTokenizer
)

################################################################################
# FLASK APP SETUP
################################################################################
app = Flask(__name__)

################################################################################
# GLOBALS
################################################################################
csv_dataframes = {}    # {filename: DataFrame}
active_csv_key = None  # which CSV is "active"
chat_history = []      # multi-turn chat memory

################################################################################
# MODEL LOADING
################################################################################

# Summarizer for "GICS Sector" if present
summarizer = hf_pipeline("summarization", model="facebook/bart-large-cnn")

# Roberta QA pipeline (for direct question-answering)
print("Loading Roberta QA pipeline...")
try:
    qa_pipeline = hf_pipeline("question-answering", model="deepset/roberta-base-squad2")
    print("Roberta QA pipeline loaded successfully!")
except Exception as e:
    print("Error loading roberta-base-squad2:", e)
    qa_pipeline = None

# GPT-2 for text generation (small enough to avoid huge downloads)
print("Loading GPT-2 text-generation model...")
try:
    textgen_pipeline = hf_pipeline("text-generation", model="gpt2")
    print("GPT-2 text-generation pipeline loaded successfully!")
except Exception as e:
    print("Error loading GPT-2 model:", e)
    textgen_pipeline = None

################################################################################
# HELPER FUNCTIONS
################################################################################

def get_active_df():
    """Return the currently active DataFrame, or None if none is selected."""
    if active_csv_key and active_csv_key in csv_dataframes:
        return csv_dataframes[active_csv_key]
    return None

def analyze_numeric_query(user_msg):
    """
    If user asked "average <col>", we find that column in the active CSV 
    and compute the mean if numeric.
    Example: "average Age"
    """
    match = re.search(r"average\s+(\w+)", user_msg, re.IGNORECASE)
    if match:
        col = match.group(1)
        df = get_active_df()
        if df is None or df.empty:
            return "No active CSV or data is empty."
        if col not in df.columns:
            return f"Column '{col}' not found in the active CSV."
        # Check if numeric
        if not pd.api.types.is_numeric_dtype(df[col]):
            return f"Column '{col}' is not numeric."
        avg_val = df[col].mean()
        return f"The average of '{col}' is approximately {avg_val:.2f}."
    return None

def decide_model(question: str) -> str:
    """
    If question has typical QA words => 'qa', else => 'textgen'.
    """
    q_lc = question.lower()
    if any(w in q_lc for w in ["which", "what", "how many", "how much", "lowest", "highest", "?"]):
        return "qa"
    else:
        return "textgen"

def generate_stats_for_all_numeric(df):
    """
    Return .describe() for all numeric columns in df.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return None
    desc = df[numeric_cols].describe().round(2)
    return desc

def detect_anomalies(df, contamination=0.01):
    """
    For anomaly detection, pick the first numeric column if user hasn't specified.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return None
    col = numeric_cols[0]
    subdf = df[[col]].dropna()
    if subdf.empty:
        return None
    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(subdf)
    preds = model.predict(subdf)
    subdf["anomaly_label"] = preds
    anomalies = subdf[subdf["anomaly_label"] == -1]
    return anomalies

def correlation_matrix(df):
    """
    Return correlation among all numeric columns.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return None
    corr_mat = df[numeric_cols].corr().round(2)
    return corr_mat

def summarize_gics_sectors(df):
    """
    If 'GICS Sector' is present, summarize distribution.
    """
    if "GICS Sector" not in df.columns:
        return None
    sector_counts = df["GICS Sector"].value_counts()
    raw_text = "Sector distribution:\n"
    for sector, count in sector_counts.items():
        raw_text += f"{sector}: {count} companies\n"
    summary = summarizer(raw_text, max_length=50, min_length=10, do_sample=False)
    return summary[0]["summary_text"]

def forecast_any(df, periods=30):
    """
    We'll try to forecast a numeric column over a 'date' column if they exist.
    We'll guess the user might have 'date' and some numeric 'value' column named e.g. 'target'.
    Or skip if not found.
    """
    # Look for 'date' or 'Date' or 'DATE'
    date_cols = [c for c in df.columns if c.lower() == "date"]
    if not date_cols:
        return None
    date_col = date_cols[0]

    # Find any numeric column to forecast
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return None
    value_col = numeric_cols[0]

    subdf = df[[date_col, value_col]].dropna()
    subdf.rename(columns={date_col: "ds", value_col: "y"}, inplace=True)
    subdf["ds"] = pd.to_datetime(subdf["ds"], errors="coerce")
    subdf.dropna(subset=["ds", "y"], inplace=True)
    if subdf.empty:
        return None

    model = Prophet(daily_seasonality=True)
    model.fit(subdf)
    future = model.make_future_dataframe(periods=periods)
    fc = model.predict(future)
    return fc[["ds", "yhat", "yhat_lower", "yhat_upper"]]

################################################################################
# HTML TEMPLATES
################################################################################

HOME_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Generic CSV AI Dashboard</title>
</head>
<body>
    <h1>Generic CSV AI Dashboard</h1>
    <p>Upload CSV(s), select an active CSV, then do stats, anomalies, correlation, forecasting, or chat!</p>
    <form action="/upload" method="post" enctype="multipart/form-data">
        <label>Upload one or more CSV files:</label><br/>
        <input type="file" name="files" multiple>
        <button type="submit">Upload</button>
    </form>
    <br/>
    <form action="/select" method="get">
        <button type="submit">Select Active CSV</button>
    </form>
    <hr/>
    <p>Current active CSV: {{active_csv}}</p>
    <form action="/stats" method="get">
        <button type="submit">Descriptive Stats (All Numeric)</button>
    </form>
    <form action="/anomalies" method="get">
        <button type="submit">Anomaly Detection (First Numeric Col)</button>
    </form>
    <form action="/corr" method="get">
        <button type="submit">Correlation Matrix (All Numeric)</button>
    </form>
    <form action="/forecast" method="get">
        <label>Days to Forecast:</label>
        <input type="number" name="periods" value="30" min="1"/>
        <button type="submit">Forecast (Date + Numeric col)</button>
    </form>
    <form action="/gics" method="get">
        <button type="submit">Summarize GICS Sectors</button>
    </form>
    <hr/>
    <form action="/chat" method="get">
        <button type="submit">Chat with AI</button>
    </form>
</body>
</html>
"""

RESULT_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Result</title>
</head>
<body>
    <h1>Result</h1>
    <div>
        {{ content|safe }}
    </div>
    <br/>
    <a href="/">Back to Home</a>
</body>
</html>
"""

SELECT_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Select Active CSV</title>
</head>
<body>
    <h1>Select which CSV is active</h1>
    {% if csv_keys %}
        <form action="/select" method="post">
            <label>Choose CSV:</label>
            <select name="csv_key">
                {% for key in csv_keys %}
                    <option value="{{key}}">{{key}}</option>
                {% endfor %}
            </select>
            <button type="submit">Set Active</button>
        </form>
    {% else %}
        <p>No CSVs uploaded yet.</p>
    {% endif %}
    <br/>
    <a href="/">Back to Home</a>
</body>
</html>
"""

CHAT_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Chat with AI</title>
</head>
<body>
    <h1>Multi-turn Chat about the Active CSV</h1>
    <p>Active CSV: {{active_csv}}</p>
    <div style="border:1px solid #ccc; padding:10px; width:600px; height:250px; overflow:auto;">
        {% for msg in messages %}
            <p><b>{{ msg.role }}:</b> {{ msg.content }}</p>
        {% endfor %}
    </div>
    <form method="post">
        <label>Your Question:</label><br/>
        <input type="text" name="question" style="width:400px;" required><br/><br/>
        <button type="submit">Ask</button>
    </form>
    <br/>
    <a href="/">Back to Home</a>
</body>
</html>
"""

################################################################################
# FLASK ROUTES
################################################################################

app = Flask(__name__)

@app.route("/")
def home():
    return render_template_string(HOME_TEMPLATE, active_csv=active_csv_key or "None")

@app.route("/upload", methods=["POST"])
def upload():
    """
    Upload one or multiple CSV files. Store them in csv_dataframes with key=filename.
    """
    files = request.files.getlist("files")
    if not files:
        return render_template_string(RESULT_TEMPLATE, content="<p>No files selected.</p>")

    msgs = []
    for f in files:
        filename = f.filename
        if not filename.endswith(".csv"):
            msgs.append(f"Skipping non-CSV file: {filename}")
            continue
        try:
            content = f.read()
            df = pd.read_csv(io.BytesIO(content))
            csv_dataframes[filename] = df
            msgs.append(f"Uploaded CSV: {filename} (rows={len(df)}, cols={list(df.columns)})")
        except Exception as e:
            msgs.append(f"Error reading {filename}: {e}")

    msg_html = "<br/>".join(msgs)
    return render_template_string(RESULT_TEMPLATE, content=msg_html)

@app.route("/select", methods=["GET", "POST"])
def select_csv():
    global active_csv_key
    if request.method == "GET":
        csv_keys = list(csv_dataframes.keys())
        return render_template_string(SELECT_TEMPLATE, csv_keys=csv_keys)
    else:
        chosen = request.form.get("csv_key")
        if chosen in csv_dataframes:
            active_csv_key = chosen
            return redirect(url_for("home"))
        else:
            return render_template_string(RESULT_TEMPLATE, content=f"<p>Invalid CSV key: {chosen}</p>")

@app.route("/stats")
def stats_route():
    df = get_active_df()
    if df is None or df.empty:
        return render_template_string(RESULT_TEMPLATE, content="<p>No active CSV or data is empty.</p>")
    desc = generate_stats_for_all_numeric(df)
    if desc is None:
        return render_template_string(RESULT_TEMPLATE, content="<p>No numeric columns found.</p>")
    html_table = desc.to_html(classes="table table-striped", justify="center")
    return render_template_string(RESULT_TEMPLATE, content=html_table)

@app.route("/anomalies")
def anomalies_route():
    df = get_active_df()
    if df is None or df.empty:
        return render_template_string(RESULT_TEMPLATE, content="<p>No active CSV or data is empty.</p>")

    anomalies = detect_anomalies(df, contamination=0.01)
    if anomalies is None or anomalies.empty:
        return render_template_string(RESULT_TEMPLATE, content="<p>No anomalies found or no numeric data.</p>")

    top_anomalies = anomalies.head(20)
    html_table = top_anomalies.to_html(classes="table table-striped", justify="center")
    result_msg = f"<p><b>Total anomalies:</b> {len(anomalies)}</p>" + html_table
    return render_template_string(RESULT_TEMPLATE, content=result_msg)

@app.route("/corr")
def corr_route():
    df = get_active_df()
    if df is None or df.empty:
        return render_template_string(RESULT_TEMPLATE, content="<p>No active CSV or data is empty.</p>")
    corr_mat = correlation_matrix(df)
    if corr_mat is None or corr_mat.empty:
        return render_template_string(RESULT_TEMPLATE, content="<p>No numeric columns to correlate.</p>")
    html_table = corr_mat.to_html(classes="table table-striped", justify="center")
    return render_template_string(RESULT_TEMPLATE, content=html_table)

@app.route("/forecast")
def forecast_route():
    df = get_active_df()
    if df is None or df.empty:
        return render_template_string(RESULT_TEMPLATE, content="<p>No active CSV or data is empty.</p>")

    periods_str = request.args.get("periods", "30")
    try:
        periods = int(periods_str)
    except ValueError:
        periods = 30

    fc = forecast_any(df, periods=periods)
    if fc is None or fc.empty:
        msg = "Unable to forecast. We need a 'date' column + numeric column."
        return render_template_string(RESULT_TEMPLATE, content=msg)

    top_fc = fc.head(20).round(2)
    html_table = top_fc.to_html(classes="table table-striped", justify="center")
    content = f"<p>Forecast (showing first 20 rows):</p>{html_table}"
    return render_template_string(RESULT_TEMPLATE, content=content)

@app.route("/gics")
def gics_route():
    df = get_active_df()
    if df is None or df.empty:
        return render_template_string(RESULT_TEMPLATE, content="<p>No active CSV or data is empty.</p>")
    summary = summarize_gics_sectors(df)
    if summary is None:
        return render_template_string(RESULT_TEMPLATE, content="<p>No 'GICS Sector' column found.</p>")
    return render_template_string(RESULT_TEMPLATE, content=summary)

@app.route("/chat", methods=["GET", "POST"])
def chat():
    global chat_history

    if request.method == "GET":
        return render_template_string(CHAT_TEMPLATE, messages=chat_history, active_csv=active_csv_key or "None")
    else:
        user_question = request.form.get("question", "").strip()
        if not user_question:
            return render_template_string(CHAT_TEMPLATE, messages=chat_history, active_csv=active_csv_key or "None")

        # 1) Append user message
        chat_history.append({"role": "user", "content": user_question})

        # 2) Check numeric query
        numeric_ans = analyze_numeric_query(user_question)
        if numeric_ans:
            ai_reply = numeric_ans
        else:
            # 3) Decide model
            model_choice = decide_model(user_question)
            if model_choice == "qa":
                if qa_pipeline is None:
                    ai_reply = "Roberta QA model not loaded."
                else:
                    df = get_active_df()
                    columns_str = "None" if df is None or df.empty else str(df.columns.tolist())
                    conversation_text = (
                        f"Active CSV: '{active_csv_key}' with columns: {columns_str}\n\n"
                        "Conversation so far:\n"
                    )
                    for msg in chat_history:
                        conversation_text += f"{msg['role'].upper()}: {msg['content']}\n"
                    try:
                        result = qa_pipeline(question=user_question, context=conversation_text)
                        ai_reply = result["answer"]
                    except Exception as ex:
                        ai_reply = f"Error in QA pipeline: {ex}"
            else:
                if textgen_pipeline is None:
                    ai_reply = "GPT-2 model not loaded."
                else:
                    df = get_active_df()
                    columns_str = "None" if df is None or df.empty else str(df.columns.tolist())
                    prompt = (
                        f"Active CSV: '{active_csv_key}' with columns: {columns_str}.\n"
                        "This is a conversation about that data.\n"
                    )
                    for msg in chat_history:
                        prompt += f"{msg['role'].upper()}: {msg['content']}\n"
                    prompt += "ASSISTANT:"
                    try:
                        response = textgen_pipeline(prompt, max_new_tokens=200, do_sample=True)
                        full_output = response[0]["generated_text"]
                        splitted = full_output.split("ASSISTANT:")
                        if len(splitted) > 1:
                            ai_reply = splitted[-1].strip()
                        else:
                            ai_reply = full_output
                    except Exception as ex:
                        ai_reply = f"Error generating text: {ex}"

        # 4) Append AI reply
        chat_history.append({"role": "assistant", "content": ai_reply})

        return render_template_string(CHAT_TEMPLATE, messages=chat_history, active_csv=active_csv_key or "None")

################################################################################
# MAIN
################################################################################

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
