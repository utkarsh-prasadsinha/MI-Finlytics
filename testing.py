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
    AutoConfig,
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
# Dictionary mapping {filename: DataFrame}
csv_dataframes = {}
active_csv_key = None  # The filename that is currently "active"

# Chat memory
chat_history = []

################################################################################
# MODEL LOADING: ROBERTA QA + DEEPSEEK TEXT GEN
################################################################################

# 1) Summarizer for GICS Sectors
summarizer = hf_pipeline("summarization", model="facebook/bart-large-cnn")

# 2) Roberta QA pipeline
print("Loading Roberta QA pipeline...")
try:
    roberta_qa_pipeline = hf_pipeline(
        "question-answering",
        model="deepset/roberta-base-squad2"
    )
    print("Roberta QA pipeline loaded successfully!")
except Exception as e:
    print("Error loading roberta-base-squad2:", e)
    roberta_qa_pipeline = None

# 3) DeepSeek text-generation pipeline
print("Loading DeepSeek-R1 model/config...")

try:
    deepseek_config = AutoConfig.from_pretrained(
        "deepseek-ai/DeepSeek-R1",
        trust_remote_code=True
    )
    # If there's a quantization config referencing fp8, we try to override it
    if hasattr(deepseek_config, "quantization_config"):
        qc = deepseek_config.quantization_config
        print("Original quantization_config:", qc)
        # Force a fallback if 'quant_method' is missing or set to 'fp8'
        if "quant_method" not in qc or qc["quant_method"] == "fp8":
            qc["quant_method"] = "none"
        # If you prefer a recognized method, e.g. 'bitsandbytes_8bit', do:
        # qc["quant_method"] = "bitsandbytes_8bit"
        print("Updated quantization_config:", deepseek_config.quantization_config)

    deepseek_model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/DeepSeek-R1",
        config=deepseek_config,
        trust_remote_code=True
    )
    deepseek_tokenizer = AutoTokenizer.from_pretrained(
        "deepseek-ai/DeepSeek-R1",
        trust_remote_code=True
    )

    deepseek_pipeline = hf_pipeline(
        "text-generation",
        model=deepseek_model,
        tokenizer=deepseek_tokenizer,
        trust_remote_code=True
    )
    print("DeepSeek pipeline loaded successfully!")
except Exception as e:
    print("Error loading DeepSeek model:", e)
    deepseek_pipeline = None

################################################################################
# HELPER FUNCTIONS
################################################################################

def get_active_df():
    """
    Returns the currently active DataFrame, or None if none selected.
    """
    if active_csv_key and active_csv_key in csv_dataframes:
        return csv_dataframes[active_csv_key]
    return None

def analyze_numeric_query(user_msg):
    """
    If user asked "average close price for <SYMBOL>", 
    we look for 'symbol' and 'close' in the active CSV.
    """
    match = re.search(r"average close price for\s+(\w+)", user_msg, re.IGNORECASE)
    if match:
        symbol = match.group(1).upper()
        df = get_active_df()
        if df is None or df.empty:
            return "No active CSV or data is empty."
        if "symbol" not in df.columns:
            return "Error: 'symbol' column not found in the active CSV."
        sub = df[df["symbol"].str.upper() == symbol]
        if sub.empty:
            return f"I couldn't find data for symbol '{symbol}'."
        avg_close = sub["close"].mean()
        return f"The average close price for {symbol} is approximately {avg_close:.2f}."
    return None

def decide_model(question: str) -> str:
    """
    If question has typical QA words => 'qa', else => 'textgen'
    """
    question_lc = question.lower()
    if any(w in question_lc for w in ["which", "what", "how many", "how much", "lowest", "highest", "?"]):
        return "qa"
    else:
        return "textgen"

def detect_anomalies(df, contamination=0.01):
    """
    If 'close' in columns, use it. Else pick the first numeric col.
    """
    if df.empty:
        return pd.DataFrame()

    if "close" in df.columns:
        col = "close"
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return pd.DataFrame()
        col = numeric_cols[0]

    subdf = df[[col]].dropna()
    if subdf.empty:
        return pd.DataFrame()

    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(subdf)
    preds = model.predict(subdf)
    subdf["anomaly_label"] = preds
    anomalies = subdf[subdf["anomaly_label"] == -1]
    return anomalies

def forecast_prices(df, periods=30):
    """
    We'll forecast 'close' over 'date' if they exist.
    """
    if not all(col in df.columns for col in ["date", "close"]):
        return pd.DataFrame()

    subdf = df.dropna(subset=["date", "close"]).copy()
    subdf.rename(columns={"date": "ds", "close": "y"}, inplace=True)
    subdf["ds"] = pd.to_datetime(subdf["ds"], errors="coerce")
    subdf.dropna(subset=["ds", "y"], inplace=True)

    if subdf.empty:
        return pd.DataFrame()

    model = Prophet(daily_seasonality=True)
    model.fit(subdf)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]

def analyze_fundamentals(df):
    """
    Correlation among numeric columns.
    """
    if df.empty:
        return None
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return None
    return numeric_df.corr().round(2)

def summarize_sectors(df):
    """
    Summarize distribution of 'GICS Sector' if present.
    """
    if df.empty or "GICS Sector" not in df.columns:
        return None
    sector_counts = df["GICS Sector"].value_counts()
    raw_text = "Sector distribution:\n"
    for sector, count in sector_counts.items():
        raw_text += f"{sector}: {count} companies\n"
    summary = summarizer(raw_text, max_length=50, min_length=10, do_sample=False)
    return summary[0]["summary_text"]

def generate_price_stats(df):
    """
    Stats for open, high, low, close, volume columns.
    """
    numeric_cols = ["open", "high", "low", "close", "volume"]
    sub = df.dropna(subset=numeric_cols, how="any", axis=0)
    if sub.empty:
        return pd.DataFrame()
    return sub[numeric_cols].describe().round(2)

################################################################################
# HTML TEMPLATES
################################################################################

HOME_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Multiple CSV AI Dashboard</title>
</head>
<body>
    <h1>Multiple CSV AI Dashboard</h1>
    <p>Upload CSV(s), select an active CSV, then do stats, anomalies, correlation, forecast, or chat!</p>
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
        <button type="submit">Show Price Stats</button>
    </form>
    <form action="/anomalies" method="get">
        <button type="submit">Detect Anomalies</button>
    </form>
    <form action="/forecast" method="get">
        <label>Days to Forecast:</label>
        <input type="number" name="periods" value="30" min="1"/>
        <button type="submit">Forecast Next N Days</button>
    </form>
    <form action="/fundamentals" method="get">
        <button type="submit">Fundamentals Correlation</button>
    </form>
    <form action="/securities-summary" method="get">
        <button type="submit">Summarize Sectors</button>
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
    <h1>Ask a question about the data (multi-turn chat)</h1>
    <p>Active CSV: {{active_csv}}</p>
    <div style="border:1px solid #ccc; padding:10px; width:600px; height:200px; overflow:auto;">
        {% for msg in messages %}
            <p><b>{{ msg.role }}:</b> {{ msg.content }}</p>
        {% endfor %}
    </div>
    <form method="post">
        <label>Your Question:</label><br>
        <input type="text" name="question" style="width:400px;" required><br><br>
        <button type="submit">Ask</button>
    </form>
    <br>
    <a href="/">Back to Home</a>
</body>
</html>
"""

################################################################################
# ROUTES
################################################################################

@app.route("/")
def home():
    global active_csv_key
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

    desc = generate_price_stats(df)
    if desc.empty:
        return render_template_string(RESULT_TEMPLATE, content="<p>No numeric columns found or no data.</p>")
    html_table = desc.to_html(classes="table table-striped", justify="center")
    return render_template_string(RESULT_TEMPLATE, content=html_table)

@app.route("/anomalies")
def anomalies_route():
    df = get_active_df()
    if df is None or df.empty:
        return render_template_string(RESULT_TEMPLATE, content="<p>No active CSV or data is empty.</p>")

    anomalies = detect_anomalies(df, contamination=0.01)
    if anomalies.empty:
        return render_template_string(RESULT_TEMPLATE, content="<p>No anomalies found or no numeric data.</p>")

    top_anomalies = anomalies.head(20)
    html_table = top_anomalies.to_html(classes="table table-striped", justify="center")
    result_msg = f"<p><b>Total anomalies:</b> {len(anomalies)}</p>" + html_table
    return render_template_string(RESULT_TEMPLATE, content=result_msg)

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

    fc = forecast_prices(df, periods=periods)
    if fc.empty:
        msg = "Not enough data to forecast or missing 'date'/'close' columns."
        return render_template_string(RESULT_TEMPLATE, content=msg)

    top_fc = fc.head(20).round(2)
    html_table = top_fc.to_html(classes="table table-striped", justify="center")
    content = f"<p>Forecast: next {periods} days (showing first 20):</p>{html_table}"
    return render_template_string(RESULT_TEMPLATE, content=content)

@app.route("/fundamentals")
def fundamentals_route():
    df = get_active_df()
    if df is None or df.empty:
        return render_template_string(RESULT_TEMPLATE, content="<p>No active CSV or data is empty.</p>")

    corr_mat = analyze_fundamentals(df)
    if corr_mat is None or corr_mat.empty:
        msg = "No numeric columns or no data to analyze."
        return render_template_string(RESULT_TEMPLATE, content=msg)

    html_table = corr_mat.to_html(classes="table table-striped", justify="center")
    return render_template_string(
        RESULT_TEMPLATE,
        content=f"<p><b>Fundamentals Correlation Matrix</b></p>{html_table}"
    )

@app.route("/securities-summary")
def securities_summary_route():
    df = get_active_df()
    if df is None or df.empty:
        return render_template_string(RESULT_TEMPLATE, content="<p>No active CSV or data is empty.</p>")

    summary = summarize_sectors(df)
    if summary is None:
        msg = "No 'GICS Sector' column found or summarizer error."
        return render_template_string(RESULT_TEMPLATE, content=msg)

    return render_template_string(
        RESULT_TEMPLATE,
        content=f"<p><b>Securities Sector Summary:</b></p><div>{summary}</div>"
    )

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

        # 2) Check numeric query first
        numeric_answer = analyze_numeric_query(user_question)
        if numeric_answer:
            ai_reply = numeric_answer
        else:
            # 3) Decide which model to use
            model_choice = decide_model(user_question)
            if model_choice == "qa":
                if roberta_qa_pipeline is None:
                    ai_reply = "Roberta QA model not loaded."
                else:
                    df = get_active_df()
                    columns_str = "None" if df is None or df.empty else str(df.columns.tolist())
                    conversation_text = (
                        f"We have an active CSV named '{active_csv_key}' with columns: {columns_str}\n\n"
                        "Here is the conversation so far:\n"
                    )
                    for msg in chat_history:
                        conversation_text += f"{msg['role'].upper()}: {msg['content']}\n"
                    try:
                        result = roberta_qa_pipeline(question=user_question, context=conversation_text)
                        ai_reply = result["answer"]
                    except Exception as ex:
                        ai_reply = f"Error in roberta QA: {ex}"
            else:
                # Use DeepSeek text-generation
                if deepseek_pipeline is None:
                    ai_reply = "DeepSeek model not loaded."
                else:
                    df = get_active_df()
                    columns_str = "None" if df is None or df.empty else str(df.columns.tolist())
                    prompt = (
                        f"We have an active CSV named '{active_csv_key}' with columns: {columns_str}.\n"
                        "This is a conversation about that data.\n"
                    )
                    for msg in chat_history:
                        prompt += f"{msg['role'].upper()}: {msg['content']}\n"
                    prompt += "ASSISTANT:"

                    try:
                        response = deepseek_pipeline(prompt, max_new_tokens=200, do_sample=True)
                        full_output = response[0]["generated_text"]
                        splitted = full_output.split("ASSISTANT:")
                        if len(splitted) > 1:
                            ai_reply = splitted[-1].strip()
                        else:
                            ai_reply = full_output
                    except Exception as ex:
                        ai_reply = f"Error generating answer with DeepSeek: {ex}"

        # 4) Append AI reply
        chat_history.append({"role": "assistant", "content": ai_reply})

        return render_template_string(CHAT_TEMPLATE, messages=chat_history, active_csv=active_csv_key or "None")


@app.route("/select", methods=["GET", "POST"])
def select_csv():
    """
    Allows user to select which CSV is active from the uploaded set.
    """
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


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
