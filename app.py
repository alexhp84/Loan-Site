import os
import io
import json
import pickle
import logging
import zipfile
import threading
import uuid

import pandas as pd
from logging.handlers import TimedRotatingFileHandler
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.exceptions import HTTPException
from flasgger import Swagger

import train_model

app = Flask(__name__)
swagger = Swagger(app)

os.makedirs("logs", exist_ok=True)
os.makedirs("model", exist_ok=True)
os.makedirs("data", exist_ok=True)

DATA_PATH = os.path.join("data", "loan_data.csv")
MODEL_PATH = os.path.join("model", "loan_svc_model.pkl")

RETRAIN_JOBS = {}
RETRAIN_LOCK = threading.Lock()


class CategoryFilter(logging.Filter):
    def __init__(self, category):
        super().__init__()
        self.category = category

    def filter(self, record):
        log_type = getattr(record, "log_type", record.levelname)
        return log_type == self.category


formatter = logging.Formatter(
    "[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def add_log_handler(filename, category):
    handler = TimedRotatingFileHandler(
        os.path.join("logs", filename),
        when="midnight",
        interval=1,
        backupCount=7,
        encoding="utf-8",
    )
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    handler.addFilter(CategoryFilter(category))
    app.logger.addHandler(handler)


add_log_handler("app.log", "INFO")
add_log_handler("model.log", "MODEL")
add_log_handler("http.log", "HTTP")
add_log_handler("error.log", "ERROR")
app.logger.setLevel(logging.INFO)


def log_app_event(message):
    app.logger.info(message, extra={"log_type": "INFO"})


def log_model_event(message):
    app.logger.info(message, extra={"log_type": "MODEL"})


log_app_event("Loan application server initialized.")


@app.before_request
def log_request_info():
    if not request.path.startswith("/static") and not request.path.startswith("/apidocs"):
        app.logger.info(
            f"HTTP {request.method} {request.path} - Remote IP: {request.remote_addr}",
            extra={"log_type": "HTTP"},
        )


@app.errorhandler(Exception)
def handle_exception(e):
    if isinstance(e, HTTPException):
        return e
    app.logger.error(
        f"Unhandled Server Exception: {str(e)}",
        exc_info=e,
        extra={"log_type": "ERROR"},
    )
    return jsonify({"status": "error", "message": "Internal Server Error"}), 500


@app.route("/favicon.ico")
def favicon():
    return "", 204


@app.route("/")
def index_page():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard_page():
    return render_template("dashboard.html")


@app.route("/model")
def model_page():
    return render_template("model.html")


@app.route("/api")
def api_page():
    return render_template("api.html")


@app.route("/logs")
def logs_page():
    return render_template("logs.html")


@app.route("/data-preview")
def data_preview():
    """Return the first ten rows of the active CSV.

    ---
    tags:
      - Data
    """
    if not os.path.exists(DATA_PATH):
        return jsonify({"headers": [], "rows": []})

    try:
        df = pd.read_csv(DATA_PATH)
        return jsonify(
            {
                "headers": list(df.columns),
                "rows": df.head(10).fillna("").values.tolist(),
                "row_count": int(len(df)),
            }
        )
    except Exception as e:
        app.logger.error(
            f"Failed to read dataset preview: {str(e)}",
            extra={"log_type": "ERROR"},
        )
        return jsonify({"headers": [], "rows": [], "row_count": 0}), 500


@app.route("/download-model")
def download_model():
    """Download the active PKL model artifact.

    ---
    tags:
      - Model
    """
    if not os.path.exists(MODEL_PATH):
        return jsonify({"status": "error", "message": "Model PKL does not exist."}), 404

    return send_file(
        MODEL_PATH,
        as_attachment=True,
        download_name="loan_svc_model.pkl",
        mimetype="application/octet-stream",
    )


def set_job(job_id, **values):
    with RETRAIN_LOCK:
        RETRAIN_JOBS.setdefault(job_id, {}).update(values)


def get_job(job_id):
    with RETRAIN_LOCK:
        return dict(RETRAIN_JOBS.get(job_id, {}))


def run_training_job(job_id):
    try:
        set_job(job_id, progress=10, stage="Loading dataset", status="running")
        log_model_event("Starting training job.")

        df_tr, df_te = train_model.load_data()

        set_job(job_id, progress=30, stage="Preparing training and test data")
        X_tr, y_tr = train_model.preprocess_features(df_tr)
        X_te, y_te = train_model.preprocess_features(df_te)

        set_job(job_id, progress=50, stage="Building model pipeline")
        pipe = train_model.build_pipeline()

        set_job(job_id, progress=65, stage="Training SVC model")
        pipe = train_model.train(pipe, X_tr, y_tr)

        set_job(job_id, progress=90, stage="Evaluating model")
        train_model.evaluate(pipe, X_te, y_te)

        with open(MODEL_PATH, "rb") as f:
            payload = pickle.load(f)

        acc_val = float(payload.get("metrics", {}).get("accuracy", 0.0))
        accuracy = acc_val * 100 if acc_val <= 1 else acc_val

        log_model_event(
            f"Retraining completed successfully. New Accuracy: {accuracy:.2f}%"
        )

        set_job(
            job_id,
            progress=100,
            stage="Completed",
            status="success",
            message=f"Model regenerated successfully. Accuracy: {accuracy:.2f}%",
            accuracy=f"{accuracy:.2f}%",
        )

    except Exception as e:
        log_model_event(f"Model retraining failed: {str(e)}")
        app.logger.error(
            f"Retraining error: {str(e)}",
            exc_info=e,
            extra={"log_type": "ERROR"},
        )
        set_job(
            job_id,
            progress=100,
            stage="Failed",
            status="error",
            message=f"Retraining failed: {str(e)}",
        )


@app.route("/retrain", methods=["POST"])
def retrain_model_endpoint():
    """Clear/retrain the model or replace the CSV and retrain.

    ---
    tags:
      - Training
    responses:
      200:
        description: Training job started
    """
    try:
        replace_dataset = request.form.get("replace_dataset") == "true"
        clear_pkl = request.form.get("clear_pkl") == "true"
        mapping_type = request.form.get("mapping_type", "default")
        custom_mappings_raw = request.form.get("custom_mappings", "{}")

        try:
            custom_mappings = json.loads(custom_mappings_raw)
        except json.JSONDecodeError:
            return jsonify({"status": "error", "message": "Invalid column mapping."}), 400

        if replace_dataset:
            uploaded = request.files.get("file")
            if uploaded is None or not uploaded.filename:
                return jsonify({"status": "error", "message": "No CSV file supplied."}), 400

            if not uploaded.filename.lower().endswith(".csv"):
                return jsonify({"status": "error", "message": "Only CSV files are supported."}), 400

            temp_path = DATA_PATH + ".uploading"
            uploaded.save(temp_path)

            try:
                df = pd.read_csv(temp_path)
                df.columns = df.columns.astype(str).str.strip()

                if mapping_type == "custom":
                    rename_map = {
                        source: target
                        for target, source in custom_mappings.items()
                        if source
                    }
                    df.rename(columns=rename_map, inplace=True)
                    log_model_event(f"Applied custom header mappings: {rename_map}")

                required = (
                    train_model.NUMERIC_FEATURES
                    + train_model.CATEGORICAL_FEATURES
                    + [train_model.TARGET_COLUMN]
                )
                missing = [column for column in required if column not in df.columns]

                if missing:
                    return jsonify(
                        {
                            "status": "error",
                            "message": f"Dataset is missing required columns: {missing}",
                        }
                    ), 400

                df.to_csv(temp_path, index=False)

                # Only replace the live files after the uploaded dataset passes validation.
                if os.path.exists(DATA_PATH):
                    os.remove(DATA_PATH)
                if os.path.exists(MODEL_PATH):
                    os.remove(MODEL_PATH)

                os.replace(temp_path, DATA_PATH)
                log_model_event("Replaced existing CSV and PKL with new dataset.")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        elif clear_pkl:
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)
                log_model_event("Cleared existing model PKL file.")
            else:
                log_model_event("Clear requested but no model PKL existed.")

        else:
            return jsonify(
                {
                    "status": "error",
                    "message": "Specify clear_pkl=true or replace_dataset=true.",
                }
            ), 400

        job_id = uuid.uuid4().hex
        set_job(
            job_id,
            progress=5,
            stage="Initializing",
            status="running",
            message="Training started.",
        )

        thread = threading.Thread(
            target=run_training_job,
            args=(job_id,),
            daemon=True,
        )
        thread.start()

        return jsonify(
            {
                "status": "started",
                "job_id": job_id,
                "message": "Training started.",
            }
        )

    except Exception as e:
        log_model_event(f"Failed to start retraining: {str(e)}")
        app.logger.error(
            f"Retraining start error: {str(e)}",
            exc_info=e,
            extra={"log_type": "ERROR"},
        )
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/retrain-status/<job_id>")
def retrain_status(job_id):
    """Return live retraining progress for a job.

    ---
    tags:
      - Training
    """
    job = get_job(job_id)
    if not job:
        return jsonify({"status": "error", "message": "Training job not found."}), 404
    return jsonify(job)


@app.route("/model-info")
def model_info():
    """Return the active model metadata, metrics, PCA projection and dataset information.

    ---
    tags:
      - Model
    responses:
      200:
        description: Active model information
    """
    if not os.path.exists(MODEL_PATH):
        return jsonify({"status": "error", "message": "Model not trained yet."}), 404

    try:
        with open(MODEL_PATH, "rb") as f:
            payload = pickle.load(f)

        metrics_payload = payload.get("metrics", {})

        def percent(value):
            if value is None:
                return 0.0
            value = float(value)
            return value * 100 if value <= 1 else value

        metrics = {
            "accuracy": percent(metrics_payload.get("accuracy", 0)),
            "precision": percent(metrics_payload.get("precision", 0)),
            "recall": percent(metrics_payload.get("recall", 0)),
            "f1_score": percent(metrics_payload.get("f1_score", metrics_payload.get("f1", 0))),
            "support_vectors": int(metrics_payload.get("support_vectors", 0)),
            "confusion_matrix": metrics_payload.get("confusion_matrix", [[0, 0], [0, 0]]),
            "margin_distribution": metrics_payload.get("margin_distribution", {"bins": [], "counts": []}),
        }

        pipeline = payload.get("pipeline")
        base_svc = None
        if pipeline is not None:
            try:
                calibrated = pipeline.named_steps.get("svc")
                if calibrated is not None and getattr(calibrated, "calibrated_classifiers_", None):
                    base_svc = calibrated.calibrated_classifiers_[0].estimator
            except Exception:
                base_svc = None

        dataset_rows = 0
        dataset_columns = 0
        if os.path.exists(DATA_PATH):
            df = pd.read_csv(DATA_PATH)
            dataset_rows = len(df)
            dataset_columns = len(df.columns)

        model_size = os.path.getsize(MODEL_PATH)
        trained_timestamp = os.path.getmtime(MODEL_PATH)
        from datetime import datetime
        trained_on = datetime.fromtimestamp(trained_timestamp).strftime("%d %b %Y %H:%M")

        model = {
            "name": "Loan Approval SVC",
            "algorithm": "SVC (Support Vector Classifier)",
            "kernel": getattr(base_svc, "kernel", "rbf"),
            "c": getattr(base_svc, "C", 1.0),
            "gamma": getattr(base_svc, "gamma", "scale"),
            "probability": "True (calibrated)",
            "class_weight": getattr(base_svc, "class_weight", None) or "None",
            "decision_function_shape": "Binary (n_samples, 1)",
            "target": train_model.TARGET_COLUMN,
            "class_0": "Rejected",
            "class_1": "Approved",
            "pipeline": "StandardScaler + OneHotEncoder + Calibrated SVC",
            "model_file": os.path.basename(MODEL_PATH),
            "file_size": f"{model_size / (1024 * 1024):.2f} MB",
            "trained_on": trained_on,
            "features": train_model.NUMERIC_FEATURES + train_model.CATEGORICAL_FEATURES,
        }

        return jsonify({
            "status": "success",
            "metrics": metrics,
            "pca_3d": payload.get("pca_3d", {}),
            "model": model,
            "dataset": {
                "filename": os.path.basename(DATA_PATH),
                "rows": dataset_rows,
                "columns": dataset_columns,
            },
        })

    except Exception as e:
        app.logger.error(
            f"Failed to read model info: {str(e)}",
            extra={"log_type": "ERROR"},
        )
        return jsonify({"status": "error", "message": str(e)}), 500


def build_input_data(req_data):
    feature_columns = train_model.NUMERIC_FEATURES + train_model.CATEGORICAL_FEATURES
    filtered_data = {}

    for key in feature_columns:
        if key in req_data and req_data[key] != "":
            if key in train_model.NUMERIC_FEATURES:
                filtered_data[key] = float(req_data[key])
            else:
                filtered_data[key] = str(req_data[key]).strip()
        else:
            filtered_data[key] = (
                0.0 if key in train_model.NUMERIC_FEATURES else "unknown"
            )

    return pd.DataFrame([filtered_data], columns=feature_columns), filtered_data


@app.route("/predict", methods=["POST"])
def predict():
    """Run loan approval prediction.

    ---
    tags:
      - Prediction
    responses:
      200:
        description: Prediction result
    """
    if not os.path.exists(MODEL_PATH):
        return jsonify({"status": "error", "message": "Model not available for predictions."}), 400

    try:
        req_data = request.get_json() if request.is_json else request.form.to_dict()
        input_df, filtered_data = build_input_data(req_data)

        with open(MODEL_PATH, "rb") as f:
            payload = pickle.load(f)

        pipeline = payload["pipeline"]
        prediction = int(pipeline.predict(input_df)[0])
        probabilities = (
            pipeline.predict_proba(input_df)[0].tolist()
            if hasattr(pipeline, "predict_proba")
            else []
        )

        log_model_event(
            f"Prediction requested: Input={filtered_data} Result={prediction}"
        )

        return jsonify(
            {
                "status": "success",
                "prediction_code": prediction,
                "prediction": "Approved" if prediction == 1 else "Denied",
                "probabilities": probabilities,
            }
        )

    except Exception as e:
        app.logger.error(
            f"Prediction error: {str(e)}",
            exc_info=e,
            extra={"log_type": "ERROR"},
        )
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/margin", methods=["POST"])
def calculate_margin():
    """Calculate the SVC decision margin and prediction.

    ---
    tags:
      - Prediction
    responses:
      200:
        description: Margin and prediction result
    """
    if not os.path.exists(MODEL_PATH):
        return jsonify({"status": "error", "message": "Model not available."}), 400

    try:
        req_data = request.get_json() if request.is_json else request.form.to_dict()
        input_df, filtered_data = build_input_data(req_data)

        with open(MODEL_PATH, "rb") as f:
            payload = pickle.load(f)

        pipeline = payload["pipeline"]
        preprocessor = pipeline.named_steps["preprocessor"]
        classifier = pipeline.named_steps["svc"]

        X_transformed = preprocessor.transform(input_df)

        if hasattr(classifier, "decision_function"):
            margin_score = float(classifier.decision_function(X_transformed)[0])
        elif (
            hasattr(classifier, "calibrated_classifiers_")
            and classifier.calibrated_classifiers_
        ):
            base_estimator = classifier.calibrated_classifiers_[0].estimator
            margin_score = float(base_estimator.decision_function(X_transformed)[0])
        else:
            margin_score = 0.0

        prediction = int(pipeline.predict(input_df)[0])

        log_model_event(
            f"Margin evaluated: Input={filtered_data} Code={prediction} Margin={margin_score:.4f}"
        )

        return jsonify(
            {
                "status": "success",
                "prediction_code": prediction,
                "prediction": "Approved" if prediction == 1 else "Denied",
                "margin": margin_score,
            }
        )

    except Exception as e:
        app.logger.error(
            f"Margin API error: {str(e)}",
            exc_info=e,
            extra={"log_type": "ERROR"},
        )
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api-status")
def api_status():
    """List the REST endpoints exposed by the application.

    ---
    tags:
      - API
    """
    model_loaded = os.path.exists(MODEL_PATH)

    endpoints = [
        {
            "endpoint": "/predict",
            "method": "POST",
            "description": "Run classification inference on applicant data",
            "status": "Online" if model_loaded else "Disabled",
        },
        {
            "endpoint": "/margin",
            "method": "POST",
            "description": "Calculate decision margin and classification",
            "status": "Online" if model_loaded else "Disabled",
        },
        {
            "endpoint": "/retrain",
            "method": "POST",
            "description": "Clear/retrain or replace dataset and retrain",
            "status": "Online",
        },
        {
            "endpoint": "/api-status",
            "method": "GET",
            "description": "List all API endpoints and their current status",
            "status": "Online",
        },
        {
            "endpoint": "/data-preview",
            "method": "GET",
            "description": "Preview the active loan dataset",
            "status": "Online",
        },
        {
            "endpoint": "/download-model",
            "method": "GET",
            "description": "Download the active PKL model artifact",
            "status": "Online" if model_loaded else "Disabled",
        },
        {
            "endpoint": "/get-log",
            "method": "GET",
            "description": "Retrieve application, model, HTTP or error logs",
            "status": "Online",
        },
        {
            "endpoint": "/download-log/<log_name>",
            "method": "GET",
            "description": "Download an individual log or all logs as a ZIP",
            "status": "Online",
        },
        {
            "endpoint": "/model-info",
            "method": "GET",
            "description": "Fetch model metrics, support vectors and PCA coordinates",
            "status": "Online" if model_loaded else "Disabled",
        },
        {
            "endpoint": "/retrain-status/<job_id>",
            "method": "GET",
            "description": "Get live retraining progress",
            "status": "Online",
        },
    ]

    return jsonify(endpoints)


@app.route("/get-log")
def get_log():
    """Return log records for the selected log source.

    ---
    tags:
      - Logs
    """
    log_type = request.args.get("type", "app")
    file_map = {
        "app": "app.log",
        "model": "model.log",
        "http": "http.log",
        "error": "error.log",
    }

    target_path = os.path.join("logs", file_map.get(log_type, "app.log"))

    if not os.path.exists(target_path):
        return jsonify({"logs": []})

    with open(target_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    return jsonify({"logs": [line.strip() for line in lines if line.strip()]})


@app.route("/download-log/<path:log_name>")
def download_log(log_name):
    """Download one log file or all logs as a ZIP.

    ---
    tags:
      - Logs
    """
    file_map = {
        "app": "app.log",
        "model": "model.log",
        "http": "http.log",
        "error": "error.log",
    }

    if log_name == "all":
        memory_file = io.BytesIO()
        with zipfile.ZipFile(memory_file, "w", zipfile.ZIP_DEFLATED) as zf:
            for filename in file_map.values():
                file_path = os.path.join("logs", filename)
                if os.path.exists(file_path):
                    zf.write(file_path, filename)

        memory_file.seek(0)
        return send_file(
            memory_file,
            mimetype="application/zip",
            as_attachment=True,
            download_name="all_logs.zip",
        )

    filename = file_map.get(log_name, log_name)
    target_path = os.path.join("logs", filename)

    if not os.path.exists(target_path):
        return "Log file not found", 404

    return send_file(target_path, as_attachment=True, download_name=filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
