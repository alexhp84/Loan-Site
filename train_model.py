import os
import pickle
import pandas as pd
import numpy as np
import logging
from logging.handlers import TimedRotatingFileHandler

from typing import Tuple
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA

"""
Setting the file paths for all relevant files. 
This helps with calling files as well as having one place to change should the file structure be amended
"""

DATA_PATH = "data/loan_data.csv"
MODEL_PATH = "model/loan_svc_model.pkl"
LOGS_PATH = "logs/model.log"

"""
Setting up the logger with 7 day rotation logs.
This ensures the logs do not get too big, saving space.
This is outside a function so it works globally.
"""

#Logging for the train_model.py script and setting the level
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

#Setting 7 day rolling logs. At midnight, any logs older than 7 days will be deleted
handler = TimedRotatingFileHandler(
    LOGS_PATH,
    when="midnight",
    interval=1,
    backupCount=7,
    encoding="utf-8"
)
#Format of the log
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))

#Preventing duplicate logs should the handler be called multiple times
if not logger.handlers:
    logger.addHandler(handler)

"""
Setting up the features on a global level.
Split between numerical and text (Categorical) features.
"""
NUMERIC_FEATURES = [
    "person_income",
    "loan_amnt",
    "cb_person_cred_hist_length",
    "credit_score",
]

CATEGORICAL_FEATURES = [
    "person_home_ownership",
    "previous_loan_defaults_on_file",
]

#The variable for the target dataset column
TARGET_COLUMN = "loan_status"

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loading the dataset, cleaning strings, and performing the train/test split.
    """
    logger.info("Loading data")
    try:
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"Dataset not found at '{DATA_PATH}'.")
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError as e:
        logger.error(f"File not found error: {str(e)}")
        raise

    #Clean column names
    logger.info("Validating data")
    df.columns = df.columns.str.strip()

    # Clean string values in categorical columns
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    #Validate essential columns
    try:
        required = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COLUMN]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Dataset is missing required columns: {missing}")
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise

    #Coerce numeric columns
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    #Drop rows missing target or critical features
    df = df.dropna(subset=[TARGET_COLUMN]).copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    """
    Train/Test split (80/20 stratified).
    Using stratified maintains the same proportion of classes 
    (approved vs. denied loans) as the original dataset, preventing a skewed split.
    """
    logger.info("Testing the model")
    df_tr, df_te = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df[TARGET_COLUMN],
    )
    return df_tr.reset_index(drop=True), df_te.reset_index(drop=True)

def preprocess_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Extract features (x) and target (y) from a DataFrame."""
    logger.info("Extracting data")
    x = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET_COLUMN].copy() if TARGET_COLUMN in df.columns else None
    return x, y

def build_pipeline() -> Pipeline:
    """
    Build preprocessing and classifier pipeline.
    Uses CalibratedClassifierCV around SVC to prevent SVC(probability=True)
    deprecation warnings in scikit-learn >= 1.9.
    """
    logger.info("Pipeline building")

    #Building a mini-pipeline to normalize numeric features by removing the mean and scaling them to unit variance.
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    #Building a second mini-pipeline for the text features using OneHotEncoder to convert categorical text values
    categorical_transformer = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore", #Ignoring new categories
                    sparse_output=False, #Returns a dense Numpy array instead of a sparse matrix
                ),
            )
        ]
    )

    #Combining the two minipipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    #Initializes an Support Vector Classifier model using an RBF kernel
    logger.info("Base SVC Estimator set")
    base_svc = SVC(
        C=100,
        kernel="rbf",
        class_weight="balanced",
        random_state=42,
    )
    #CalibratedClassifierCV provides predict_proba without SVC probability=True warnings
    logger.info("Performing Cross-Validation")
    calibrated_svc = CalibratedClassifierCV(
        estimator=base_svc,
        ensemble=False,
    )

    #Chains the feature preprocessor and the calibrated classifier together into a single sequential workflow
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("svc", calibrated_svc),
        ]
    )
    return pipeline

def train(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """Train pipeline directly on dataset and save model to disk."""
    logger.info("Training the model directly")
    print("=== Training Model Directly ===")
    pipeline.fit(X, y)
    return pipeline

def evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Evaluate pipeline on test dataset."""
    logger.info("Pipeline evaluation")
    if y_test is None:
        print("No test target provided.")
        return

    #Calculating the prediction variables
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred)

    print("\n=== Test-set Evaluation ===")
    print(f"Accuracy: {acc:.4f}\n")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    #Generating the 3D PCA projection coordinates for Plotly for the web app
    X_trans = pipeline.named_steps['preprocessor'].transform(X_test.iloc[:500])
    pca = PCA(n_components=3)
    X_3d = pca.fit_transform(X_trans)

    pca_3d = {
        "class_0": [{"x": float(r[0]), "y": float(r[1]), "z": float(r[2])} for r, l in zip(X_3d, y_test[:500]) if
                    l == 0],
        "class_1": [{"x": float(r[0]), "y": float(r[1]), "z": float(r[2])} for r, l in zip(X_3d, y_test[:500]) if
                    l == 1]
    }

    #Extract support vectors count from base SVC
    calibrated = pipeline.named_steps['svc']
    base_svc = calibrated.calibrated_classifiers_[0].estimator
    sv_count = int(np.sum(base_svc.n_support_)) if hasattr(base_svc, 'n_support_') else 0

    #Calculate the SVC margin distribution for dashboard/API metrics
    X_transformed_test = pipeline.named_steps["preprocessor"].transform(X_test)
    margins = base_svc.decision_function(X_transformed_test)

    counts, edges = np.histogram(margins, bins=7)

    margin_bins = []
    for i in range(len(edges) - 1):
        margin_bins.append(
            f"{edges[i]:.2f} to {edges[i + 1]:.2f}"
        )

    margin_distribution = {
        "bins": margin_bins,
        "counts": counts.astype(int).tolist(),
    }

    #Package model payload with metrics for app.py and APIs
    logger.info("Generating payload")
    payload = {
        "pipeline": pipeline,
        "metrics": {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "support_vectors": sv_count,
            "confusion_matrix": cm.tolist(),
            "margin_distribution": margin_distribution,
        },
        "pca_3d": pca_3d,
    }

    logger.info("Exporting the model")
    #Exporting the model to PKL so it can be called
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(payload, f)

    print(f"\nModel saved to {MODEL_PATH}")

def main() -> None:
    """
    Putting all the functions together.
    This coordinates the data loading, preprocessing, model building, training, and evaluation steps
    It is inside a Try/Except block to capture error
    """
    try:
        logger.info("Starting model training pipeline.")
        df_tr, df_te = load_data()
        X_tr, y_tr = preprocess_features(df_tr)
        X_te, y_te = preprocess_features(df_te)
        pipe = build_pipeline()
        pipe = train(pipe, X_tr, y_tr)
        evaluate(pipe, X_te, y_te)
        logger.info("Pipeline completed successfully.")
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise

"""
Main execution flow, running the trainer.
"""
if __name__ == "__main__":
    main()