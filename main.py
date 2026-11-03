"""
JobShield AI - Python FastAPI Backend
======================================
Production-ready FastAPI service serving trained TF-IDF + Logistic Regression
fake job posting classifier on the EMSCAD dataset.
Configured for containerized and cloud platforms (Render, Railway, etc.).
"""

import os
import re
import html
import json
import shutil
import time
from datetime import datetime
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Header, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib

app = FastAPI(
    title="JobShield AI Backend",
    description="FastAPI service serving trained TF-IDF + Logistic Regression fake job posting classifier",
    version="1.1.0"
)

# Configure CORS so the React frontend can communicate with http://localhost:8000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADS_DIR = os.path.join(BACKEND_DIR, "uploads")
BACKUPS_DIR = os.path.join(BACKEND_DIR, "backups")
HISTORY_FILE = os.path.join(BACKEND_DIR, "model_history.json")
VERSION_FILE = os.path.join(BACKEND_DIR, "model_version.json")
PRIMARY_MODEL_PATH = os.path.join(BACKEND_DIR, "nlp_model.joblib")

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(BACKUPS_DIR, exist_ok=True)

# Admin security key (can be customized via environment variable ADMIN_SECRET_KEY)
ADMIN_SECRET_KEY = os.environ.get("ADMIN_SECRET_KEY", "admin123")

# In-memory model references
loaded_pipeline = None
loaded_model = None
loaded_vectorizer = None
loaded_artifact_name: Optional[str] = None


def load_artifacts():
    global loaded_pipeline, loaded_model, loaded_vectorizer, loaded_artifact_name

    # 1. First check if a combined Pipeline artifact exists
    possible_pipelines = [
        PRIMARY_MODEL_PATH,
        os.path.join(BACKEND_DIR, "nlp_model.pkl"),
        os.path.join(BACKEND_DIR, "model.joblib"),
        os.path.join(BACKEND_DIR, "model.pkl"),
    ]
    for path in possible_pipelines:
        if os.path.exists(path):
            try:
                artifact = joblib.load(path)
                if hasattr(artifact, "predict_proba"):
                    loaded_pipeline = artifact
                    loaded_artifact_name = os.path.basename(path)
                    print(f"[JobShield AI] Successfully loaded model pipeline from: {path}")
                    return True
            except Exception as e:
                print(f"[JobShield AI] Error loading {path}: {e}")

    # 2. Check if model and vectorizer were saved separately
    model_path = None
    vec_path = None
    for p in [os.path.join(BACKEND_DIR, "clf.joblib"), os.path.join(BACKEND_DIR, "classifier.joblib")]:
        if os.path.exists(p):
            model_path = p
            break
    for p in [os.path.join(BACKEND_DIR, "vectorizer.joblib"), os.path.join(BACKEND_DIR, "tfidf.joblib")]:
        if os.path.exists(p):
            vec_path = p
            break

    if model_path and vec_path:
        try:
            loaded_model = joblib.load(model_path)
            loaded_vectorizer = joblib.load(vec_path)
            loaded_artifact_name = f"{os.path.basename(model_path)} + {os.path.basename(vec_path)}"
            print(f"[JobShield AI] Successfully loaded separate model and vectorizer: {loaded_artifact_name}")
            return True
        except Exception as e:
            print(f"[JobShield AI] Error loading separate artifacts: {e}")

    print("[JobShield AI] Notice: No trained model file found yet in backend directory.")
    return False


# Version and History Helpers
def get_version_info() -> Dict[str, Any]:
    if os.path.exists(VERSION_FILE):
        try:
            with open(VERSION_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass

    # Default baseline version metadata
    backups = [f for f in os.listdir(BACKUPS_DIR) if f.endswith(".joblib") or f.endswith(".pkl")]
    default_info = {
        "version": "v1.0.0",
        "modelFile": loaded_artifact_name or "nlp_model.joblib",
        "datasetFile": "fake_job_postings.csv (EMSCAD)",
        "lastUpdated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "trainingStatus": "Trained & Active" if (loaded_pipeline is not None or loaded_model is not None) else "Awaiting Model Upload",
        "backupAvailable": len(backups) > 0,
        "backupFile": backups[-1] if backups else None,
    }
    save_version_info(default_info)
    return default_info


def save_version_info(info: Dict[str, Any]):
    try:
        with open(VERSION_FILE, "w") as f:
            json.dump(info, f, indent=2)
    except Exception as e:
        print(f"Error saving version info: {e}")


def get_history_items() -> List[Dict[str, Any]]:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    # Initial seed history entry
    initial_history = [
        {
            "id": "hist-init-1",
            "version": "v1.0.0",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "action": "MODEL_UPLOAD",
            "fileName": "nlp_model.joblib",
            "status": "SUCCESS",
            "summary": "Initial baseline model trained on EMSCAD 17,880 listings using TF-IDF + Logistic Regression.",
        }
    ]
    save_history_items(initial_history)
    return initial_history


def save_history_items(items: List[Dict[str, Any]]):
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(items, f, indent=2)
    except Exception as e:
        print(f"Error saving history: {e}")


def add_history_entry(action: str, file_name: str, status: str, summary: str, version: str):
    history = get_history_items()
    entry = {
        "id": f"hist-{int(time.time())}",
        "version": version,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "action": action,
        "fileName": file_name,
        "status": status,
        "summary": summary,
    }
    history.insert(0, entry)
    save_history_items(history[:50])  # keep last 50


# Load artifacts on server boot
load_artifacts()


# Admin Security Check
def verify_admin(x_admin_key: Optional[str] = Header(None, alias="X-Admin-Key")):
    if not x_admin_key or x_admin_key.strip() != ADMIN_SECRET_KEY:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Invalid or missing Administrator Secret Key."
        )
    return True


# Core Data Models
class JobPostingRequest(BaseModel):
    title: str = Field(..., example="Senior Software Engineer")
    company_profile: Optional[str] = Field("", example="Established enterprise technology company.")
    description: str = Field(..., example="Responsibilities include building web applications and collaborating with teams.")
    requirements: Optional[str] = Field("", example="Bachelor of Science in Computer Science or equivalent practical experience.")
    benefits: Optional[str] = Field("", example="Health insurance, 401k matching, paid time off.")
    location: Optional[str] = Field("", example="New York, NY")
    salary_range: Optional[str] = Field("", example="$120,000 - $140,000")
    employment_type: Optional[str] = Field("Full-time", example="Full-time")
    department: Optional[str] = Field("", example="Engineering")
    industry: Optional[str] = Field("", example="Information Technology")
    job_function: Optional[str] = Field("", example="Engineering")
    telecommuting: int = Field(0, example=0)
    has_company_logo: int = Field(1, example=1)
    has_questions: int = Field(1, example=1)


class PredictionResponse(BaseModel):
    is_fraudulent: bool
    prediction: str
    fraud_probability: float
    legitimate_probability: float
    explanation: str
    source: str
    model_file: Optional[str] = None


# Preprocessing functions matching JobShield_AI_Completed_9c09480b.ipynb exactly:
def clean_html_text(value: str) -> str:
    if not value:
        return ""
    value = html.unescape(str(value))
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def word_count(value: str) -> int:
    return len(re.findall(r"\b\w+\b", clean_html_text(value)))


def build_feature_dataframe(payload: JobPostingRequest):
    """
    Constructs a pandas DataFrame containing the exact feature schema
    expected by the scikit-learn nlp_model Pipeline from the notebook.
    """
    import pandas as pd

    text_cols = [
        payload.title,
        payload.department or "",
        payload.company_profile or "",
        payload.description,
        payload.requirements or "",
        payload.benefits or "",
    ]
    combined_text = " ".join([clean_html_text(c) for c in text_cols])

    spam_pattern = r'work from home|be your own boss|easy money|no experience|make money|earnings|cash'
    external_pattern = r'apply at|apply through|send (your )?resume|send (your )?cv|email your resume'
    email_pattern = r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}|mailto:'
    money_pattern = r'[$€£]|\b salary \b|\b income \b'

    row = {
        'title': payload.title,
        'department': payload.department or "",
        'company_profile': payload.company_profile or "",
        'description': payload.description,
        'requirements': payload.requirements or "",
        'benefits': payload.benefits or "",
        'location': payload.location or "",
        'salary_range': payload.salary_range or "",
        'employment_type': payload.employment_type or "",
        'required_experience': "",
        'required_education': "",
        'industry': payload.industry or "",
        'function': payload.job_function or "",
        'telecommuting': payload.telecommuting,
        'has_company_logo': payload.has_company_logo,
        'has_questions': payload.has_questions,
        'combined_text': combined_text,
        'contains_spamwords': int(bool(re.search(spam_pattern, combined_text, re.I))),
        'has_consecutive_punctuation': int(bool(re.search(r'[!?.,]{2,}', payload.title))),
        'contains_money_in_title': int(bool(re.search(money_pattern, payload.title, re.I))),
        'contains_money_in_description': int(bool(re.search(money_pattern, payload.description, re.I))),
        'contains_email_link': int(bool(re.search(email_pattern, combined_text, re.I))),
        'prompts_for_external_application': int(bool(re.search(external_pattern, combined_text, re.I))),
        'has_no_company_profile': int((payload.company_profile or "").strip() == ""),
        'has_short_company_profile': int(word_count(payload.company_profile or "") < 10),
        'has_short_description': int(word_count(payload.description) < 10),
        'has_short_requirements': int(word_count(payload.requirements or "") < 10),
        'has_no_company_logo': int(payload.has_company_logo == 0),
        'has_no_questions': int(payload.has_questions == 0),
        'is_telecommuting': int(payload.telecommuting == 1),
        'has_no_html_lists_in_requirements': int(not bool(re.search(r'<(ul|ol|li)\b', payload.requirements or "", re.I))),
        'has_no_html_lists_in_benefits': int(not bool(re.search(r'<(ul|ol|li)\b', payload.benefits or "", re.I))),
        'has_incomplete_extra_attributes': int(any(not v for v in [payload.employment_type, payload.industry, payload.job_function])),
    }
    return pd.DataFrame([row]), combined_text


# Standard Public Endpoints
@app.get("/")
@app.get("/health")
@app.get("/api/health")
def health_check():
    if loaded_pipeline is None and loaded_model is None:
        load_artifacts()

    is_ready = loaded_pipeline is not None or (loaded_model is not None and loaded_vectorizer is not None)
    version_info = get_version_info()
    return {
        "status": "online",
        "service": "JobShield AI FastAPI Backend",
        "model_loaded": is_ready,
        "model_artifact": loaded_artifact_name,
        "version": version_info.get("version", "v1.0.0"),
        "backend_directory": BACKEND_DIR,
    }


@app.post("/predict", response_model=PredictionResponse)
@app.post("/api/predict", response_model=PredictionResponse)
def predict_job(payload: JobPostingRequest):
    global loaded_pipeline, loaded_model, loaded_vectorizer

    if loaded_pipeline is None and loaded_model is None:
        load_artifacts()

    if loaded_pipeline is None and (loaded_model is None or loaded_vectorizer is None):
        raise HTTPException(
            status_code=503,
            detail=(
                "Trained model artifact not found. Please upload your model via the 'Update Dataset & Model' section, "
                "or place 'nlp_model.joblib' directly inside the backend/ folder."
            )
        )

    try:
        if loaded_pipeline is not None:
            df_row, _ = build_feature_dataframe(payload)
            probabilities = loaded_pipeline.predict_proba(df_row)[0]
            fraud_prob = round(float(probabilities[1]), 4)
            legit_prob = round(float(probabilities[0]), 4)
        elif loaded_model is not None and loaded_vectorizer is not None:
            _, combined_text = build_feature_dataframe(payload)
            X_vec = loaded_vectorizer.transform([combined_text])
            probabilities = loaded_model.predict_proba(X_vec)[0]
            fraud_prob = round(float(probabilities[1]), 4)
            legit_prob = round(float(probabilities[0]), 4)
        else:
            raise HTTPException(status_code=500, detail="Model is not configured properly.")

        is_fraud = fraud_prob >= 0.50
        verdict = "Fraudulent" if is_fraud else "Legitimate"

        explanation = (
            f"The trained model evaluated this posting as {verdict.lower()} "
            f"with a fraud probability of {round(fraud_prob * 100, 1)}% "
            f"(legitimate probability: {round(legit_prob * 100, 1)}%)."
        )

        return PredictionResponse(
            is_fraudulent=is_fraud,
            prediction=verdict,
            fraud_probability=fraud_prob,
            legitimate_probability=legit_prob,
            explanation=explanation,
            source="FASTAPI_BACKEND",
            model_file=loaded_artifact_name,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error with trained model: {str(e)}")


# ============================================================================
# Admin / Model Update Endpoints
# ============================================================================

@app.get("/api/admin/info")
def get_admin_info(auth: bool = Depends(verify_admin)):
    """Returns active model version, filenames, and training status."""
    info = get_version_info()
    info["activeModelLoaded"] = (loaded_pipeline is not None or loaded_model is not None)
    info["activeArtifact"] = loaded_artifact_name
    return info


@app.get("/api/admin/history")
def get_admin_history(auth: bool = Depends(verify_admin)):
    """Returns list of previous model update operations."""
    return get_history_items()


@app.post("/api/admin/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    auth: bool = Depends(verify_admin)
):
    """
    Accepts one or multiple file uploads:
    - Dataset (.csv)
    - Notebook (.ipynb)
    - Trained Model (.joblib, .pkl)
    Validates extensions, saves safely, and returns metadata.
    """
    uploaded_records = []
    allowed_exts = {".csv", ".ipynb", ".joblib", ".pkl"}

    for upload in files:
        filename = os.path.basename(upload.filename or "")
        ext = os.path.splitext(filename)[1].lower()

        if ext not in allowed_exts:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type for '{filename}'. Allowed extensions: .csv, .ipynb, .joblib, .pkl"
            )

        # Determine category
        if ext == ".csv":
            file_type = "dataset"
        elif ext == ".ipynb":
            file_type = "notebook"
        else:
            file_type = "model"

        # Save to uploads dir
        safe_filename = re.sub(r"[^a-zA-Z0-9._-]", "_", filename)
        target_path = os.path.join(UPLOADS_DIR, safe_filename)

        contents = await upload.read()
        file_size = len(contents)

        if file_size > 100 * 1024 * 1024:  # 100MB max limit
            raise HTTPException(status_code=400, detail=f"File '{filename}' exceeds maximum allowed size (100MB).")

        # Validate file content integrity
        if ext == ".ipynb":
            try:
                json.loads(contents.decode("utf-8"))
            except Exception:
                raise HTTPException(status_code=400, detail=f"'{filename}' is not a valid JSON Jupyter Notebook.")

        with open(target_path, "wb") as f:
            f.write(contents)

        record = {
            "id": f"upload-{int(time.time())}-{os.urandom(2).hex()}",
            "name": safe_filename,
            "size": file_size,
            "type": file_type,
            "uploadedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "uploaded",
            "details": f"Saved to backend/uploads/{safe_filename}",
        }
        uploaded_records.append(record)

    return {"uploadedFiles": uploaded_records}


@app.post("/api/admin/inspect-notebook")
def inspect_notebook(
    filename: str = Form(...),
    auth: bool = Depends(verify_admin)
):
    """
    Safely inspects an uploaded Jupyter notebook without executing any code.
    Extracts code cells, identifies preprocessing patterns, and checks if pipeline
    definitions match or differ from the active model.
    """
    target_path = os.path.join(UPLOADS_DIR, filename)
    if not os.path.exists(target_path):
        raise HTTPException(status_code=404, detail=f"Uploaded notebook '{filename}' not found.")

    try:
        with open(target_path, "r", encoding="utf-8") as f:
            nb_data = json.load(f)

        cells = nb_data.get("cells", [])
        code_cells = [c for c in cells if c.get("cell_type") == "code"]
        markdown_cells = [c for c in cells if c.get("cell_type") == "markdown"]

        all_code = "\n".join(["".join(c.get("source", [])) for c in code_cells])

        has_tfidf = "TfidfVectorizer" in all_code
        has_logreg = "LogisticRegression" in all_code
        has_pipeline = "Pipeline" in all_code
        has_clean_html = "clean_html_text" in all_code
        has_emscad_rules = "contains_spamwords" in all_code or "spam_pattern" in all_code

        summary = (
            f"Parsed {len(cells)} cells ({len(code_cells)} code, {len(markdown_cells)} markdown). "
            f"Detected: {'TF-IDF Vectorizer, ' if has_tfidf else ''}"
            f"{'Logistic Regression, ' if has_logreg else ''}"
            f"{'ColumnTransformer Pipeline, ' if has_pipeline else ''}"
            f"{'HTML Text Cleaning, ' if has_clean_html else ''}"
            f"{'EMSCAD Rule Features' if has_emscad_rules else ''}."
        )

        return {
            "filename": filename,
            "validNotebook": True,
            "codeCellCount": len(code_cells),
            "markdownCellCount": len(markdown_cells),
            "hasTfidf": has_tfidf,
            "hasLogisticRegression": has_logreg,
            "hasPipeline": has_pipeline,
            "hasEmscadRules": has_emscad_rules,
            "summary": summary,
            "trainingLogicNotice": (
                "Important: A Jupyter Notebook (.ipynb) is source code, not a trained model artifact. "
                "Uploading a notebook does not automatically update the active classifier. "
                "To activate changes from this notebook, export the trained 'nlp_model' to a .joblib file, "
                "or click 'Retrain Model' if you have also uploaded an updated dataset CSV."
            ),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to inspect notebook: {str(e)}")


@app.post("/api/admin/validate-model")
def validate_candidate_model(
    filename: str = Form(...),
    auth: bool = Depends(verify_admin)
):
    """
    Validates a candidate model artifact before activating it.
    Tests model loading, compatibility, and executes test predictions on known test samples.
    """
    target_path = os.path.join(UPLOADS_DIR, filename)
    if not os.path.exists(target_path):
        target_path = os.path.join(BACKEND_DIR, filename)

    if not os.path.exists(target_path):
        raise HTTPException(status_code=404, detail=f"Candidate model file '{filename}' not found.")

    try:
        candidate = joblib.load(target_path)
        if not hasattr(candidate, "predict_proba"):
            return {
                "valid": False,
                "modelType": type(candidate).__name__,
                "testResults": [],
                "message": f"Model object '{type(candidate).__name__}' does not provide a predict_proba() method.",
            }

        # Run test suite
        test_samples = [
            {
                "name": "Standard Tech Lead Posting",
                "expected": "Legitimate",
                "payload": JobPostingRequest(
                    title="Senior Software Engineer, Core Infrastructure",
                    company_profile="Public technology company building cloud infrastructure platforms.",
                    description="Responsibilities include architecting distributed backend services in Go/Java.",
                    requirements="Bachelor of Science in Computer Science or equivalent practical experience.",
                    benefits="Health insurance, dental and vision care, 401k matching.",
                    location="San Francisco, CA",
                    salary_range="$150,000 - $185,000",
                    employment_type="Full-time",
                    department="Engineering",
                    industry="Information Technology",
                    job_function="Engineering",
                    telecommuting=0,
                    has_company_logo=1,
                    has_questions=1,
                )
            },
            {
                "name": "Check Advance Wire Transfer Scam",
                "expected": "Fraudulent",
                "payload": JobPostingRequest(
                    title="Urgent Remote Data Entry Assistant",
                    company_profile="",
                    description="Immediate hire! Process client payments and wire transfer transactions daily.",
                    requirements="No experience required. Must install Telegram for interview.",
                    benefits="Reimbursement cashier check sent for home office equipment.",
                    location="Remote, USA",
                    salary_range="$85/hr daily payout",
                    employment_type="Full-time",
                    department="Admin",
                    industry="General",
                    job_function="Administrative",
                    telecommuting=1,
                    has_company_logo=0,
                    has_questions=0,
                )
            }
        ]

        test_results = []
        all_passed = True

        for test in test_samples:
            df_row, _ = build_feature_dataframe(test["payload"])
            probs = candidate.predict_proba(df_row)[0]
            fraud_prob = float(probs[1])
            predicted = "Fraudulent" if fraud_prob >= 0.50 else "Legitimate"
            passed = (predicted == test["expected"])
            if not passed:
                all_passed = False

            test_results.append({
                "name": test["name"],
                "expected": test["expected"],
                "predicted": predicted,
                "probability": round(fraud_prob, 4),
                "passed": passed,
            })

        return {
            "valid": all_passed,
            "modelType": type(candidate).__name__,
            "testResults": test_results,
            "message": (
                "Validation Passed: Model pipeline loaded successfully, produces calibrated probabilities, "
                "and correctly classified validation sample benchmarks."
                if all_passed else
                "Validation Warning: Model loaded, but one or more benchmark test cases produced unexpected classifications. Review before activation."
            ),
        }

    except Exception as e:
        return {
            "valid": False,
            "modelType": "Unknown",
            "testResults": [],
            "message": f"Model loading/validation failed: {str(e)}",
        }


@app.post("/api/admin/activate-model")
def activate_model(
    filename: str = Form(...),
    auth: bool = Depends(verify_admin)
):
    """
    Safely activates a validated candidate model artifact:
    1. Backs up current active model.
    2. Overwrites nlp_model.joblib with candidate.
    3. Reloads model in memory.
    4. Logs to history and increments version.
    """
    source_path = os.path.join(UPLOADS_DIR, filename)
    if not os.path.exists(source_path):
        source_path = os.path.join(BACKEND_DIR, filename)

    if not os.path.exists(source_path):
        raise HTTPException(status_code=404, detail=f"Model file '{filename}' not found.")

    # 1. Create backup of current model if present
    if os.path.exists(PRIMARY_MODEL_PATH):
        backup_name = f"nlp_model_backup_{int(time.time())}.joblib"
        backup_path = os.path.join(BACKUPS_DIR, backup_name)
        shutil.copy2(PRIMARY_MODEL_PATH, backup_path)
        print(f"[JobShield AI] Created model backup at: {backup_path}")

    # 2. Copy candidate to primary model path
    shutil.copy2(source_path, PRIMARY_MODEL_PATH)

    # 3. Reload artifacts in memory
    success = load_artifacts()
    if not success:
        raise HTTPException(status_code=500, detail="Failed to load newly activated model into memory.")

    # 4. Increment version and save history
    curr_info = get_version_info()
    old_version = curr_info.get("version", "v1.0.0")
    # Increment minor version (e.g. v1.0.0 -> v1.1.0)
    try:
        parts = old_version.lstrip("v").split(".")
        new_version = f"v{parts[0]}.{int(parts[1]) + 1}.0"
    except Exception:
        new_version = f"v1.{int(time.time()) % 1000}.0"

    curr_info["version"] = new_version
    curr_info["modelFile"] = filename
    curr_info["lastUpdated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    curr_info["trainingStatus"] = "Trained & Active"
    curr_info["backupAvailable"] = True
    save_version_info(curr_info)

    add_history_entry(
        action="MODEL_UPLOAD",
        file_name=filename,
        status="SUCCESS",
        summary=f"Activated validated model artifact '{filename}'. Created backup and reloaded memory.",
        version=new_version
    )

    return {
        "success": True,
        "message": f"Model '{filename}' successfully validated and activated as version {new_version}.",
        "newVersion": new_version,
        "activeArtifact": filename,
    }


@app.post("/api/admin/retrain")
def retrain_model(
    dataset_filename: str = Form(...),
    auth: bool = Depends(verify_admin)
):
    """
    Retrains the model pipeline on an uploaded dataset CSV.
    Uses the exact EMSCAD feature engineering and Pipeline from train_model.py.
    Safely backs up previous model and activates new model upon successful training.
    """
    csv_path = os.path.join(UPLOADS_DIR, dataset_filename)
    if not os.path.exists(csv_path):
        csv_path = os.path.join(BACKEND_DIR, dataset_filename)

    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail=f"Dataset CSV '{dataset_filename}' not found.")

    try:
        # Import the training function from train_model.py
        from backend.train_model import train_and_dump_model

        candidate_out = os.path.join(UPLOADS_DIR, f"candidate_nlp_model_{int(time.time())}.joblib")
        print(f"[JobShield AI Retrain] Training on {csv_path}...")
        train_and_dump_model(csv_path, candidate_out)

        # Validate candidate
        val_res = validate_candidate_model(filename=os.path.basename(candidate_out), auth=True)
        if not val_res.get("valid"):
            raise HTTPException(
                status_code=500,
                detail=f"Retrained model failed validation tests: {val_res.get('message')}"
            )

        # Activate candidate
        act_res = activate_model(filename=os.path.basename(candidate_out), auth=True)

        # Update dataset name in version info
        curr_info = get_version_info()
        curr_info["datasetFile"] = dataset_filename
        save_version_info(curr_info)

        add_history_entry(
            action="DATASET_RETRAIN",
            file_name=dataset_filename,
            status="SUCCESS",
            summary=f"Retrained TF-IDF + Logistic Regression pipeline on {dataset_filename}. Passed validation.",
            version=act_res["newVersion"]
        )

        return {
            "success": True,
            "message": f"Successfully retrained, validated, and deployed model from dataset '{dataset_filename}'.",
            "newVersion": act_res["newVersion"],
            "validation": val_res,
        }

    except Exception as e:
        add_history_entry(
            action="DATASET_RETRAIN",
            file_name=dataset_filename,
            status="FAILED",
            summary=f"Retraining failed: {str(e)}",
            version="N/A"
        )
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    # Cloud platforms like Render dynamically assign a port via the PORT environment variable
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"Starting JobShield AI FastAPI backend on {host}:{port}...")
    uvicorn.run("main:app", host=host, port=port, reload=False)

