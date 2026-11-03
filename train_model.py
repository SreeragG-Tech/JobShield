"""
JobShield AI - Model Training & Export Script
===========================================
Trains the exact TF-IDF + Logistic Regression Pipeline from JobShield_AI_Completed_9c09480b.ipynb
on fake_job_postings.csv (EMSCAD dataset) and exports `backend/nlp_model.joblib`.

Usage:
    python backend/train_model.py --dataset fake_job_postings.csv
"""

import os
import re
import html
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib


def clean_html_text(value):
    if not value or pd.isna(value):
        return ""
    value = html.unescape(str(value))
    value = re.sub(r'<[^>]+>', ' ', value)
    value = re.sub(r'\s+', ' ', value)
    return value.strip()


def word_count(value):
    return len(re.findall(r'\b\w+\b', clean_html_text(value)))


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    text_cols = ['title', 'department', 'company_profile', 'description', 'requirements', 'benefits']
    for col in text_cols:
        df[col] = df[col].fillna('').astype(str)

    df['combined_text'] = df[text_cols].apply(
        lambda row: ' '.join(clean_html_text(value) for value in row),
        axis=1
    )

    spam_pattern = r'work from home|be your own boss|easy money|no experience|make money|earnings|cash'
    external_pattern = r'apply at|apply through|send (your )?resume|send (your )?cv|email your resume'
    email_pattern = r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}|mailto:'
    money_pattern = r'[$€£]|\b salary \b|\b income \b'

    df['contains_spamwords'] = df['combined_text'].str.contains(spam_pattern, case=False, regex=True).astype(int)
    df['has_consecutive_punctuation'] = df['title'].str.contains(r'[!?.,]{2,}', regex=True).astype(int)
    df['contains_money_in_title'] = df['title'].str.contains(money_pattern, case=False, regex=True).astype(int)
    df['contains_money_in_description'] = df['description'].str.contains(money_pattern, case=False, regex=True).astype(int)
    df['contains_email_link'] = df['combined_text'].str.contains(email_pattern, case=False, regex=True).astype(int)
    df['prompts_for_external_application'] = df['combined_text'].str.contains(external_pattern, case=False, regex=True).astype(int)

    df['has_no_company_profile'] = (df['company_profile'].str.strip() == '').astype(int)
    df['has_short_company_profile'] = (df['company_profile'].apply(word_count) < 10).astype(int)
    df['has_short_description'] = (df['description'].apply(word_count) < 10).astype(int)
    df['has_short_requirements'] = (df['requirements'].apply(word_count) < 10).astype(int)

    df['has_no_company_logo'] = (df['has_company_logo'] == 0).astype(int)
    df['has_no_questions'] = (df['has_questions'] == 0).astype(int)
    df['is_telecommuting'] = (df['telecommuting'] == 1).astype(int)

    df['has_no_html_lists_in_requirements'] = (
        ~df['requirements'].str.contains(r'<(ul|ol|li)\b', case=False, regex=True)
    ).astype(int)
    df['has_no_html_lists_in_benefits'] = (
        ~df['benefits'].str.contains(r'<(ul|ol|li)\b', case=False, regex=True)
    ).astype(int)

    extra_cols = ['employment_type', 'required_experience', 'required_education', 'industry', 'function']
    for col in extra_cols:
        if col not in df.columns:
            df[col] = ''
    df['has_incomplete_extra_attributes'] = df[extra_cols].isna().sum(axis=1).gt(0).astype(int)

    return df


def train_and_dump_model(csv_path: str, output_path: str):
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['fraudulent']).copy()
    df['fraudulent'] = pd.to_numeric(df['fraudulent'], errors='coerce').astype(int)
    df = engineer_features(df)

    rule_cols = [
        'contains_spamwords', 'has_consecutive_punctuation',
        'contains_money_in_title', 'contains_money_in_description',
        'contains_email_link', 'prompts_for_external_application',
        'has_no_company_profile', 'has_short_company_profile',
        'has_short_description', 'has_short_requirements',
        'has_no_company_logo', 'has_no_questions', 'is_telecommuting',
        'has_no_html_lists_in_requirements', 'has_no_html_lists_in_benefits',
        'has_incomplete_extra_attributes'
    ]

    numeric_features = ['telecommuting', 'has_company_logo', 'has_questions'] + rule_cols
    categorical_features = [
        'location', 'salary_range', 'employment_type',
        'required_experience', 'required_education', 'industry', 'function'
    ]

    X = df.drop(columns=['job_id', 'fraudulent'], errors='ignore')
    y = df['fraudulent']

    text_transformer = TfidfVectorizer(
        lowercase=True,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_features=15000,
        sublinear_tf=True
    )

    nlp_num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    nlp_cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    nlp_preprocessor = ColumnTransformer([
        ('text', text_transformer, 'combined_text'),
        ('num', nlp_num_transformer, numeric_features),
        ('cat', nlp_cat_transformer, categorical_features)
    ])

    nlp_model = Pipeline([
        ('preprocessor', nlp_preprocessor),
        ('model', LogisticRegression(
            max_iter=2000, class_weight='balanced',
            solver='liblinear', random_state=42
        ))
    ])

    print("Fitting nlp_model pipeline on dataset...")
    nlp_model.fit(X, y)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    joblib.dump(nlp_model, output_path)
    print(f"Successfully saved trained model pipeline to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="fake_job_postings.csv")
    parser.add_argument("--output", type=str, default=os.path.join(os.path.dirname(__file__), "nlp_model.joblib"))
    args = parser.parse_args()

    if os.path.exists(args.dataset):
        train_and_dump_model(args.dataset, args.output)
    else:
        print(f"Dataset {args.dataset} not found. If your model is already trained in Jupyter, run:")
        print("    import joblib")
        print("    joblib.dump(nlp_model, 'backend/nlp_model.joblib')")
