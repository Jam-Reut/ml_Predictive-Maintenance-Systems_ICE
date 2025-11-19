import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

def train_models(data_path='predictive_maintenance_db.csv'):
    df = pd.read_csv(data_path)

    X = df.drop("failure_within_30d", axis=1)
    y = df["failure_within_30d"]

    categorical = ["component_type", "weekday"]
    numeric = [col for col in X.columns if col not in categorical]

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", "passthrough", numeric)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    log_reg = Pipeline(steps=[
        ("preprocess", preprocess),
        ("lr", LogisticRegression(max_iter=200, class_weight='balanced'))
    ])

    log_reg.fit(X_train, y_train)
    y_pred_lr = log_reg.predict(X_test)
    y_proba_lr = log_reg.predict_proba(X_test)[:, 1]

    print("### Logistische Regression ###")
    print(classification_report(y_test, y_pred_lr))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba_lr))

    rf = Pipeline(steps=[
        ("preprocess", preprocess),
        ("rf", RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'))
    ])

    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)[:, 1]

    print("\n### Random Forest ###")
    print(classification_report(y_test, y_pred_rf))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba_rf))

if __name__ == '__main__':
    train_models()
