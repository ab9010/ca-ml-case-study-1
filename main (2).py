import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix 

# 1. Load dataset
df = pd.read_csv(r"C:\python\Case_Study_1\diabetic_data.csv")

# 2. Create 30-day readmission target
df = df[df["readmitted"].isin(["<30", ">30", "NO"])]

y = (df["readmitted"] == "<30").astype(int)

X = df[
    [
        "age",
        "diag_1",
        "diag_2",
        "diag_3",
        "number_outpatient",
        "number_emergency",
        "number_inpatient"
    ]
]


# 3. Identify columns
numerical_features = [
    "number_outpatient",
    "number_emergency",
    "number_inpatient"
]

categorical_features = [
    "age",
    "diag_1",
    "diag_2",
    "diag_3"
]


# 4. Preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])


# 5. Logistic Regression with L2 regularization
model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", LogisticRegression(
        C=1.0,
        max_iter=1000
    ))
])


# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 7. Train model
model.fit(X_train, y_train)


# 8. Predict probability
y_prob = model.predict_proba(X_test)[:, 1]


# 9. ROC-AUC
auc = roc_auc_score(y_test, y_prob)

print("ROC-AUC:", auc)


# 10. Classification report
y_pred = model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Classification Report:")
print(classification_report(y_test, y_pred))

joblib.dump(model, "hospital_readmission_model.pkl")