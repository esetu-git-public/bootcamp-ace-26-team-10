import os
import joblib
from sklearn.linear_model import LogisticRegression
from ckd_utils.preprocessing import preprocess_pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "dataset", "Training_CKD_dataset.csv")
TEST_PATH = os.path.join(BASE_DIR, "dataset", "Testing_CKD_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "test.pkl")

X_train, X_test, y_train, y_test, encoders, scaler, class_names = preprocess_pipeline(TRAIN_PATH, TEST_PATH)

model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(model, MODEL_PATH)

print("Saved successfully")

loaded = joblib.load(MODEL_PATH)
print("Loaded successfully:", loaded)