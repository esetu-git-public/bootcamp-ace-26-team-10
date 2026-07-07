import joblib
from sklearn.linear_model import LogisticRegression
from utils.preprocessing import preprocess_pipeline

TRAIN_PATH = "dataset/Training_CKD_dataset.csv"
TEST_PATH = "dataset/Testing_CKD_dataset.csv"

X_train, X_test, y_train, y_test, encoders, scaler, class_names = preprocess_pipeline(TRAIN_PATH, TEST_PATH)

model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

joblib.dump(model, "model/test.pkl")

print("Saved successfully")

loaded = joblib.load("model/test.pkl")
print("Loaded successfully:", loaded)