import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. Đọc và chuẩn bị dữ liệu
df = pd.read_csv('medication_history.csv')

# Chọn các feature đơn giản để bắt đầu
features = ['day_of_week', 'time_of_day', 'blood_pressure_systolic']
target = 'was_taken'

X = df[features]
y = df[target]

# 2. Phân chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Huấn luyện mô hình
# Cây quyết định là một lựa chọn tốt vì nó dễ giải thích
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# 4. Đánh giá mô hình
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Độ chính xác của mô hình trên tập kiểm tra: {acc:.2f}")

# 5. Lưu mô hình đã huấn luyện
joblib.dump(model, 'medication_model.pkl')
print("Đã lưu mô hình tại 'medication_model.pkl'")