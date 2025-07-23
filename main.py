import joblib
import pandas as pd

# Tải mô hình đã được huấn luyện
model = joblib.load('medication_model.pkl')
print("Chào mừng đến với Trợ lý sức khỏe AI!")
print("------------------------------------")

def predict_and_remind():
    """Hàm mô phỏng việc dự đoán và nhắc nhở cho một liều thuốc."""
    try:
        # 1. Thu thập thông tin ngữ cảnh từ người dùng
        day = int(input("Nhập thứ trong tuần (0=T2, 1=T3, ..., 6=CN): "))
        time_of_day = int(input("Nhập buổi (0=Sáng, 1=Tối): "))
        systolic = int(input("Nhập chỉ số huyết áp tâm thu của bạn: "))

        # 2. Tạo DataFrame cho dữ liệu đầu vào
        input_data = pd.DataFrame([[day, time_of_day, systolic]], 
                                  columns=['day_of_week', 'time_of_day', 'blood_pressure_systolic'])

        # 3. Đưa ra dự đoán
        prediction_proba = model.predict_proba(input_data)[0]
        prob_miss = prediction_proba[0] # Xác suất của nhãn 0 (quên)

        print("\n--- Phân tích từ AI ---")
        if prob_miss > 0.5:
            print(f"Cảnh báo! AI dự đoán nguy cơ bạn quên liều này là {prob_miss:.0%}.")
            print("Đây là lời nhắc nhở CƯỜNG ĐỘ CAO. Vui lòng uống thuốc ngay!")
        else:
            print(f"AI dự đoán nguy cơ bạn quên liều này thấp ({prob_miss:.0%}).")
            print("Đây là lời nhắc nhở nhẹ nhàng: Đã đến giờ uống thuốc.")
        print("-----------------------\n")

    except ValueError:
        print("Lỗi: Vui lòng nhập đúng định dạng số.")
    except Exception as e:
        print(f"Đã có lỗi xảy ra: {e}")

# Chạy vòng lặp chính của ứng dụng
if __name__ == "__main__":
    while True:
        predict_and_remind()
        cont = input("Bạn có muốn thử dự đoán cho liều tiếp theo không? (y/n): ")
        if cont.lower() != 'y':
            break