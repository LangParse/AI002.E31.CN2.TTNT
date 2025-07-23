import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Cấu hình
users = ['user_A', 'user_B', 'user_C']
num_days = 90  # Mô phỏng dữ liệu trong 90 ngày
start_date = datetime.now() - timedelta(days=num_days)

data = []

for user in users:
    for i in range(num_days * 2): # 2 liều mỗi ngày
        date = start_date + timedelta(days=i // 2, hours=(8 if i % 2 == 0 else 20))
        
        # Giả lập tính cách người dùng
        # user_A: tuân thủ tốt
        # user_B: hay quên vào cuối tuần
        # user_C: tuân thủ kém
        
        prob_miss = 0.1 # Mặc định
        if user == 'user_B' and date.weekday() >= 5: # Thứ 7 hoặc CN
            prob_miss = 0.6
        elif user == 'user_C':
            prob_miss = 0.4
            
        was_taken = 1 if np.random.rand() > prob_miss else 0
        
        # Mô phỏng huyết áp
        systolic = np.random.randint(110, 160)
        diastolic = np.random.randint(70, 100)
        
        # Nếu quên liều trước, huyết áp có thể cao hơn
        if len(data) > 0 and data[-1]['user_id'] == user and data[-1]['was_taken'] == 0:
            systolic += np.random.randint(5, 15)

        data.append({
            'user_id': user,
            'timestamp': date,
            'day_of_week': date.weekday(), # 0: Thứ 2, ..., 6: Chủ nhật
            'time_of_day': 0 if date.hour < 12 else 1, # 0: Sáng, 1: Tối
            'blood_pressure_systolic': systolic,
            'blood_pressure_diastolic': diastolic,
            'was_taken': was_taken
        })

df = pd.DataFrame(data)
df.to_csv('medication_history.csv', index=False)
print("Đã tạo dữ liệu mô phỏng thành công tại 'medication_history.csv'")
