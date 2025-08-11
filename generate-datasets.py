import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta

# =====================
# Cấu hình
# =====================
NUM_PATIENTS = 1000  # số bệnh nhân mô phỏng
DAYS = 180  # số ngày mỗi bệnh nhân
TRAIN_RATIO = 0.8  # Tỷ lệ dữ liệu cho tập train (0.8 nghĩa là 80% train, 20% test)
TRAIN_FILE = "data/synthetic_hypertension_train.csv"
TEST_FILE = "data/synthetic_hypertension_test.csv"
BATCH_SIZE = 50000  # ghi theo batch để tiết kiệm RAM

np.random.seed(42)
random.seed(42)

# Danh sách thuốc mẫu phong phú hơn cho bệnh cao huyết áp
medications = [
    ("Perindopril", 4),
    ("Perindopril", 8),
    ("Amlodipine", 5),
    ("Amlodipine", 10),
    ("Losartan", 50),
    ("Losartan", 100),
    ("Valsartan", 80),
    ("Valsartan", 160),
    ("Hydrochlorothiazide", 12.5),
    ("Hydrochlorothiazide", 25),
    ("Metoprolol", 50),
    ("Metoprolol", 100),
    ("Atenolol", 25),
    ("Atenolol", 50),
    ("Enalapril", 10),
    ("Enalapril", 20),
    ("Ramipril", 5),
    ("Ramipril", 10),
    ("Felodipine", 5),
    ("Felodipine", 10),
    ("Candesartan", 8),
    ("Candesartan", 16),
    ("Irbesartan", 150),
    ("Irbesartan", 300),
]

# =====================
# Sinh thông tin thói quen bệnh nhân
# =====================
patients = []
for pid in range(NUM_PATIENTS):
    age = int(np.random.randint(45, 85))
    gender = int(np.random.randint(0, 2))
    occupation = random.choice(["Retired", "Office", "Manual", "Unemployed"])
    income = random.choice(["Low", "Medium", "High"])
    living_status = random.choice(["Alone", "WithFamily"])
    years_htn = int(np.random.randint(1, 20))
    comorb = random.choice(["None", "Diabetes", "Heart Disease"])
    med_name, dose = random.choice(medications)
    scheduled_time = "08:00"

    salty_food_often = int(np.random.randint(0, 2))
    smoking = int(np.random.randint(0, 2))
    alcohol_habit = int(np.random.randint(0, 2))  # 0: ít, 1: hay uống
    caffeine_habit = int(np.random.randint(0, 2))  # 0: ít, 1: nhiều
    late_sleep_habit = int(np.random.randint(0, 2))

    patients.append(
        {
            "age": age,
            "gender": gender,
            "occupation": occupation,
            "income_level": income,
            "living_status": living_status,
            "years_with_hypertension": years_htn,
            "comorbidities": comorb,
            "medication_name": med_name,
            "dose_mg": int(dose),  # Đảm bảo int cho dose
            "scheduled_time": scheduled_time,
            "salty_food_often": salty_food_often,
            "smoking_status": smoking,
            "alcohol_habit": alcohol_habit,
            "caffeine_habit": caffeine_habit,
            "late_sleep_habit": late_sleep_habit,
        }
    )


# =====================
# Hàm sinh dữ liệu hàng ngày
# =====================
def simulate_day(patient):
    """
    Sinh dữ liệu cho một ngày của bệnh nhân dựa trên thói quen và yếu tố ngẫu nhiên.

    Args:
        patient (dict): Thông tin cố định của bệnh nhân.

    Returns:
        dict: Dữ liệu hàng ngày bao gồm việc uống thuốc, huyết áp, và các yếu tố khác.
    """
    # Yếu tố hàng ngày
    stress = int(np.random.randint(1, 10))
    bed_time_hour = 22 + np.random.randint(0, 3)  # 22h-24h
    if patient["late_sleep_habit"]:
        bed_time_hour += np.random.randint(0, 2)  # có thể đến 26h (tức 2h sáng hôm sau)
    # Chuẩn hóa về 00-23
    bed_time_hour = bed_time_hour % 24
    bed_time_minute = np.random.randint(0, 60)
    bed_time = f"{bed_time_hour:02d}:{bed_time_minute:02d}"

    late_night = (
        1 if bed_time_hour >= 23 or bed_time_hour < 4 else 0
    )  # Coi muộn nếu >=23h hoặc <4h
    sleep_hours = round(
        np.random.uniform(4.5, 8.5) - (late_night * np.random.uniform(0.5, 1.5)), 1
    )

    sodium = round(
        np.random.uniform(2.0, 6.0)
        + (patient["salty_food_often"] * np.random.uniform(1.0, 2.0)),
        2,
    )
    exercise = int(np.random.randint(0, 60))
    caffeine = int(
        np.random.randint(50, 250)
        if patient["caffeine_habit"]
        else np.random.randint(0, 80)
    )
    alcohol = int(
        np.random.randint(50, 200)
        if patient["alcohol_habit"]
        else np.random.randint(0, 30)
    )

    # Xác suất uống thuốc
    prob_taken = 0.92
    if stress > 7:
        prob_taken -= 0.15
    if sleep_hours < 6:
        prob_taken -= 0.1
    taken = int(1 if np.random.rand() < prob_taken else 0)

    time_taken = "None"  # Sử dụng 'None' thay vì bỏ trống
    if taken:
        delay = int(np.random.randint(-15, 120))  # trễ tối đa 2h
        sched_dt = datetime.strptime(patient["scheduled_time"], "%H:%M")
        actual_dt = sched_dt + timedelta(minutes=delay)
        time_taken = actual_dt.strftime("%H:%M")

    # Huyết áp (base + ảnh hưởng)
    base_sys = int(np.random.randint(120, 140))
    base_dia = int(np.random.randint(80, 90))
    sys = int(
        base_sys
        + (0 if taken else np.random.randint(5, 15))
        + int((stress - 5) * 1.5)
        + int((sodium - 3) * 2)
    )
    dia = int(
        base_dia
        + (0 if taken else np.random.randint(3, 8))
        + int((stress - 5) * 1.0)
        + int((sodium - 3) * 1.5)
    )

    return {
        **patient,
        "taken": taken,
        "time_taken": time_taken,
        "blood_pressure_systolic": sys,
        "blood_pressure_diastolic": dia,
        "stress_level": stress,
        "bed_time": bed_time,
        "late_night": late_night,
        "sleep_hours": sleep_hours,
        "sodium_intake_g": sodium,
        "exercise_minutes": exercise,
        "caffeine_mg": caffeine,
        "alcohol_ml": alcohol,
    }


# =====================
# Sinh dữ liệu & chia thành train/test
# =====================
# Kiểm tra và tạo thư mục 'data' nếu chưa tồn tại
if not os.path.exists("data"):
    os.makedirs("data")

all_rows = []
for patient in patients:
    for _ in range(DAYS):
        row = simulate_day(patient)
        all_rows.append(row)

# Trộn dữ liệu để ngẫu nhiên
random.shuffle(all_rows)

# Tính số dòng train/test dựa trên tỷ lệ
total_records = len(all_rows)
train_size = int(total_records * TRAIN_RATIO)
train_rows = all_rows[:train_size]
test_rows = all_rows[train_size:]

print(f"Tổng số bản ghi: {total_records}")
print(f"Tập train: {len(train_rows)} bản ghi")
print(f"Tập test: {len(test_rows)} bản ghi")


# Ghi file train
def write_csv_in_batches(rows, filename):
    """
    Ghi dữ liệu vào file CSV theo batch.

    Args:
        rows (list): Danh sách các hàng dữ liệu.
        filename (str): Đường dẫn file output.
    """
    header_written = False
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i : i + BATCH_SIZE]
        df = pd.DataFrame(batch)
        mode = "w" if not header_written else "a"
        df.to_csv(filename, index=False, header=not header_written, mode=mode)
        header_written = True


write_csv_in_batches(train_rows, TRAIN_FILE)
write_csv_in_batches(test_rows, TEST_FILE)

print(f"✅ Đã tạo {TRAIN_FILE} và {TEST_FILE}")
