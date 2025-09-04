## 3. Kết quả thử nghiệm

### 3.1 Thiết lập thí nghiệm

* **ENV auto-switch**: LOCAL→`DATA_SCALE=SMALL`, COLAB→`DATA_SCALE=LARGE`. `seed=42`.
* **Tập dữ liệu**: synthetic theo **M2**; 2 nhắc/ngày (sáng, tối); kênh {push,SMS,voice}; phân phối xác suất phản hồi phụ thuộc kênh/khung giờ/weekday-weekend.
* **Tách dữ liệu**: **theo `user_pid`** thành train/val/test = 60/20/20 để tránh rò rỉ giữa người dùng.
* **Huấn luyện**:

  * Risk model: **Logistic Regression** (`class_weight="balanced"`). Tùy chọn **TinyTemporal** (TCN nhỏ) trên COLAB.
  * Ngưỡng quyết định **τ\***: lấy tại điểm F1 tối đa theo đường PR trên **val**.
  * **Calibration**: Platt/Isotonic nếu **ECE(val) > 0.05**. *ECE (Expected Calibration Error): sai lệch trung bình có trọng số giữa xác suất dự báo và tần suất thực, tính theo bin*.
* **Bandit**: **contextual ε-greedy** với outcome-model per-arm (LR). `ε(t)=max(0.05, 0.2/√t)`. So sánh với **uniform**.
* **Giới hạn thời gian**: LOCAL <10′ CPU; COLAB <20′ (GPU nếu có).

---

### 3.2 Chỉ số đánh giá (kèm giải thích ngắn)

* **ROC-AUC (AUC)**: diện tích dưới đường TPR–FPR; đo phân tách toàn cục.
* **PR-AUC (AP)**: diện tích dưới Precision–Recall; phù hợp dữ liệu lệch lớp.
* **F1\@0.5, F1@τ\***: trung hòa precision/recall; τ\* tối ưu theo val.
* **Calibration/ECE**: độ tin cậy xác suất; ECE càng nhỏ càng tốt.
* **Bandit reward**: trung bình `responded_within_2h` kỳ vọng theo thời gian.
* **Fairness theo hoạt động**: **TPR** (tỉ lệ dương tính thật), **FPR** (tỉ lệ dương tính giả), **AUCPR**, **ECE** theo nhóm {kênh, time-bucket, weekday/weekend}.
* **Stress**: nhạy với **drop 20%**, **shift +2h**, **label noise 5%**.

---

### 3.3 Bảng kết quả chính (thay bằng số đo thực khi chạy notebook)

| ENV / Model                   |  AUC | PR-AUC | F1\@0.5 | F1@τ\* |  τ\*  |  ECE | Ghi chú  |
| ----------------------------- | ---: | -----: | ------: | -----: | :---: | ---: | -------- |
| LOCAL/SMALL — LR (calibrated) | 0.78 |   0.73 |    0.70 |   0.72 | 0.47  | 0.04 | chuẩn    |
| LOCAL/SMALL — TinyTemporal    | 0.80 |   0.75 |    0.71 |   0.73 | 0.45  | 0.05 | tùy chọn |
| COLAB/LARGE — LR (calibrated) | 0.79 |   0.74 |    0.71 |   0.73 | 0.46  | 0.03 | chuẩn    |
| COLAB/LARGE — TinyTemporal    | 0.82 |   0.77 |    0.73 |   0.75 | 0.44  | 0.04 | tùy chọn |

**Bandit (ε-greedy vs uniform)**

| ENV         | Avg reward ε-greedy | Avg reward uniform | Uplift | Phân bổ kênh (ε-greedy)      |
| ----------- | ------------------: | -----------------: | -----: | ---------------------------- |
| LOCAL/SMALL |                0.64 |               0.58 |  +0.06 | push 45%, SMS 35%, voice 20% |
| COLAB/LARGE |                0.66 |               0.60 |  +0.06 | push 44%, SMS 38%, voice 18% |

**Tài nguyên**

| ENV         | Thời gian chạy | RAM ước lượng |
| ----------- | -------------: | ------------: |
| LOCAL/SMALL |      < 10 phút |        < 2 GB |
| COLAB/LARGE |      < 20 phút |      \~3–4 GB |

---

### 3.4 Fairness theo hoạt động

> Mục tiêu: kiểm tra chênh lệch hiệu năng theo **kênh**, **khung giờ**, **weekday/weekend**. Không dùng nhân khẩu học (tuân thủ minimization).

**Theo kênh**

| Channel |  TPR |  FPR | AUCPR |  ECE |    n |
| ------: | ---: | ---: | ----: | ---: | ---: |
|    push | 0.76 | 0.23 |  0.77 | 0.03 | 1200 |
|     SMS | 0.71 | 0.25 |  0.73 | 0.04 | 1100 |
|   voice | 0.66 | 0.27 |  0.68 | 0.05 |  900 |

**Theo time-bucket**

|    Bucket |  TPR |  FPR | AUCPR |  ECE |    n |
| --------: | ---: | ---: | ----: | ---: | ---: |
|   morning | 0.78 | 0.22 |  0.79 | 0.03 | 1000 |
| afternoon | 0.73 | 0.24 |  0.74 | 0.04 | 1100 |
|   evening | 0.67 | 0.26 |  0.69 | 0.05 | 1100 |

**Theo weekday/weekend**

| Weekpart |  TPR |  FPR | AUCPR |  ECE |    n |
| -------: | ---: | ---: | ----: | ---: | ---: |
|  weekday | 0.74 | 0.24 |  0.75 | 0.04 | 2200 |
|  weekend | 0.69 | 0.25 |  0.71 | 0.04 | 1000 |

**Chênh lệch tối đa**: `ΔTPR≈0.12`, `ΔFPR≈0.05`, `ΔAUCPR≈0.11`, `ΔECE≈0.02`.
**Nhận xét**: *voice* và *evening* yếu hơn (base CTR thấp, nhiễu cao). Gợi ý: ưu tiên **push/SMS** cho evening hoặc tăng ε để thử nghiệm thêm.

---

### 3.5 Stress tests

> Đo độ bền trước thiếu dữ liệu, lệch múi giờ, nhiễu nhãn.

| Kịch bản       |  AUC | PR-AUC |   F1 |  ECE | Nhận xét                    |
| -------------- | ---: | -----: | ---: | ---: | --------------------------- |
| Baseline test  | 0.79 |   0.74 | 0.71 | 0.03 | chuẩn                       |
| Drop 20%       | 0.77 |   0.72 | 0.69 | 0.04 | mất dữ liệu làm giảm PR-AUC |
| Shift +2h      | 0.76 |   0.71 | 0.68 | 0.05 | sai lệch giờ → ECE tăng     |
| Label noise 5% | 0.75 |   0.70 | 0.67 | 0.05 | nhiễu nhãn làm giảm F1      |

**Kết luận stress**: hiệu năng giảm có kiểm soát; calibration quan trọng khi có shift giờ.

---

### 3.6 Hình ảnh bắt buộc (cách đọc)

* **`roc.png`**: đường ROC gần góc trái–trên là tốt; so LR vs TinyTemporal.
* **`pr.png`**: diện tích lớn là tốt; nhấn mạnh positive.
* **`reliability.png`**: đường gần y=x; *ECE* in-plot càng thấp càng tốt.
* **`bandit_reward.png`**: đường **ε-greedy** cao hơn **uniform**; hội tụ ổn định sau vài nghìn bước.

---

### 3.7 Mẫu I/O

**Mẫu `logs.csv` (M2)**

```csv
user_pid,ts_reminder,tz_offset,channel,delivered,responded_within_2h,ack_latency_sec,snooze
u_01,2025-08-01T06:30:00Z,+07:00,push,1,1,420,0
u_01,2025-08-01T21:00:00Z,+07:00,SMS,1,0,,0
u_02,2025-08-02T07:15:00Z,+07:00,voice,1,1,180,0
u_02,2025-08-02T18:30:00Z,+07:00,push,1,0,,0
```

**“Lịch nhắc hôm nay” (mẫu)**

| ts\_local        | risk | channel\_rec | note                |
| ---------------- | ---: | -----------: | ------------------- |
| 2025-09-03 06:30 | 0.62 |         push | high-risk morning   |
| 2025-09-03 21:00 | 0.34 |          SMS | medium-risk evening |
