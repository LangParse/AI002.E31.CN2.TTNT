## 2. Thiết kế mô hình & pipeline

### 2.1 Tóm tắt mục tiêu và ràng buộc

* **Mục tiêu**: dự báo xác suất *“uống trong 2 giờ”* cho từng nhắc (risk model) và tối ưu **kênh/thời điểm nhắc** bằng **contextual bandit** (bài toán chọn hành động dựa trên ngữ cảnh hiện tại).
* **Ràng buộc**: **DATA\_MODE=M2** (chỉ “nhắc–phản hồi”, không PII/PHI); **minimization dữ liệu**; **tái lập** với `seed`; **ENV auto-switch**:

  * **LOCAL/SMALL**: 10–20 user × 14–21 ngày, CPU, offline-friendly.
  * **COLAB/LARGE**: 100–300 user × 30–60 ngày, pip OK, GPU nếu có.

### 2.2 Dữ liệu và schema (M2)

* **File**: `logs.csv`
* **Trường**:
  `user_pid` (UUID ẩn danh), `ts_reminder` (UTC ISO8601), `tz_offset` (±HH\:MM),
  `channel` ∈ {push,SMS,voice}, `delivered` ∈ {0,1},
  `responded_within_2h` ∈ {0,1}, `ack_latency_sec` (nullable), `snooze` ∈ {0,1}.
* **Ràng buộc hợp lệ**:

  * Nếu `responded_within_2h=0` ⇒ `ack_latency_sec=null` (không có độ trễ khi không phản hồi).
  * `delivered=0` ⇒ có thể cho `responded_within_2h=0` (không nhận thì không thể phản hồi).
  * `snooze=1` ⇒ sự kiện được hoãn, vẫn giữ `responded_within_2h` theo thực tế.
    → Lý do: bảo toàn tính nhất quán và tránh “rò rỉ” thông tin không tồn tại.
* **Trường mở rộng KHÔNG nhạy cảm (tùy chọn)**: `app_version`, `os_family`, `delivery_retry_count`, `quiet_hours_flag`, `net_type`
  → Hữu ích cho chẩn đoán hệ thống; không nhận diện cá nhân.

### 2.3 Hướng dẫn tạo dữ liệu synthetic (có mã giả)

* **Quy mô**:

  * **SMALL (LOCAL)**: ≈10–20 user × 14–21 ngày ⇒ ≈3k–8k bản ghi.
  * **LARGE (COLAB)**: ≈100–300 user × 30–60 ngày ⇒ ≈100k± bản ghi.
* **Generator**:

  1. **Base CTR theo kênh** \~ Beta(α,β) (Beta: phân bố xác suất trên \[0,1]).
  2. **Nhịp sinh học** `g(hour)` có đỉnh sáng và xế chiều (hàm điều chế theo giờ).
  3. **Hiệu ứng ngày** `h(weekday/weekend)` giảm nhẹ cuối tuần.
  4. Xác suất phản hồi: `p = CTR_channel × g(hour) × h(weekday)` (clip về \[0.05,0.95]).
     Sinh `responded_within_2h ~ Bernoulli(p)`; nếu =1, sinh `ack_latency_sec ~ LogNormal(μ,σ)` (LogNormal: biến dương lệch phải).
* **Mã giả**:

```python
rng = np.random.default_rng(seed)
ctr_push = rng.beta(6,4); ctr_sms = rng.beta(5,5); ctr_voice = rng.beta(4,6)

def g_hour(h):
    # đỉnh sáng ~+10%, tối ~-5%
    bonus = 0.10 if 5 <= h <= 11 else (-0.05 if 18 <= h <= 22 else 0.0)
    return 1.0 + bonus

def h_weekday(dow):
    # cuối tuần -3%
    return 0.97 if dow in [5,6] else 1.0

def p_resp(channel, hour, dow):
    base = {"push": ctr_push, "SMS": ctr_sms, "voice": ctr_voice}[channel]
    return float(np.clip(base * g_hour(hour) * h_weekday(dow), 0.05, 0.95))

# responded ~ Bernoulli(p); nếu 1 thì ack_latency_sec ~ LogNormal(mu=5.3, sigma=0.4)  # ~150–300s
```

* **Vì sao đủ để kiểm thử công bằng theo *trục hoạt động***: p phụ thuộc **kênh** và **khung giờ** ⇒ tạo khác biệt hiệu năng giữa nhóm “hoạt động” (channel/time buckets). Không cần nhân khẩu học.

* **5–10 dòng mẫu**:

```csv
user_pid,ts_reminder,tz_offset,channel,delivered,responded_within_2h,ack_latency_sec,snooze
u_01,2025-08-01T06:30:00Z,+07:00,push,1,1,420,0
u_01,2025-08-01T21:00:00Z,+07:00,SMS,1,0,,0
u_02,2025-08-02T07:15:00Z,+07:00,voice,1,1,180,0
u_02,2025-08-02T18:30:00Z,+07:00,push,1,0,,0
u_03,2025-08-03T06:45:00Z,+07:00,SMS,1,1,300,0
u_03,2025-08-03T21:15:00Z,+07:00,voice,1,1,240,0
u_04,2025-08-04T07:00:00Z,+07:00,push,1,0,,0
u_04,2025-08-04T19:45:00Z,+07:00,SMS,1,1,210,0
u_05,2025-08-05T06:30:00Z,+07:00,voice,1,0,,0
u_05,2025-08-05T20:30:00Z,+07:00,push,1,1,150,0
```

### 2.4 Feature engineering (giải thích ngắn từng thuật ngữ)

* **Mã hóa thời gian trong ngày**: `tod_sin = sin(2π·hour/24)`, `tod_cos = cos(2π·hour/24)`
  *(mã hóa chu kỳ; tránh biên 23→0 bị xa nhau)*.
* **Ngày trong tuần**: one-hot `dow_0..dow_6`; **`is_weekend`** ∈ {0,1}.
* **Khoảng cách nhắc** `gap_min`: phút giữa sự kiện i và i−1 theo user *(đặc trưng nhịp)*.
* **Lịch sử k bước**: tổng `responded_within_2h` của 1..k lần gần nhất *(k=5)*.
* **CTR trượt theo user**: `ctr_7d_user`, `ctr_14d_user`; **theo kênh**: `ctr_user_channel` *(tỷ lệ phản hồi gần đây)*.
* **Độ trễ** `latency_bucket` ∈ {≤5m, 5–30m, >30m, none} *(phân loại phản hồi nhanh/chậm)*.
* **Suy giảm theo thời gian**: `exp_decay = Σ responded(i−j)·λ^j`, λ=0.8 *(nhấn mạnh tín hiệu gần đây)*.
* **One-hot `channel`**: push/SMS/voice.

### 2.5 Mô hình và công thức

* **Risk model — Logistic Regression (LR)**: `p = σ(wᵀx + b)` với `σ` là sigmoid *(hồi quy logistic dự báo xác suất)*.
  **TinyTemporal** (tùy chọn khi COLAB): **TCN** hoặc **TinyTransformer** nhỏ *(mạng chuỗi thời gian gọn)*.
* **Calibration**: **Platt scaling** (hồi quy logistic trên logits) hoặc **Isotonic** (hồi quy đơn điệu) để giảm **ECE** *(Expected Calibration Error: sai lệch giữa xác suất dự báo và tần suất thực)*.
* **Policy — Contextual bandit ε-greedy**: arms={push,SMS,voice}; **ε-greedy**: xác suất ε để *thử* ngẫu nhiên (explore), còn lại *khai thác* arm tốt nhất (exploit).

### 2.6 Chia dữ liệu và huấn luyện

* **Split theo `user_pid`**: train/val/test để tránh rò rỉ giữa người dùng.
* **Class imbalance**: `class_weight=balanced` hoặc chọn ngưỡng theo đường PR.
* **Siêu tham số mặc định**: `K_hist=5`, `λ=0.8`, `EPS_START=0.2 → EPS_END=0.05`, `EPS_DECAY≈1/√t`.
* **Lưu artefact**: `model.pkl`, `scaler.pkl`, `feature_schema.json`.

### 2.7 Suy luận và vận hành

* **Sinh lưới slot “hôm nay”**: \[06:00..22:00], bước 30–60′, hoặc lấy lịch dùng giả lập.
* Mỗi slot: build **context** → tính **risk** bằng LR → **policy** chọn `channel` → ghi lịch đề xuất.
* **Bảng xuất**: `ts_local, risk, channel_rec, note` (note nêu lý do ngắn: high-risk morning → push).

### 2.8 Sơ đồ pipeline (Mermaid — bắt buộc)

```mermaid
flowchart LR
  A[logs.csv (M2)] --> B[Validate + Timezone normalize]
  B --> C[Feature builder]
  C --> D1[Risk model: LR + TinyTemporal (opt)]
  C --> D2[Policy: Contextual bandit ε-greedy]
  E[prescription_demo.json] --> D3[DDI rules (toy)]
  D1 & D2 & D3 --> F[Inference: lịch nhắc hôm nay]
  F --> G[Giám sát, Calibration, Fairness, Stress tests]
```

### 2.9 Bảo mật, riêng tư, công bằng

* **Minimization**: không PII/PHI; `user_pid` ẩn danh; consent; quy trình xóa dữ liệu sau chấm.
* **Công bằng theo *hoạt động***: so sánh theo `channel`, khung giờ {05–11,12–17,18–22}, weekday/weekend; báo cáo TPR/FPR, AUCPR, ECE.
* **Giới hạn**: không có nhân khẩu học ⇒ không đánh giá theo nhóm bảo vệ; quyết định có chủ đích để giảm rủi ro riêng tư.

---

## 3. Kết quả thử nghiệm

### 3.1 Thiết lập thí nghiệm

* **ENV auto-switch**: LOCAL/SMALL và COLAB/LARGE; `seed` cố định.
* **Split** theo `user_pid`. **Epoch** ít, batch nhỏ để **<10′** LOCAL, **<20′** COLAB.
* **Calibration** bật Platt nếu ECE > ngưỡng (ví dụ 0.05).

### 3.2 Chỉ số đánh giá (giải thích)

* **AUC (ROC-AUC)**: diện tích dưới đường ROC.
* **PR-AUC**: phù hợp khi lệch lớp (nhấn mạnh positive).
* **F1**: trung hòa precision/recall.
* **Calibration/ECE**: độ tin cậy xác suất; ECE càng nhỏ càng tốt.
* **CTR/Reward bandit**: trung bình `responded_within_2h` theo thời gian.
* **TPR/FPR**: tỉ lệ dương tính thật/giả ở ngưỡng 0.5.

### 3.3 Bảng kết quả chính

*(giá trị minh họa từ dữ liệu synthetic; thay bằng số thực tế khi chạy notebook)*

| ENV / Model                |  AUC | PR-AUC | F1\@0.5 |  ECE | Ghi chú             |
| -------------------------- | ---: | -----: | ------: | ---: | ------------------- |
| LOCAL/SMALL — LR           | 0.78 |   0.73 |    0.70 | 0.04 | đã calibrate        |
| LOCAL/SMALL — TinyTemporal | 0.80 |   0.75 |    0.71 | 0.05 | opt, có GPU tốt hơn |
| COLAB/LARGE — LR           | 0.79 |   0.74 |    0.71 | 0.03 | đã calibrate        |
| COLAB/LARGE — TinyTemporal | 0.82 |   0.77 |    0.73 | 0.04 | opt                 |

**Bandit (ε-greedy vs uniform)**

| ENV         | Avg reward ε-greedy | Avg reward uniform | Uplift | Phân bổ kênh (ε-greedy)      |
| ----------- | ------------------: | -----------------: | -----: | ---------------------------- |
| LOCAL/SMALL |                0.64 |               0.58 |  +0.06 | push 45%, SMS 35%, voice 20% |
| COLAB/LARGE |                0.66 |               0.60 |  +0.06 | push 44%, SMS 38%, voice 18% |

**Tài nguyên**

| ENV         | Thời gian chạy | RAM ước lượng |
| ----------- | -------------: | ------------: |
| LOCAL/SMALL |      < 10 phút |        < 2 GB |
| COLAB/LARGE |      < 20 phút |     \~ 3–4 GB |

### 3.4 Fairness theo hoạt động

*(test set; ngưỡng 0.5; minh họa)*

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

**Chênh lệch**: max–min *(TPR ≈ 0.12, FPR ≈ 0.05, ΔAUCPR ≈ 0.11, ΔECE ≈ 0.02)*.
**Nhận xét**: *voice* và *evening* yếu hơn; do base CTR thấp và nhiễu hành vi tối.

### 3.5 Stress tests

*(minh họa; báo cáo trước/sau)*

| Kịch bản       |  AUC | PR-AUC |   F1 |  ECE |
| -------------- | ---: | -----: | ---: | ---: |
| Baseline test  | 0.79 |   0.74 | 0.71 | 0.03 |
| Drop 20%       | 0.77 |   0.72 | 0.69 | 0.04 |
| Shift +2h      | 0.76 |   0.71 | 0.68 | 0.05 |
| Label noise 5% | 0.75 |   0.70 | 0.67 | 0.05 |

**Độ bền**: giảm dần nhưng chấp nhận; shift giờ ảnh hưởng ECE.

### 3.6 Hình ảnh bắt buộc

* `roc.png` (đường ROC; càng gần góc trái trên càng tốt).
* `pr.png` (đường Precision–Recall; diện tích càng lớn càng tốt).
* `reliability.png` (đường hiệu chuẩn; bám đường chéo y=x là tốt).
* `bandit_reward.png` (trung bình reward tích lũy; đường ε-greedy nằm trên uniform là tốt).

### 3.7 Mẫu I/O

**Mẫu `logs.csv`**: xem 2.3.
**Lịch nhắc “hôm nay” (mẫu)**:

| ts\_local        | risk | channel\_rec | note                |
| ---------------- | ---: | -----------: | ------------------- |
| 2025-09-03 06:30 | 0.62 |         push | high-risk morning   |
| 2025-09-03 21:00 | 0.34 |          SMS | medium-risk evening |

---

## 4. Đánh giá và tư duy phản biện

### 4.1 Phân tích lỗi

* **Sai dương (FP)**: evening + voice; nguyên nhân: base CTR thấp, mẫu ít.
* **Sai âm (FN)**: morning nhưng lịch sử k bước kém; thiếu bối cảnh “đột biến”.
* **Calibration**: Platt giảm ECE → ngưỡng quyết định ổn định hơn, ít over- or under-alert.

### 4.2 Công bằng, đạo đức, riêng tư

* **Data Card (M2)**: chỉ “nhắc–phản hồi”, không nội dung tin nhắn, không vị trí, không device ID; `user_pid` ẩn danh; consent; xóa sau chấm.
* **Model Card (ngắn)**: mục tiêu ước lượng risk 24–72h; giới hạn do thiếu nhân khẩu học; fairness báo cáo theo *hoạt động*; **không thay thế tư vấn y tế**.
* **Rủi ro thao tác kênh**: nếu cố “ép” push, người dùng có thể mệt mỏi; giới hạn tần suất nhắc, hỗn hợp kênh, kiểm tra định kỳ.

### 4.3 Hạn chế và rủi ro

* Không có nhân khẩu học ⇒ không đánh giá *group fairness* cổ điển; synthetic có thể lệch giả.
* Bandit **offline replay** không phản ánh đầy đủ tương tác người dùng sống.
* Nguy cơ **over-alert**; hành vi trôi dạt theo mùa, kỳ nghỉ.

### 4.4 Hướng cải tiến

* Dữ liệu phong phú hơn (FHIR/OMOP), wearable thật; **federated + DP** để bảo vệ riêng tư.
* Bandit tốt hơn: **Thompson Sampling**, **LinUCB** (học tuyến tính theo ngữ cảnh).
* A/B test online, giám sát drift và tái hiệu chuẩn định kỳ.

### 4.5 Decision log

1. **DATA minimization**: chỉ M2 để giảm rủi ro riêng tư.
2. **Split theo user**: tránh rò rỉ, phản ánh triển khai thực.
3. **LR baseline + calibration**: nhanh, tin cậy xác suất, dễ vận hành LOCAL.
4. **Bandit ε-greedy**: đơn giản, ổn định; dễ mở rộng Thompson/LinUCB.
5. **Fairness theo hoạt động**: phù hợp M2; đo được tác động kênh/giờ.
6. **TinyTemporal tùy chọn**: chỉ bật khi tài nguyên cho phép.
7. **Stress tests**: bắt buộc để kiểm tra độ bền trước thiếu dữ liệu, lệch giờ, nhiễu nhãn.
