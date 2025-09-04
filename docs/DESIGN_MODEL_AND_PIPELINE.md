## 2. Thiết kế mô hình & pipeline

### 2.1 Tóm tắt mục tiêu và ràng buộc

* Mục tiêu đo được:

  1. Dự báo xác suất **p(resp trong 2h)** cho từng nhắc.
  2. Chọn **kênh/thời điểm** để tối đa hóa tỷ lệ phản hồi (reward).
* KPI: ROC-AUC, PR-AUC, F1@τ\*, ECE (Expected Calibration Error), avg reward bandit.
* Ràng buộc: **DATA\_MODE=M2** (chỉ “nhắc–phản hồi”, không PII/PHI). **Tái lập** với `seed`. **ENV auto-switch**: LOCAL→SMALL, COLAB→LARGE. Thời gian chạy LOCAL <10′ CPU. COLAB <20′.

---

### 2.2 Dữ liệu và schema (M2)

**File**: `logs.csv`

| Trường                | Kiểu       | Ràng buộc hợp lệ                                 | Lý do                           |
| --------------------- | ---------- | ------------------------------------------------ | ------------------------------- |
| `user_pid`            | str        | UUID/`u_XX` duy nhất                             | Ẩn danh người dùng              |
| `ts_reminder`         | str        | UTC ISO8601, hậu tố `Z`                          | Chuẩn hóa thời gian             |
| `tz_offset`           | str        | `±HH:MM`                                         | Quy đổi giờ địa phương          |
| `channel`             | enum       | `{push,SMS,voice}`                               | Không nhạy cảm                  |
| `delivered`           | {0,1}      | 0⇒`responded_within_2h=0`                        | Không nhận thì không phản hồi   |
| `responded_within_2h` | {0,1}      | nhị phân                                         | Nhãn mục tiêu                   |
| `ack_latency_sec`     | float/NULL | =NULL nếu `responded_within_2h=0`; 30–3600 nếu 1 | Độ trễ chỉ tồn tại khi phản hồi |
| `snooze`              | {0,1}/NULL | tùy chọn                                         | Ghi nhận hoãn nhắc              |

**Bổ sung KHÔNG nhạy cảm (tùy chọn)**: `app_version`, `os_family`, `delivery_retry_count`, `quiet_hours_flag`, `net_type`.
**Kiểm tra hợp lệ**: định dạng thời gian, miền giá trị, đơn trị trên `(user_pid, ts_reminder)`, không trùng bản ghi, không chứa nội dung tin nhắn/vị trí/device-ID.

---

### 2.3 Hướng dẫn tạo dữ liệu synthetic (mã giả)

**Quy mô**: SMALL≈10–20×14–21 ngày (≈3k–8k). LARGE≈100–300×30–60 ngày (\~100k).
**Quy trình**:

1. **CTR theo kênh** \~ Beta(α,β) (phân bố trên \[0,1]).
2. **Nhịp sinh học** `g(hour)` (đỉnh sáng +10%, tối −5%).
3. **Hiệu ứng ngày** `h(weekday/weekend)` (cuối tuần −3%).
4. Xác suất: `p = clip(CTR_channel × g(hour) × h(dow), 0.05, 0.95)`.
5. Sinh nhãn: `responded_within_2h ~ Bernoulli(p)`. Nếu =1 thì `ack_latency_sec ~ LogNormal(μ,σ)`.

**Mã giả**:

```python
rng = np.random.default_rng(seed)
ctr = {"push": rng.beta(6,4), "SMS": rng.beta(5,5), "voice": rng.beta(4,6)}

def g_hour(h):
    return 1.10 if 5<=h<=11 else (0.95 if 18<=h<=22 else 1.0)

def h_dow(dow):  # 0=Mon..6=Sun
    return 0.97 if dow in [5,6] else 1.0

def p_resp(ch, h, dow):
    return float(np.clip(ctr[ch] * g_hour(h) * h_dow(dow), 0.05, 0.95))

# responded ~ Bernoulli(p)
# ack_latency_sec ~ LogNormal(mu=5.3, sigma=0.4) if responded==1 else NULL
```

**Mẫu dữ liệu (8 dòng)**:

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
```

**Vì sao đủ để kiểm thử công bằng theo trục hoạt động**: p phụ thuộc **kênh** và **khung giờ**, nên các nhóm hoạt động có phân phối nhãn khác nhau → đo TPR/FPR/ AUCPR/ECE theo nhóm.

---

### 2.4 Feature engineering (giải thích ngắn)

Bảng đặc trưng cốt lõi:

| Tên                          | Kiểu        | Công thức                             | Lý do                                |
| ---------------------------- | ----------- | ------------------------------------- | ------------------------------------ |
| `tod_sin`,`tod_cos`          | float       | `sin(2π·hour/24)`, `cos(2π·hour/24)`  | Mã hóa chu kỳ ngày (tránh biên 23→0) |
| `dow_0..6`,`is_weekend`      | one-hot,int | từ `ts_reminder`                      | Nhịp theo ngày                       |
| `gap_min`                    | float       | phút giữa sự kiện i và i−1 (per user) | Nhịp nhắc                            |
| `hist_k`                     | int         | tổng phản hồi 1..k bước trước (k=5)   | Thói quen gần                        |
| `ctr_7d_user`,`ctr_14d_user` | float       | mean rolling per user                 | Xu hướng người dùng                  |
| `ctr_user_channel`           | float       | rolling per user×channel              | Ảnh hưởng kênh cá nhân               |
| `latency_bucket`             | one-hot     | {≤5m,5–30m,>30m,none}                 | Tốc độ phản hồi                      |
| `exp_decay`                  | float       | `Σ responded(i−j)·λ^j`, λ=0.8         | Nhấn mạnh quá khứ gần                |
| `channel_*`                  | one-hot     | từ `channel`                          | Cho bandit/predictor                 |

Xử lý NA: điền median cho số, “none” cho bucket. Chuẩn hóa z-score cho số. Kiểm tra rò rỉ: không dùng trường hậu quả sau thời điểm dự báo.

---

### 2.5 Mô hình và công thức

* **Risk model — Logistic Regression (baseline)**: `p = σ(wᵀx + b)`; tối ưu log-loss; `class_weight="balanced"`.
* **TinyTemporal (tùy chọn, COLAB)**: TCN nhỏ (kernel=2, channels=16) hoặc Tiny-Transformer; dùng cửa sổ 4–8 bước.
* **Calibration**: Platt scaling (logistic trên logits) hoặc Isotonic; bật khi **ECE>0.05** trên validation.
* **Policy — Contextual bandit ε-greedy**: arms={push,SMS,voice}.

  * *Contextual bandit*: bài toán chọn hành động tối ưu dựa trên **ngữ cảnh** x hiện tại.
  * *ε-greedy*: với xác suất ε “thử” ngẫu nhiên (explore), còn lại chọn arm có **reward kỳ vọng** cao nhất (exploit).
  * Ước lượng reward offline bằng **outcome-model per-arm** (LR trên các hàng của arm đó).

---

### 2.6 Chia dữ liệu và huấn luyện

* **Split theo `user_pid`** → train/val/test (60/20/20). Không trộn user.
* **Lệch lớp**: `class_weight` hoặc chọn **τ\*** theo điểm F1 tối đa trên đường PR.
* **Siêu tham số**: `K_hist=5`, `λ=0.8`, `EPS_START=0.2`, `EPS_END=0.05`, `ε(t)=max(EPS_END, EPS_START/√t)`.
* **Artefact**: `model.pkl`, `scaler.pkl`, `feature_schema.json` (tên, dtype, nguồn, chuẩn hóa).

---

### 2.7 Suy luận và vận hành

* **Sinh slot “hôm nay”**: \[06:00..22:00], bước 30–60′, hoặc dùng lịch giả lập 2 slot/ngày (sáng/tối).
* Với mỗi slot: **build context** → **risk = LR(x)** → **policy chọn kênh** → ghi `ts_local, risk, channel_rec, note`.
* **Ghi chú điều hành**: giới hạn tần suất nhắc, chống lặp kênh, log tổng hợp (không PII). Fallback mặc định: `SMS` khi thiếu đặc trưng.

---

### 2.8 Sơ đồ pipeline (Mermaid)

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

---

### 2.9 Bảo mật, riêng tư, công bằng

* **Minimization**: chỉ M2; không nội dung tin nhắn, không vị trí, không device-ID; `user_pid` ẩn danh; consent; quy trình xóa sau chấm.
* **Fairness theo trục hoạt động**: báo cáo TPR/FPR, AUCPR, ECE theo **channel**, **khung giờ {05–11,12–17,18–22}**, **weekday/weekend**; theo dõi chênh lệch max–min.
* **Giới hạn**: không có nhân khẩu học → không đánh giá nhóm bảo vệ; ghi rõ trong Model/Data Card.
