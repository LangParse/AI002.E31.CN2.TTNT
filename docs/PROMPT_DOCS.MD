# 1. Giới thiệu

Vai trò: Bạn là kỹ sư ML + technical writer. Soạn thảo tài liệu báo cáo học phần cho đề tài
“Trợ lý AI theo dõi dùng thuốc và nhắc nhở liều dùng cá nhân hóa”.
Viết bằng tiếng Việt, rõ ràng, có ví dụ, có giải thích thuật ngữ ngay khi xuất hiện.
Cố định DATA_MODE = M2 (chỉ “nhắc–phản hồi”, không PII/PHI). Hỗ trợ LOCAL/SMALL và COLAB/LARGE.

ĐẦU RA BẮT BUỘC: soạn **đúng 3 mục lớn** sau đây, theo cấu trúc chi tiết, kèm sơ đồ Mermaid, bảng, và mẫu dữ liệu.
(Đừng viết phần khác ngoài 3 mục yêu cầu.)

# 2. Thiết kế mô hình & pipeline

2.1 Tóm tắt mục tiêu và ràng buộc  
- Mục tiêu: dự báo xác suất “uống trong 2 giờ” và tối ưu kênh/thời điểm nhắc.  
- Ràng buộc: minimization dữ liệu; tái lập; ENV auto-switch (LwOCAL/SMALL, COLAB/LARGE).

2.2 Dữ liệu và schema (M2)  
- File: `logs.csv`. Trường:  
  `user_pid` (UUID ẩn danh), `ts_reminder` (UTC ISO8601), `tz_offset` (±HH:MM),  
  `channel` ∈ {push,SMS,voice}, `delivered` ∈ {0,1},  
  `responded_within_2h` ∈ {0,1}, `ack_latency_sec` (nullable), `snooze` ∈ {0,1}.  
- Ràng buộc hợp lệ (giải thích vì sao): nếu `responded_within_2h=0` → `ack_latency_sec=null`, v.v.  
- Trường mở rộng KHÔNG nhạy cảm (tùy chọn): `app_version`, `os_family`, `delivery_retry_count`, `quiet_hours_flag`, `net_type`.

2.3 Hướng dẫn tạo dữ liệu synthetic (có mã giả)  
- Quy mô: SMALL (LOCAL) ≈ 10–20 user × 14–21 ngày (3k–8k records); LARGE (COLAB) ≈ 100–300 user × 30–60 ngày (~100k).  
- Generator:  
  1) Khởi tạo base CTR theo kênh từ Beta(α,β).  
  2) Hàm “nhịp sinh học” g(hour) tạo đỉnh buổi sáng/tối.  
  3) Xác suất phản hồi = CTR_channel × g(hour) × h(weekday/weekend).  
  4) `responded_within_2h ~ Bernoulli(p)`; nếu 1 thì `ack_latency_sec ~ LogNormal(μ,σ)`, else null.  
- Đặt `seed` để tái lập. Xuất 5–10 dòng mẫu ngay trong tài liệu.  
- Giải thích tại sao mô hình tạo dữ liệu này đủ để kiểm thử công bằng theo **trục hoạt động**.

2.4 Feature engineering (giải thích ngắn từng thuật ngữ)  
- Mã hóa thời gian trong ngày: `tod_sin = sin(2π·hour/24)`, `tod_cos = cos(2π·hour/24)`  
  *Giải thích*: biểu diễn chu kỳ ngày, tránh biên 23→0.  
- Ngày trong tuần: one-hot; `is_weekend`.  
- Khoảng cách nhắc: `gap_min` giữa sự kiện i và i−1 của cùng user.  
- Lịch sử k bước: tổng phản hồi 1..k bước trước (*k* mặc định 5).  
- CTR trượt theo user: `ctr_7d_user`, `ctr_14d_user`; theo kênh: `ctr_user_channel`.  
- Độ trễ: `latency_bucket` ∈ {≤5m, 5–30m, >30m, none}.  
- Suy giảm theo thời gian: `exp_decay = Σ responded(i−j)·λ^j` (λ mặc định 0.8).  
- One-hot `channel`.

2.5 Mô hình và công thức  
- **Risk model**: Logistic Regression (baseline) → xác suất `p = σ(wᵀx + b)`;  
  tùy chọn **TinyTemporal** (TCN/TinyTransformer nhỏ) khi COLAB.  
- **Calibration**: Platt scaling hoặc Isotonic để độ tin cậy tốt hơn.  
- **Policy**: Contextual bandit ε-greedy; arms = {push,SMS,voice};  
  *Giải thích “bandit” và “ε-greedy”*: thuật toán thử–khai thác, ε là xác suất thử ngẫu nhiên.

2.6 Chia dữ liệu và huấn luyện  
- Split theo `user_pid` thành train/val/test để tránh rò rỉ.  
- Class imbalance: dùng `class_weight=balanced` hoặc điều chỉnh ngưỡng theo PR.  
- Siêu tham số mặc định: `K_hist=5`, `λ=0.8`, `EPS_START=0.2 → EPS_END=0.05`, `EPS_DECAY≈1/√t`.  
- Lưu `model.pkl`, `scaler.pkl`, `feature_schema.json`.

2.7 Suy luận và vận hành  
- Sinh lưới slot giờ “hôm nay” [06:00..22:00] bước 30–60′ hoặc lấy từ lịch dùng giả lập.  
- Với mỗi slot: build context → tính `risk` → policy chọn `channel` → ghi lịch đề xuất.  
- Xuất bảng: `ts_local, risk, channel_rec, note`.

2.8 Sơ đồ pipeline (Mermaid — bắt buộc chèn code)  
```mermaid
flowchart LR
  A[logs.csv (M2)] --> B[Validate + Timezone normalize]
  B --> C[Feature builder]
  C --> D1[Risk model: LR + TinyTemporal (opt)]
  C --> D2[Policy: Contextual bandit ε-greedy]
  E[prescription_demo.json] --> D3[DDI rules (toy)]
  D1 & D2 & D3 --> F[Inference: lịch nhắc hôm nay]
  F --> G[Giám sát, Calibration, Fairness, Stress tests]
````

2.9 Bảo mật, riêng tư, công bằng

* Minimization: không thu PII/PHI; ẩn danh `user_pid`; consent; quy trình xóa dữ liệu.
* Công bằng theo **hoạt động**: so sánh theo `channel`, khung giờ {05–11,12–17,18–22}, weekday/weekend.
* Nêu giới hạn: không có nhân khẩu học nên không đánh giá theo nhóm bảo vệ.

---

# 3. Kết quả thử nghiệm

3.1 Thiết lập thí nghiệm

* ENV auto-switch: LOCAL/SMALL và COLAB/LARGE; seed cố định.
* Chia train/val/test theo user. Số vòng lặp và batch nhỏ để dưới 10′ LOCAL, 20′ COLAB.

3.2 Chỉ số đánh giá (giải thích thuật ngữ)

* **AUC**: diện tích ROC; **PR-AUC**: phù hợp dữ liệu lệch lớp; **F1**: cân bằng precision/recall.
* **Calibration** và **ECE**: độ khớp xác suất dự báo với tần suất thực tế.
* **CTR/Reward bandit**: tỷ lệ phản hồi trung bình theo thời gian.
* **TPR/FPR**: tỉ lệ dương tính thật/giả theo ngưỡng.

3.3 Bảng kết quả chính (yêu cầu chèn bảng Markdown)

* So sánh Baseline LR vs TinyTemporal (nếu bật) trên SMALL và LARGE: AUC, PR-AUC, F1, ECE.
* Bandit: đường reward theo epoch, phân bổ kênh, uplift so với uniform.
* Thời gian chạy và RAM ước lượng.

3.4 Fairness theo hoạt động

* Bảng chỉ số theo nhóm: {channel}, {time bucket}, {weekday/weekend}.
* Báo cáo chênh lệch: max–min TPR/FPR, ΔAUCPR, ΔECE.
* Nhận xét ngắn: nhóm nào yếu, nguyên nhân khả dĩ.

3.5 Stress tests

* Drop 20% sự kiện; Shift múi giờ +2h; Noise nhãn 5%.
* Bảng trước/sau cho AUC, PR-AUC, F1, ECE; nhận xét độ bền.

3.6 Hình ảnh bắt buộc

* `roc.png`, `pr.png`, `reliability.png`, `bandit_reward.png`.
* Mô tả ngắn cách đọc mỗi hình.

3.7 Mẫu I/O

* 5–10 dòng `logs.csv` mẫu.
* Bảng “lịch nhắc hôm nay” mẫu: `ts_local, risk, channel_rec`.

---

# 4. Đánh giá và tư duy phản biện

4.1 Phân tích lỗi

* Top-3 mẫu dự báo sai; lý do (thiếu lịch sử, khung giờ nhiễu, kênh kém hiệu quả).
* Ảnh hưởng của calibration đến quyết định nhắc.

4.2 Công bằng, đạo đức, riêng tư

* Tóm Data Card (M2) và Model Card ngắn.
* Công bằng theo hoạt động: kết quả, khoảng tin cậy, rủi ro thao tác kênh.
* Riêng tư: minimization, ẩn danh, consent, quy trình xóa.
* Tuyên bố “không thay thế tư vấn y tế”.

4.3 Hạn chế và rủi ro

* Không có nhân khẩu học → giới hạn phân tích fairness; synthetic bias; replay offline cho bandit.
* Nguy cơ cảnh báo quá mức; drift hành vi theo mùa/đổi lịch.

4.4 Hướng cải tiến

* Dữ liệu phong phú hơn (FHIR/OMOP), wearables thật, federated + DP.
* Bandit tốt hơn: Thompson Sampling, LinUCB.
* Quan trắc dài hạn và A/B test.

4.5 Decision log

* Ghi 5–7 quyết định thiết kế quan trọng và trade-off.

PHỤ LỤC — Giải thích thuật ngữ nhanh (chèn trong tài liệu khi lần đầu xuất hiện)

* **Contextual bandit**: bài toán chọn hành động tối ưu dựa trên ngữ cảnh hiện tại.
* **ε-greedy**: chọn ngẫu nhiên với xác suất ε để khám phá; còn lại khai thác phương án tốt nhất hiện có.
* **Calibration / ECE**: hiệu chỉnh xác suất; ECE đo khoảng cách giữa xác suất dự báo và tần suất thật.
* **PR-AUC**: diện tích dưới đường Precision–Recall, ổn khi dữ liệu lệch lớp.
* **Platt/Isotonic**: hai kỹ thuật hiệu chỉnh xác suất đầu ra mô hình.
* **Equalized Odds (tham khảo)**: chênh lệch TPR/FPR giữa nhóm; ở M2, thay nhóm bảo vệ bằng nhóm hoạt động.

YÊU CẦU ĐỊNH DẠNG

* Dùng Markdown H2/H3 đúng tiêu đề 2–4.
* Bảng Markdown, code block cho Mermaid và mẫu CSV.
* Viết giải thích ngắn gọn ngay sau mỗi thuật ngữ chuyên môn.
* Không đưa PII/PHI. Không chèn nội dung ngoài phạm vi 2–4.
* Có thể chia ra nhiều file Markdown nếu cần.
