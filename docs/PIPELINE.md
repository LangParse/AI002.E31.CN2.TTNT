# Phụ lục A — Đặc tả dữ liệu M2 và pipeline (chỉ “nhắc–phản hồi”)

## A.1 Vì sao schema tối giản vẫn đủ?
- Mục tiêu dự án: dự đoán `responded_within_2h` và chọn kênh/thời điểm. Không cần chẩn đoán, thuốc, hay PII.
- Từ `ts_reminder`, `tz_offset`, `channel`, `responded_within_2h` có thể tạo đặc trưng theo thời gian, lịch sử, và hành vi.
- Thiết kế “data minimization”: giảm rủi ro pháp lý, tăng khả năng tái lập, vẫn hỗ trợ đánh giá công bằng theo **trục hoạt động**.

## A.2 Schema bắt buộc (logs.csv)
- user_pid: mã ẩn danh (UUID4 ngắn, ví dụ “u_01”).
- ts_reminder: thời điểm nhắc (UTC ISO8601).
- tz_offset: ví dụ “+07:00”.
- channel: {push, SMS, voice}.
- delivered: {0,1}.
- responded_within_2h: {0,1}.
- ack_latency_sec: số giây đến khi xác nhận (null nếu không xác nhận).
- snooze: {0,1}.

Ràng buộc hợp lệ:
- Nếu responded_within_2h=0 → ack_latency_sec phải null.
- channel ∈ {push,SMS,voice}. tz_offset theo định dạng ±HH:MM.
- ts_reminder tăng theo thời gian trong cùng user_pid (không bắt buộc tuyệt đối nhưng nên).

## A.3 Trường mở rộng KHÔNG nhạy cảm (tùy chọn)
- app_version, os_family ∈ {ios, android, web} (phân tích lỗi, không là PII).
- delivery_retry_count (chất lượng kênh).
- quiet_hours_flag ∈ {0,1} (chế độ yên lặng hệ thống, không phải lịch cá nhân).
- net_type ∈ {wifi,cellular,unknown} (chỉ mức tổng quát).

## A.4 Suy diễn đặc trưng từ schema tối giản
Ký hiệu: với sự kiện i của user u tại thời điểm địa phương tᵢ (đã cộng tz_offset).

- Mã hoá thời gian trong ngày:
  - hour = hour(tᵢ);  tod_sin = sin(2π·hour/24);  tod_cos = cos(2π·hour/24)
- Ngày trong tuần: one-hot(dow ∈ {Mon..Sun})
- Khoảng cách giữa nhắc: gap_min = (tᵢ − tᵢ₋₁)/60
- Lịch sử gần k bước: hist_k = Σ_{j=1..k} responded_within_2hᵢ₋ⱼ
- CTR trượt theo user:
  - ctr_7d_user = (#resp trong 7d)/(#nhắc 7d); tương tự ctr_14d_user
- Thống kê theo kênh:
  - ctr_user_channel = mean(responded_within_2h | user, channel)
- Độ trễ xác nhận:
  - latency_bucket ∈ {≤5m, 5–30m, >30m, none}
- Weekpart:
  - is_weekend ∈ {0,1}
- Recency:
  - exp_decay = Σ_j responded_within_2hᵢ₋ⱼ · λ^j, λ∈(0,1)

Mô hình điểm rủi ro (ví dụ LR baseline):
- x = concat(tod_sin,tod_cos,dow,gap_min,hist_k,ctr_7d_user,ctr_14d_user,ctr_user_channel,is_weekend,latency_bucket_onehot,channel_onehot)
- p(risk) = σ(wᵀx + b)

## A.5 Luồng pipeline end-to-end

1) **Ingest & Validate**
   - Đọc logs.csv → kiểm tra schema và ràng buộc.
   - Chuẩn UTC → local bằng tz_offset. Bỏ bản ghi lỗi.

2) **Feature Engineering**
   - Tạo đặc trưng A.4. Chuẩn hoá/scale cần thiết. Lưu schema_features.json.

3) **Split theo user**
   - Train/Val/Test tách theo user_pid để tránh rò rỉ.

4) **Huấn luyện mô hình tuân thủ**
   - Baseline: Logistic Regression.
   - Tuỳ chọn: TinyTemporal (TCN/TinyTransformer) khi COLAB.
   - Calibration: isotonic hoặc Platt nếu đủ dữ liệu.

5) **Chính sách nhắc (Contextual Bandit)**
   - Ngữ cảnh: [đặc trưng thời gian + risk score].
   - Arms: {push,SMS,voice}. Thuật toán: ε-greedy.
   - Reward: responded_within_2h. Log lựa chọn và phần thưởng.

6) **Suy luận & Lập lịch “hôm nay”**
   - Với mỗi user, sinh các slot giờ ứng viên → dự đoán risk → chọn kênh bằng bandit → tạo lịch đề xuất.

7) **Đánh giá**
   - Mô hình: AUC, PR-AUC, F1, calibration (ECE).
   - Bandit: reward trung bình theo thời gian, CTR, phân bổ kênh.

8) **Công bằng theo trục hoạt động**
   - Nhóm theo channel, time bucket {05–11, 12–17, 18–22}, weekday/weekend.
   - Báo cáo chênh lệch TPR/FPR, AUCPR, ECE giữa nhóm.

9) **Stress tests**
   - Drop 20% events; Shift múi giờ +2h; Noise nhãn 5%.
   - So sánh chỉ số trước/sau.

10) **Artefacts**
   - model.pkl, metrics.json, fairness.json, figures/, schema_features.json, decision_log.md.

## A.6 Sinh dữ liệu synthetic (khi không có logs M2 thật)
- SMALL (LOCAL): N_USERS≈10–20, N_DAYS≈14–21 → 3k–8k sự kiện.
- LARGE (COLAB): N_USERS≈100–300, N_DAYS≈30–60 → ~100k sự kiện.
- Cách sinh:
  - Với mỗi user u, lấy base CTR theo kênh từ Beta(α,β).
  - Áp mô hình diurnal: multiplier g(hour) với 3 đỉnh nhẹ.
  - responded_within_2h ~ Bernoulli(CTR_channel · g(hour) · h(weekday/weekend)).
  - ack_latency_sec ~ LogNormal(μ,σ) nếu responded=1; else null.

## A.7 Giới hạn và quyết định
- Không có nhân khẩu học → không đánh giá fairness theo nhóm bảo vệ. Chủ đích để tuân thủ minimization.
- Công bằng báo cáo theo **hành vi sử dụng** và **thời gian** để phát hiện lệch kênh/khung giờ.
- Có thể mở rộng trường “quiet_hours_flag”, “net_type”, “delivery_retry_count” nếu cần phân tích lỗi mà không chạm PII/PHI.
