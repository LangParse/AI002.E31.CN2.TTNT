## 4. Đánh giá và tư duy phản biện

### 4.1 Phân tích lỗi

**Phương pháp**

* *Error slicing* (chẻ lỗi): phân tích theo **kênh** và **khung giờ** để tìm mẫu sai phổ biến.
* *Calibration check*: so đường **reliability** và **ECE** (*Expected Calibration Error*: sai lệch giữa xác suất dự báo và tần suất thực).
* *Confusion@τ*\*\*: dùng ngưỡng τ\* (điểm F1 tối đa trên tập val) để đọc **TP/FP/FN/TN**.

**Top-3 mẫu sai điển hình** *(minh họa; thay bằng số thực khi chạy)*

|   ID | Ngữ cảnh                                          | Dự báo `p` | Nhãn | Sai kiểu | Giả thuyết nguyên nhân             | Hướng khắc phục                                       |
| ---: | ------------------------------------------------- | ---------: | ---: | -------- | ---------------------------------- | ----------------------------------------------------- |
|   E1 | *evening* + **voice**, `ctr_user_channel` thấp    |       0.62 |    0 | FP       | kênh voice nhiễu, LR overconfident | calibrate + hạ trọng số voice trong policy            |
|   E2 | *morning* + **push**, lịch sử kém (`hist_k` thấp) |       0.31 |    1 | FN       | dependency theo chuỗi chưa nắm     | bật **TinyTemporal** hoặc thêm đặc trưng *exp\_decay* |
|   E3 | *afternoon* + **SMS**, gap dài bất thường         |       0.55 |    0 | FP       | pattern hiếm (outlier)             | robust scaler + regularization mạnh hơn               |

**Ảnh hưởng của calibration**

* Trước: ECE=0.08 ⇒ nhiều **FP** ở evening.
* Sau **Platt/Isotonic**: ECE≈0.03 ⇒ điểm cắt τ\* ổn định hơn, giảm cảnh báo thừa ở kênh yếu.

---

### 4.2 Công bằng, đạo đức, riêng tư

#### 4.2.1 Công bằng theo **trục hoạt động**

* Trục: **channel** ∈ {push,SMS,voice}, **time-bucket** ∈ {morning, afternoon, evening}, **weekday/weekend**.
* Chỉ số: **TPR**, **FPR**, **AUCPR**, **ECE** theo nhóm.
* **Khoảng tin cậy (CI)**: dùng **bootstrap theo user** (lấy mẫu lại có hoàn lại ở cấp người dùng) 1 000 lần; báo cáo CI 95%.
* **Ngưỡng hành động**: nếu *ΔTPR* hoặc *ΔAUCPR* > 0.1 giữa các nhóm →

  1. tăng **ε** của bandit ở nhóm kém, 2) áp giới hạn tần suất nhắc, 3) xem xét điều chỉnh loss *class\_weight* theo nhóm hoạt động.

**Rủi ro thao tác kênh**

* *Over-exploitation* push gây mệt mỏi người dùng → áp **cool-down** per-channel, **diversity constraint** (ví dụ không quá 60% push/tuần).

#### 4.2.2 Riêng tư & bảo mật (DATA\_MODE = M2)

* **Minimization**: chỉ lưu *nhắc–phản hồi*; **không** PII/PHI, **không** nội dung tin nhắn, **không** vị trí/thiết bị.
* **Ẩn danh**: `user_pid` = UUID giả; bảng ánh xạ giữ cục bộ, không commit.
* **Consent**: file *consent.md* nêu mục đích, thời hạn lưu, quyền rút lại, quy trình xóa.
* **Lưu trữ**: thư mục `data/` nội bộ; xóa sau chấm điểm; artefact chỉ chứa số liệu tổng hợp.
* **Kiểm soát truy cập**: chỉ nhóm chấm có quyền đọc; log truy cập ở mức tổng hợp.
* **Tuyên bố**: hệ thống **không thay thế tư vấn y tế**; cảnh báo chỉ mang tính hỗ trợ.

#### 4.2.3 Quản trị mô hình

* **Data Card (M2)**: nguồn synthetic + nhật ký tối giản; known gaps: không nhân khẩu học.
* **Model Card (ngắn)**: mục tiêu ước lượng *risk 24–72h*; base LR + calibration; dải dữ liệu hợp lệ; hạn chế & cảnh báo sử dụng.
* **Giám sát triển khai**: theo dõi **AUC/PR-AUC/ECE** và fairness theo tháng; cảnh báo khi drift.

---

### 4.3 Hạn chế và rủi ro

**Kỹ thuật**

* Không có nhân khẩu học ⇒ không đo *group fairness* cổ điển (giới, tuổi).
* Synthetic bias: quy luật tạo dữ liệu có thể khác thực tế.
* Bandit **offline replay** dựa *outcome-model* ⇒ dễ lạc quan giả (*model bias*).
* Concept drift: thói quen uống thay đổi theo mùa/kỳ nghỉ.

**Vận hành**

* *Over-alert*: nhắc quá dày gây “mệt mỏi thông báo”.
* Lệch kênh do chi phí/khả dụng (ví dụ voice giới hạn hạ tầng).
* Sai số múi giờ dẫn tới nhắc không phù hợp.

**Sổ rủi ro (risk register)**

| Rủi ro             |   Khả năng |   Tác động | Giảm thiểu                                                      |
| ------------------ | ---------: | ---------: | --------------------------------------------------------------- |
| Over-alert         | Trung bình |        Cao | giới hạn tần suất, τ\* động, ưu tiên slot có risk cao nhất      |
| Model bias offline |        Cao | Trung bình | đánh giá **off-policy** (IPS/SNIPS/DR\*) và A/B nhỏ             |
| Drift hành vi      | Trung bình | Trung bình | monitor PSI/KS theo đặc trưng thời gian, tái huấn luyện định kỳ |
| Sai múi giờ        |       Thấp | Trung bình | chuẩn hóa UTC+offset, kiểm thử **shift +2h** trong stress       |
| Lộ dữ liệu         |       Thấp |        Cao | minimization, ẩn danh, quy trình xóa, kiểm soát truy cập        |

\* **IPS/SNIPS/DR**: *Inverse Propensity Scoring*, *Self-Normalized IPS*, *Doubly-Robust* — kỹ thuật **đánh giá ngoài tuyến** (off-policy evaluation) để ước lượng reward chính sách mới từ dữ liệu cũ.

---

### 4.4 Hướng cải tiến

**Dữ liệu & riêng tư**

* Kết nối chuẩn **FHIR/OMOP** khi có ủy quyền; thêm wearable (nhịp tim, giấc ngủ).
* **Federated learning** + **Differential Privacy** khi mở rộng nhiều cơ sở.

**Mô hình**

* Policy: **Thompson Sampling**, **LinUCB** (bandit tuyến tính theo ngữ cảnh) để hội tụ nhanh hơn.
* Đánh giá: áp **IPS/SNIPS/DR** + **bootstrap theo user** làm CI cho uplift.
* Giải thích: **SHAP** mức phiên để hỗ trợ audit; ràng buộc đơn điệu trên đặc trưng thời gian nếu cần.

**Vận hành**

* A/B test online nhỏ, canh chừng mệt mỏi thông báo bằng thước đo *opt-out rate*.
* Tái hiệu chuẩn (calibration) định kỳ; cảnh báo khi **ECE** vượt ngưỡng.
* Guardrail: không gửi >N nhắc/24h, không quá M nhắc liên tiếp cùng kênh.

---

### 4.5 Decision log (7 quyết định chính)

1. **Chỉ thu M2** để giảm rủi ro riêng tư ↔ chấp nhận mất *group fairness* cổ điển.
2. **Split theo user** để tránh rò rỉ ↔ giảm số mẫu train; bù bằng synthetic LARGE trên COLAB.
3. **LR + calibration** làm baseline vì nhanh, ổn định ↔ có thể kém chuỗi dài; bổ sung **TinyTemporal** tùy chọn.
4. **Fairness theo hoạt động** thay vì nhân khẩu học ↔ phạm vi hẹp hơn nhưng phù hợp M2.
5. **Bandit ε-greedy** đơn giản, dễ kiểm soát ↔ hiệu quả hội tụ vừa; roadmap sang Thompson/LinUCB.
6. **Outcome-model per-arm** cho replay offline ↔ có bias; bù bằng **off-policy evaluation** khi có log propensities.
7. **Stress tests** (drop/shift/noise) bắt buộc trong CI để phát hiện giòn mong manh sớm; tạo quy tắc hiệu chuẩn lại khi ECE ↑.
