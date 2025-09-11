## 1) Lý do không dùng dữ liệu thật

* Rủi ro pháp lý và đạo đức: PII/PHI, consent, IRB.
* Tuân thủ nguyên tắc data minimization của GDPR và yêu cầu “minimum necessary” của HIPAA.

## 2) M2 là gì

* **M2 = Minimal, Message-level, Medication-reminder logs.**
* Nhật ký tối thiểu chỉ chứa sự kiện nhắc và phản hồi. Không chứa PII/PHI. Không chứa đơn thuốc hay chẩn đoán.
* Mục tiêu của đồ án: dự báo **responded\_within\_2h** và tối ưu **kênh** cùng **thời điểm** nhắc. Với hai mục tiêu này chỉ cần hành vi theo thời gian.

## 3) Vì sao không dùng Synthea mà chọn schema M2

* **Phù hợp mục tiêu**: Bài toán cần log nhắc–phản hồi để học tuân thủ và tối ưu kênh/giờ. EHR đầy đủ của Synthea dư thừa, tăng footprint và phức tạp hóa pipeline FHIR/OMOP.
* **Riêng tư và tuân thủ**: M2 hiện thực hóa data minimization. Tránh PII/PHI nên dễ công bố và lược bỏ dữ liệu sau khi chấm.
* **Tái lập offline**: Chạy nhanh trên CPU cục bộ, không phụ thuộc tải bộ dữ liệu lớn hay generator.
* **Đánh giá công bằng theo “trục hoạt động”**: Cần khác biệt theo kênh và khung giờ, không cần nhân khẩu học.

## 4) Cơ sở học thuật cho schema M2 và ngưỡng 2 giờ

### Vì sao các cột có mặt

* `ts_reminder`, `channel`, `delivered`: mô tả can thiệp và tính khả dụng của can thiệp theo chuẩn báo cáo mHealth như mERA và CONSORT-EHEALTH.
* `responded_within_2h`: **proximal outcome** theo thiết kế JITAI hoặc Micro-Randomized Trial. Outcome được định nghĩa trong cửa sổ ngắn ngay sau can thiệp.
* `ack_latency_sec`: đo độ trễ phản hồi để phân tích hiệu chuẩn và tối ưu thời điểm.
* `tz_offset`: chuẩn hóa múi giờ để suy ra giờ địa phương, cần cho phân tích theo morning–afternoon–evening.

### Tại sao ngưỡng 2 giờ hợp lý

* Thực hành lâm sàng: ±2 giờ thường được chấp nhận với thuốc không time-critical theo hướng dẫn ISMP và CMS.
* Nghiên cứu triển khai: nhiều nghiên cứu định nghĩa “timely dose” trong 2 giờ hoặc trigger đánh giá sau mốc 2 giờ.
* Đo tuân thủ bằng MEMS: “timing compliance” hay dùng khung 2–4 giờ, củng cố tính hợp lệ cho mốc 2 giờ.

## 5) Schema 8 cột và vai trò

| Cột                           | Lý do tồn tại                  | Dùng cho                                                  |
| ----------------------------- | ------------------------------ | --------------------------------------------------------- |
| `user_pid`                    | Khóa ẩn danh để gom theo người | Tạo đặc trưng theo user. Chia train/val/test theo user    |
| `ts_reminder` (UTC)           | Mốc thời gian nhắc             | Suy giờ trong ngày. Khoảng cách giữa nhắc. Thứ trong tuần |
| `tz_offset`                   | Chuyển UTC về giờ địa phương   | Bảo toàn nhịp sinh học theo khu vực                       |
| `channel` ∈ {push,SMS,voice}  | Đòn bẩy can thiệp cần tối ưu   | Học hiệu quả theo kênh. Context cho bandit                |
| `delivered` ∈ {0,1}           | Kiểm soát gửi thành công       | Loại nhiễu “không giao được nhưng bị tính fail”           |
| `responded_within_2h` ∈ {0,1} | Nhãn mục tiêu                  | Huấn luyện. Reward cho bandit                             |
| `ack_latency_sec`             | Độ trễ phản hồi                | Phân tích latency. Bucket hóa để giảm nhiễu               |
| `snooze` ∈ {0,1}              | Hành vi hoãn                   | Tín hiệu sớm nguy cơ bỏ lỡ                                |

Không đưa tuổi, giới, chẩn đoán, đơn thuốc, thiết bị, vị trí. Lý do: không cần cho hai mục tiêu cốt lõi. Tăng rủi ro riêng tư và chi phí tái lập. Nếu cần vận hành có thể bổ sung thuộc tính không định danh như `os_family`, `app_version`, `delivery_retry_count`.

## 6) Từ 8 cột suy ra đặc trưng gì và tính thế nào

Nguyên tắc: tối thiểu nhưng đủ dùng. Không rò rỉ tương lai. Tái lập, O(n). Bám nhịp sinh học, thói quen, recency. Dữ liệu mượt để hiệu chuẩn.

| Nhóm             | Đặc trưng                                                       | Nắm bắt                | Lý do                                       | Cách tính                                                              |
| ---------------- | --------------------------------------------------------------- | ---------------------- | ------------------------------------------- | ---------------------------------------------------------------------- |
| Thời gian        | `hour_sin`, `hour_cos`                                          | Chu kỳ 24h             | Tránh đứt gãy 23→0                          | `hour = h + m/60`; `sin(2π·hour/24)`, `cos(2π·hour/24)`                |
| Thời gian        | `dow_sin`, `dow_cos`                                            | Chu kỳ 7 ngày          | Hành vi khác cuối tuần                      | `dow ∈ {0..6}`; `sin(2π·dow/7)`, `cos(2π·dow/7)`                       |
| Thời gian        | `time_bucket` ∈ {morning 05–11, afternoon 12–17, evening 18–22} | Khung giờ thô          | Dễ so sánh theo nhóm trong bandit           | Từ `hour`                                                              |
| Hành vi ngắn hạn | `lag_1`, `lag_2` của `responded_within_2h` theo user            | Quán tính hành vi      | Hiệu ứng “vừa phản hồi”                     | Dịch 1–2 bước theo chuỗi từng user                                     |
| Nhịp nhắc        | `hours_since_prev`                                              | Khoảng cách giữa nhắc  | Quá dày gây mệt mỏi. Quá thưa mất thói quen | `(t_i − t_{i−1})/3600` theo user                                       |
| Mức độ gắn kết   | `ctr_7d_user`, `ctr_14d_user`                                   | Tỷ lệ phản hồi gần đây | Proxy động cho hứng thú                     | Trung bình trượt theo user                                             |
| Ưa thích kênh    | `ctr_user_channel`                                              | Hiệu ứng kênh cá nhân  | Chọn kênh trong bandit                      | Trung bình trượt theo `(user, channel)`                                |
| Độ trễ           | `ack_latency_sec` hoặc `latency_bucket`                         | Nhanh hay chậm         | Tương quan với khả năng trong 2h            | Impute median (per-user rồi global), clip \[30, 1800] giây, bucket hóa |

Loại trừ: đặc trưng nội dung thông điệp, vị trí, thiết bị vì ràng buộc đạo đức và không cần cho mục tiêu.

## 7) Chiến lược tạo dữ liệu synthetic

* Mục tiêu: có dao động theo thời gian và khác biệt giữa kênh để kiểm thử mô hình và bandit.
* Thiết lập: đặt **base CTR theo kênh** bằng phân phối Beta. Cộng **hàm diurnal** theo giờ và hiệu ứng cuối tuần. Lấy mẫu Bernoulli cho `responded_within_2h`.
* `ack_latency_sec`: lấy từ phân phối log-normal để phản ánh hành vi mở app chậm nhanh.
* Ưu điểm: đúng cấu trúc pipeline sẽ học. Có nhịp ngày, hiệu ứng kênh, thói quen ngắn hạn.

## 8) Mô hình: Logistic Regression và TinyTemporal

### Logistic Regression

* Phù hợp phân loại nhị phân trên tabular. Dễ hiệu chuẩn xác suất (Platt hoặc Isotonic). Chạy CPU nhanh.
* Công thức: $p(y=1|x)=\sigma(w^\top x+b)$. Trọng số $w$ cho diễn giải.
* Hạn chế: không tự học phụ thuộc chuỗi dài khi đặc trưng thủ công thiếu.

### TinyTemporal

* Mục tiêu: bắt phụ thuộc chuỗi ngoài phần feature thủ công. Kích thước rất nhỏ để chạy nhanh.
* Hai biến thể:

  1. **TCN nhỏ**: conv 1D nhân quả, có giãn. Song song tốt, ít tham số, phù hợp cửa sổ T×F gần đây.
  2. **Transformer nhỏ**: 1–2 layer, 2–4 heads, ẩn nhỏ. Linh hoạt với chuỗi không đều, chậm hơn TCN khi tài nguyên ít.
* Bật khi có đủ chuỗi dài và kỳ vọng vượt baseline. Tắt khi cấu hình LOCAL/SMALL hoặc uplift thấp.

## 9) Pipeline chuẩn và lý do

1. Ingest và chuẩn hóa thời gian bằng `tz_offset` để đặc trưng thời gian có ý nghĩa.
2. Feature engineering giá rẻ, giải thích được, làm context cho bandit.
3. Chia train/val/test theo **user** để tránh rò rỉ.
4. Huấn luyện LR và **hiệu chuẩn xác suất** làm baseline mạnh.
5. Tùy chọn TinyTemporal khi dữ liệu đủ lớn.
6. Chính sách nhắc bằng **contextual bandit** (ví dụ ε-greedy) dùng score rủi ro và đặc trưng thời gian để chọn kênh. Đánh giá bằng replay/simulation.
7. Suy luận vận hành: quét lưới giờ trong ngày, tính risk, chọn kênh, xuất lịch.
8. Giám sát: AUC, PR-AUC, F1, ECE. Fairness theo kênh và khung giờ. Stress tests.

## 10) Kết luận

* **M2** hiện thực hóa data minimization nhưng vẫn đủ để học hành vi theo thời gian và hiệu ứng kênh.
* Từ 8 cột suy ra trọn bộ đặc trưng cần cho dự báo và bandit.
* **LR** cung cấp baseline minh bạch. **TinyTemporal** khai thác phụ thuộc chuỗi khi có lợi.
* Ngưỡng **2 giờ** bám thực hành lâm sàng và tiền lệ nghiên cứu.
* Toàn bộ thiết kế chạy nhanh, tái lập offline, an toàn riêng tư.

---
## Tài liệu tham khảo

[1]: https://www.fda.gov/drugs/development-resources/drug-interactions-labeling "Drug Interactions & Labeling"
[2]: https://snap.stanford.edu/decagon/ "Decagon is a graph convolutional neural network for ..."
[3]: https://academic.oup.com/bioinformatics/article/34/13/i457/5045770 "Modeling polypharmacy side effects with graph convolutional ..."
[4]: https://pubmed.ncbi.nlm.nih.gov/29949996/ "Modeling polypharmacy side effects with graph ..."
[5]: https://www.pnas.org/doi/10.1073/pnas.1803294115 "Deep learning improves prediction of drug– ... - PNAS"
[6]: https://synthetichealth.github.io/synthea/ "Synthea - GitHub Pages"
[7]: https://github.com/synthetichealth/synthea "synthetichealth/synthea: Synthetic Patient Population ..."
[8]: https://mitre.github.io/fhir-for-research/modules/synthea-overview "Synthea Synthetic Data Overview – FHIR® for Research ..."
[9]: https://gdpr-info.eu/art-5-gdpr/ "Art. 5 GDPR – Principles relating to processing of personal ..."
[10]: https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/data-protection-principles/a-guide-to-the-data-protection-principles/data-minimisation/ "Principle (c): Data minimisation | ICO"
[11]: https://www.hhs.gov/hipaa/for-professionals/privacy/guidance/minimum-necessary-requirement/index.html "Minimum Necessary Requirement"
[12]: https://pmc.ncbi.nlm.nih.gov/articles/PMC3278112/ "CONSORT-EHEALTH: Improving and Standardizing ..."
[13]: https://www.who.int/news/item/21-03-2016-new-checklist-published-to-help-improve-reporting-of-mhealth-interventions "New checklist published to help improve reporting of ..."
[14]: https://www.equator-network.org/reporting-guidelines/consort-ehealth-improving-and-standardizing-evaluation-reports-of-web-based-and-mobile-health-interventions/ "CONSORT-EHEALTH - Reporting guideline"
[15]: https://pmc.ncbi.nlm.nih.gov/articles/PMC4732571/ "Micro-Randomized Trials: An Experimental Design for ..."
[16]: https://pmc.ncbi.nlm.nih.gov/articles/PMC7193439/ "Usage Metrics of Web-Based Interventions Evaluated in ..."
[17]: https://www.ismp.org/sites/default/files/attachments/2018-02/tasm.pdf "ISMP Acute Care Guidelines for Timely Administration of ..."
[18]: https://www.cms.gov/medicare/provider-enrollment-and-certification/surveycertificationgeninfo/downloads/scletter12_05.pdf "DEPARTMENT OF HEALTH & HUMAN SERVICES"
[19]: https://www.providence.org/-/media/Project/psjh/providence/kadlec/files/student-faculty-resources/student-nursing/medication-administration-timing-precision-6501106.pdf?hash=422BC748B48F29579F2A9E1092F8D1EC&la=en&utm_source=chatgpt.com "Medication Administration Timing Precision, 650.11.06"
[20]: https://pmc.ncbi.nlm.nih.gov/articles/PMC9164090/ "An In-Home Medication Dispensing System to Support ..."
[21]: https://aidsrestherapy.biomedcentral.com/articles/10.1186/s12981-024-00653-0 "Effect of mobile health intervention on medication time ..."
[22]: https://pmc.ncbi.nlm.nih.gov/articles/PMC7379515/ "Outcome measures for adherence data from a medication ..."
