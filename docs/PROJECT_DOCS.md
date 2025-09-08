# AI Medication Reminder System

## 1) Phân tích bài toán (AI Thinking)

### 1.1. Bối cảnh, mục tiêu, người dùng
- Bối cảnh:
  - Tỷ lệ tuân thủ dùng thuốc (medication adherence) thấp ở người lớn tuổi, bệnh nhân mãn tính (đái tháo đường, tăng huyết áp) và người bận rộn, dẫn đến kết quả điều trị kém, gia tăng nhập viện.
  - Lịch dùng thuốc đa dạng, có tương tác thuốc, phụ thuộc bối cảnh (giờ giấc, ngày trong tuần), và kênh nhắc (push/SMS/voice).
- Mục tiêu:
  - Dự đoán khả năng phản hồi trong 2 giờ sau nhắc nhở.
  - Gợi ý kênh nhắc tối ưu theo ngữ cảnh để tăng xác suất phản hồi.
  - Phát hiện cảnh báo tương tác/ chống chỉ định cơ bản khi người dùng khai báo thuốc đang dùng.
- Người dùng chính:
  - Người cao tuổi sống độc lập, bệnh nhân mãn tính, người bận rộn; người chăm sóc; điều phối viên y tế từ xa.

### 1.2. Mục tiêu AI cụ thể
- Mô hình dự đoán xác suất phản hồi trong 2 giờ (binary classification).
- Hệ khuyến nghị kênh nhắc (contextual bandit epsilon-greedy).
- Kiểm tra tương tác thuốc / chống chỉ định mức cơ bản (rule-based + cơ sở dữ liệu mô phỏng).
- Theo dõi công bằng mô hình theo các nhóm bối cảnh (kênh, thời điểm, ngày).

### 1.3. Phương pháp AI phù hợp
- ML truyền thống cho baseline: Logistic Regression với pipeline tiền xử lý, giải thích được và nhanh.
- Dữ liệu chuỗi thời gian: có thể nâng cấp bằng mô hình thời gian (RNN/Transformer). Repo có TinyTemporal (PyTorch) như hướng mở rộng khi có GPU.
- Hệ gợi ý kênh nhắc: Contextual bandit (epsilon-greedy) để cân bằng thăm dò/khai thác, học từ phản hồi.
- NLP y tế: Có thể dùng sau này cho phân tích hướng dẫn thuốc, nhưng hiện phiên bản này dùng rule/checker đơn giản.
- Dự đoán tương tác thuốc: Hiện là rule-based mock DB; có thể thay bằng mô hình học máy (DDI prediction) hoặc tích hợp API dược điển.

Lý do chọn:
- Baseline ML: đủ mạnh trên dữ liệu hành vi, dễ huấn luyện, nhanh lặp.
- Bandit: phù hợp setting online, phản hồi nhị phân, cá nhân hóa theo ngữ cảnh.
- Rule-based DDI: đơn giản, minh bạch, dễ kiểm thử trong môi trường học thuật.

### 1.4. Dữ liệu, nguồn gốc và rủi ro
- Cần:
  - Lịch sử nhắc và phản hồi: ts_reminder, channel, responded_within_2h, ack_latency_sec, tz_offset,...
  - Bối cảnh: giờ, ngày, cuối tuần, khoảng thời gian từ lần trước, hành vi lịch sử (CTR rolling, lag).
  - Tùy chọn: chỉ số sinh học, wearable, hồ sơ bệnh án, thuốc đang dùng (người dùng nhập tay).
- Nguồn:
  - Phiên bản đồ án: Synthetic data (module generator) mô phỏng kịch bản thực.
  - Thực tế: EHR, hệ thống CRM/engagement, wearable APIs, pharmacy claims.
- Rủi ro:
  - Thiếu hụt/không đồng nhất: định dạng thời gian, múi giờ, dữ liệu lịch sử ít.
  - Chất lượng: nhiễu nhãn, hành vi bất thường.
  - Quyền riêng tư: dữ liệu sức khỏe nhạy cảm; cần ẩn danh, kiểm soát truy cập, tuân thủ chuẩn (HIPAA/GDPR).

### 1.5. Công cụ/mô hình dự kiến
- Framework: scikit-learn, pandas, numpy, matplotlib; PyTorch cho mô hình TinyTemporal (nếu có GPU).
- Kiến trúc hệ thống: Pipeline dữ liệu → tính đặc trưng → huấn luyện → đánh giá → bandit simulation → inference + cảnh báo DDI.
- Hạ tầng chạy: Local/Lab/Colab; config tự phát hiện môi trường.

### 1.6. Bối cảnh xã hội/đạo đức
- Quyền riêng tư dữ liệu y tế: bắt buộc mã hóa dữ liệu định danh, phân quyền, log truy cập, tránh dữ liệu vượt mục đích.
- Trách nhiệm pháp lý: công cụ mang tính hỗ trợ, không thay thế bác sĩ; cảnh báo và khuyến cáo rõ ràng giới hạn mô hình.
- Nguy cơ thiên lệch: nhóm giờ/nhóm kênh có thể được tối ưu khác nhau; đánh giá fairness across groups; audit định kỳ.
- Giải thích mô hình: ưu tiên baseline có thể diễn giải; đối với DL cần áp dụng XAI (SHAP/attention viz) khi triển khai thực tế.

---

## 2) Thiết kế mô hình & Pipeline

### 2.1. Input – Output hệ thống
- Input:
  - Dữ liệu tương tác lịch sử: logs.csv (user_pid, ts_reminder, tz_offset, channel, delivered, responded_within_2h, ack_latency_sec, snooze)
  - Tham số môi trường từ Config (SMALL/LARGE).
  - Dữ liệu người dùng cho inference: hour, dow, hành vi gần đây, channel hiện tại, time_bucket, …
  - Danh sách thuốc (tùy chọn) để kiểm tra tương tác.
- Output:
  - Mô hình đã huấn luyện (models/baseline_model.pkl, optional tiny_temporal).
  - Chỉ số đánh giá chi tiết và fairness (metrics/*evaluation.json).
  - Kết quả mô phỏng bandit (metrics/bandit_results.json).
  - Inference: recommended_channel, response_probability, confidence, warnings DDI/contraindications.

### 2.2. Pipeline xử lý dữ liệu (thu thập → tiền xử lý → trích chọn → huấn luyện → suy luận → nhắc nhở)
- Thu thập:
  - DataProcessor.load_or_generate_data: tạo synthetic nếu thiếu (scale SMALL/LARGE), validate schema/constraints/temporal.
- Tiền xử lý:
  - Chuẩn hóa thời gian UTC, sort theo user và thời gian, ép kiểu số/cờ.
- Trích chọn đặc trưng:
  - TemporalFeatures: hour/dow cyclical, time_bucket, is_weekend, hours_since_prev.
  - BehavioralFeatures: lag_1/lag_2, rolling CTR (ctr7, ctr14), ctr_user_channel, ack latency bucket, exp_decay_response, tz_offset_hours.
  - FeatureEngineer handles NaN fill, lưu feature schema.
- Huấn luyện:
  - Baseline Logistic Regression qua pipeline tiền xử lý (impute, scale, OneHot).
  - Optional TinyTemporal (PyTorch) cho chuỗi thời gian (với sequences).
- Đánh giá:
  - MetricsCalculator (accuracy, precision, recall, F1, AUC, calibration, thresholds).
  - FairnessAnalyzer trên channel/time_bucket/weekpart (weekday/weekend).
  - Stress testing: thiếu đặc trưng, nhiễu nhãn.
- Bandit simulation:
  - EpsilonGreedyBandit: học từ lịch sử mô phỏng, so sánh với random/fixed policy.
- Suy luận & nhắc nhở:
  - Inference với feature mock: dự đoán probability, chọn kênh bằng bandit, kiểm tra DDI/contraindications (warnings).

### 2.3. Lý do chọn kiến trúc
- Baseline Logistic Regression: nhanh, ổn định, có hệ tiền xử lý rõ ràng, phù hợp dữ liệu tabular hành vi/bối cảnh.
- Contextual bandit epsilon-greedy: phù hợp setting sequential decision với phản hồi nhị phân; dễ giải thích và mở rộng (LinUCB, Thompson Sampling).
- Temporal DL (RNN/Transformer) là hướng mở rộng có thể tăng hiệu quả nếu có chuỗi dài và tài nguyên.

---

## 3) Kết quả thử nghiệm

### 3.1. Cách huấn luyện thử với dữ liệu demo
- Dùng synthetic data (src/data/generator.py) với scale SMALL trên local; LARGE trên Colab.
- Chạy pipeline:
  - python main.py --run-pipeline
  - Optional: --force-retrain để huấn luyện lại.
- Inference ví dụ:
  - python main.py --inference --user-data '{"hour":9,"dow":1,"channel":"push"}' --medications aspirin warfarin

### 3.2. Tiêu chí đánh giá
- Phân loại: accuracy, precision, recall, F1, AUC.
- Calibration: Brier score, ECE, MCE.
- Thresholds: Youden J, Max F1 threshold.
- Fairness: chênh lệch accuracy/precision/recall/F1 giữa các nhóm (kênh, time_bucket, weekday/weekend).
- Bandit: average reward, cumulative reward, regret bounds (ước lượng).

### 3.3. Ví dụ minh họa đầu ra
- Lịch cá nhân hóa: time_bucket phù hợp, kênh khuyến nghị push buổi sáng, tránh voice buổi tối.
- Biểu đồ tuân thủ: đường CTR rolling theo thời gian (có thể xuất thêm trong tương lai).
- Nhắc nhở: “Hôm nay 6:30 sáng, đề xuất push; xác suất phản hồi 0.72; cảnh báo tương tác aspirin–warfarin: tăng nguy cơ chảy máu.”

---

## 4) Đánh giá và tư duy phản biện

### 4.1. Ưu điểm – hạn chế
- Ưu:
  - Pipeline end-to-end rõ ràng, reproducible.
  - Baseline dễ giải thích, fairness/stress test tích hợp.
  - Bandit mô phỏng đa chính sách, dễ mở rộng online learning.
- Hạn chế:
  - Synthetic data chưa phản ánh đầy đủ phức tạp thực tế.
  - DDI checker đơn giản; chưa tích hợp cơ sở dữ liệu dược uy tín.
  - Chưa có XAI chuyên sâu cho DL; chuỗi thời gian hạn chế.

### 4.2. Yếu tố xã hội, đạo đức, công bằng
- Công bằng: giám sát chất lượng theo nhóm (kênh/khung giờ/weekday vs weekend); cần mở rộng nhóm nhân khẩu học khi có dữ liệu.
- Bảo mật: ẩn danh, mã hóa, tối thiểu hóa dữ liệu, đánh giá tác động quyền riêng tư.
- Hộp đen của DL: dùng baseline diễn giải; nếu dùng DL phải có XAI, calibration và human-in-the-loop.

### 4.3. Hướng cải tiến/triển khai
- Kết nối IoT/wearable (nhịp tim, giấc ngủ) để điều chỉnh thời điểm/kênh.
- Tích hợp EHR, tiêu chuẩn HL7 FHIR.
- DDI: tích hợp cơ sở dữ liệu chuẩn (DrugBank, RxNorm, ONC, OpenFDA).
- Cá nhân hóa theo gen (PGx) cho thuốc có biến thiên chuyển hóa.
- Online learning bandit thực chiến (A/B/n), guardrails an toàn.

### 4.4. Câu hỏi phản biện
- Làm sao bảo vệ dữ liệu sức khỏe cá nhân khi cá nhân hóa? (ẩn danh, differential privacy, RBAC, audit logs, tách dữ liệu PII/PHI).
- Chuẩn hóa dữ liệu đa nguồn? (chuẩn FHIR, mapping mã, kiểm tra schema, kiểm soát chất lượng và lineage).
- Giải quyết “black-box” của DL? (XAI, surrogate models, constraints, kiểm thử hành vi, model cards, calibration).

### 4.5. Tham khảo tiêu biểu
- AI for Medication Adherence:
  - Demiris et al., JAMIA; Vrijens et al., “A new taxonomy for describing and defining adherence,” Br J Clin Pharmacol, 2012.
  - Kamarthi et al., “Improving medication adherence with machine learning,” AMIA/AAAI Workshops.
- Contextual Bandits:
  - Lattimore & Szepesvári, Bandit Algorithms, 2020.
  - Li et al., “A contextual-bandit approach to personalized news article recommendation,” WWW, 2010.
- Ethical AI in Healthcare:
  - Obermeyer et al., “Dissecting racial bias in an algorithm,” Science, 2019.
  - EU GDPR; HIPAA standards; Model Cards (Mitchell et al., FAT* 2019).
- Drug-Drug Interaction Prediction:
  - Zitnik et al., “Modeling polypharmacy side effects with graph convolutional networks,” Bioinformatics, 2018.
  - Google Scholar: “drug-drug interaction prediction deep learning”.

(Gợi ý: chèn link Google Scholar/DOI khi biên soạn chính thức.)

---

## 2) Tài liệu hóa dự án (main → src/*)

### main.py
- Vai trò: CLI entrypoint để chạy pipeline, inference, kiểm tra setup.
- Cách dùng:
  - Chạy pipeline: 
````python path=main.py mode=EXCERPT
    python main.py --run-pipeline --force-retrain
````
  - Inference một người dùng:
````python path=main.py mode=EXCERPT
    python main.py --inference --user-data '{"hour":9,"dow":1}' --medications aspirin warfarin
````
- Luồng chính:
  - Parse args → validate_setup (tuỳ chọn) → create_config → khởi tạo Pipeline → run_full_pipeline hoặc run_inference.

### src/__init__.py
- Xuất bản lớp Config và Pipeline:
````python path=src/__init__.py mode=EXCERPT
  from .config import Config
  from .pipeline import Pipeline
````

### src/config.py
- Cấu hình theo dataclass:
  - EnvironmentConfig: nhận biết Colab/Local, GPU, seed, data_scale SMALL/LARGE.
  - PathConfig: thư mục data/models/figures/metrics (auto create).
  - DataConfig: schema, danh sách kênh, time buckets, kích thước synthetic.
  - ModelConfig: tham số ML, TinyTemporal (epochs, hidden,...).
  - BanditConfig: epsilon, features ngữ cảnh, arms, ngưỡng tối thiểu mẫu/ success rate.
  - EvaluationConfig: threshold, calibration bins, fairness groups, stress test params.
- Khởi tạo nhanh:
````python path=src/config.py mode=EXCERPT
  cfg = Config.from_env()
````

### src/pipeline.py (Pipeline)
- Orchestrator 5 bước:
  1) Data Processing: load/generate → clean → split.
  2) Feature Engineering: temporal + behavioral → fill NaN → save feature_schema.json.
  3) Model Training: baseline + optional TinyTemporal → save models.
  4) Model Evaluation: metrics + fairness + stress tests → save metrics/*.json.
  5) Bandit Simulation: compare epsilon_greedy với random/fixed → save bandit_results.json.
- Inference:
  - Load model baseline; dựng feature vector tối giản từ user_data; epsilon-greedy chọn kênh; kiểm tra DDI/contraindications → trả recommendations + warnings.

### src/data/processor.py (DataProcessor)
- Chức năng:
  - load_or_generate_data(scale): đọc data/logs.csv hoặc tạo synthetic.
  - validate_full: kiểm tra schema/constraints/temporal.
  - parse_timestamps, sort_by_user_time, clean_data: chuẩn hóa dữ liệu, kiểu dữ liệu.
  - split_users: tách theo user để tránh leakage → train/val/test.
  - prepare_data: pipeline hoàn chỉnh, trả (full_df, train_df, val_df, test_df).
- I/O:
  - Input: logs.csv hoặc synthetic.
  - Output: các DataFrame đã làm sạch và tách.

### src/data/generator.py (SyntheticDataGenerator)
- Sinh dữ liệu synthetic có logic:
  - 2 nhắc mỗi ngày (sáng 6:30 ± jitters, tối 20:30 ± jitters).
  - Xác suất phản hồi phụ thuộc kênh/giờ/cuối tuần.
- save_synthetic_data(path, scale): tạo và lưu CSV.

### src/data/validator.py (DataValidator)
- validate_schema: cột bắt buộc.
- validate_constraints: hợp lệ kênh, quan hệ responded/ack_latency, tz_offset, format timestamp.
- validate_temporal_ordering: cảnh báo nếu timestamp không đơn điệu.
- validate_full: trả report chi tiết.

### src/features/temporal.py (TemporalFeatures)
- Tạo đặc trưng thời gian:
  - hour, hour_sin/cos; dow, dow_sin/cos; time_bucket; is_weekend; hours_since_prev.
- Dùng cho hành vi theo nhịp sinh học/ngày giờ (circadian).

### src/features/behavioral.py (BehavioralFeatures)
- Lag features: lag_1, lag_2 (phản hồi trước).
- Rolling CTR: ctr7, ctr14 (loại bỏ rò rỉ bằng shift).
- CTR theo user-channel: ctr_user_channel (expanding mean trễ 1).
- Ack latency features: ack_latency_sec median fill, latency_bucket.
- Exponential decay history: exp_decay_response (lambda decay).
- create_all_behavioral_features: ghép tất cả.

### src/features/engineer.py (FeatureEngineer)
- Điều phối tạo đặc trưng temporal + behavioral, xử lý tz_offset → tz_offset_hours.
- Fill NaN: median cho số; mode/’unknown’ cho categorical.
- get_feature_columns: phân nhóm numeric/categorical và all.
- prepare_model_features: chọn X (numeric+categorical), y=responded_within_2h.
- create_sequences_for_temporal_model: tạo sequences khi dùng TinyTemporal.
- save_feature_schema(path): lưu schema JSON.

### src/models/baseline.py (BaselineModel)
- Pipeline scikit-learn:
  - ColumnTransformer: Imputer + StandardScaler cho số; OneHotEncoder cho categorical (drop="first", handle_unknown="ignore").
  - LogisticRegression(class_weight="balanced", max_iter cấu hình).
- API:
  - fit, predict, predict_proba.
  - save/load bằng pickle, lưu cả feature_names và config.
  - get_feature_importance: dựa trên coefficients (đơn giản, không giải phẫu one-hot).

### src/models/trainer.py (ModelTrainer)
- train_baseline: huấn luyện, in Validation Accuracy/AUC, lưu models/baseline_model.pkl.
- train_tiny_temporal: nếu được bật và có torch; fit/predict, lưu .pth.
- train_all_models: chuẩn bị features/sequences, huấn luyện tuần tự, trả dict models.
- load_models: tải lại baseline/tiny_temporal nếu có.

### src/models/tiny_temporal.py
- Lưu ý: Repo có file; dùng PyTorch để mô hình chuỗi nhỏ gọn (chi tiết trong file tương ứng). Bật qua config.model.use_tiny_temporal (auto bật nếu Colab + GPU).
  - Gợi ý khi báo cáo: mô tả input sequences (window_size), embedding/hidden, optimizer, epochs.

### src/evaluation/metrics.py (MetricsCalculator)
- Tính:
  - Basic: accuracy, precision, recall, F1, AUC.
  - Confusion-derived: specificity, sensitivity, PPV, NPV.
  - Calibration: Brier, ECE, MCE (+ binning).
  - Thresholds: Youden J, Max F1 threshold.
- calculate_all_metrics: hợp nhất.
- print_metrics_summary: in tóm tắt đẹp.

### src/evaluation/fairness.py (FairnessAnalyzer)
- Theo dõi theo nhóm: channel, time_bucket, weekpart (configurable).
- Tính chênh lệch tối đa accuracy/precision/recall/F1, positive_rate, độ lệch chuẩn, fairness_score.
- In báo cáo nhóm và đánh giá (GOOD/MODERATE/POOR).

### src/evaluation/evaluator.py (ModelEvaluator)
- evaluate_model: tính metrics + fairness (nếu có test_df); lưu predictions.
- stress_test_model: thiếu đặc trưng (drop fraction + impute), nhiễu nhãn (label noise), so sánh AUC drop.
- evaluate_all_models: lưu kết quả từng model, compare đa model, lưu all_models_evaluation.json.

### src/bandit/epsilon_greedy.py (EpsilonGreedyBandit)
- Epsilon decay sqrt/linear; outcome models LogisticRegression theo từng kênh, học từ history khi đủ mẫu và class balance hợp lý.
- select_arm(context): thăm dò ngẫu nhiên với epsilon, khai thác dựa trên predict_proba hoặc trung bình kinh nghiệm.
- update: cập nhật history, huấn luyện định kỳ; trang thống kê arm stats & regret bounds.

### src/bandit/simulator.py (BanditSimulator)
- simulate_bandit_policy: chạy qua test_df, chọn arm, tính reward (counterfactual đơn giản khi arm ≠ channel), cập nhật bandit, trả cumulative stats.
- compare_policies: epsilon=0.1/0.2/0.3 vs random vs fixed_best; xếp hạng theo average_reward; có tiện ích plot (nếu có matplotlib).

### src/utils/helpers.py
- setup_logging(log_level, log_file): cấu hình logger.
- save_results/load_results: JSON/pickle/CSV tiện ích.
- validate_config(dict).
- format_duration, print_section_header.

### src/utils/drug_interactions.py (DrugInteractionChecker)
- CSDL tương tác thuốc mock (warfarin–aspirin,...), chống chỉ định theo điều kiện/tuổi/thai kỳ (giả lập).
- check_drug_interactions(list meds): trả danh sách cảnh báo drug–drug.
- check_contraindications(med, patient_profile): cảnh báo theo profile (age, conditions, pregnancy).

---

## Hướng dẫn chạy nhanh và kiểm thử
- Cài dependencies: pip install -r requirements.txt
- Kiểm tra setup:
````python path=main.py mode=EXCERPT
  python main.py --validate-setup
````
- Chạy pipeline:
````python path=main.py mode=EXCERPT
  python main.py --run-pipeline
````
- Inference:
````python path=main.py mode=EXCERPT
  python main.py --inference --user-data '{"hour":9,"dow":1,"channel":"push"}' --medications aspirin warfarin
````
- Gợi ý kiểm thử tự động:
  - Thêm unit tests cho:
    - DataValidator.validate_full với dữ liệu sai schema.
    - FeatureEngineer: kiểm tra có cột bắt buộc sau khi transform.
    - BaselineModel.save/load round-trip.
    - Bandit: verify decay epsilon, update counts, selection rate bounds.
    - Evaluator: metrics nhất quán khi thay threshold.

Nếu bạn muốn, tôi có thể bổ sung skeleton unit tests (pytest) cho các phần trên.

---

## Gợi ý mở rộng tài liệu
- Thêm sơ đồ pipeline.svg (repo có sẵn file pipeline.svg) vào báo cáo để minh họa luồng.
- Viết Model Card ngắn cho baseline (dữ liệu huấn luyện, metric, ràng buộc, bias, intended use).
- Bổ sung phụ lục: cấu hình Colab (docs/COLAB_SETUP.md), mô tả Pipeline (docs/PIPELINE.md), thiết kế (docs/DESIGN_MODEL_AND_PIPELINE.md), đánh giá (docs/EVALUATION.md). Repo đã có các tài liệu này; có thể trích chọn nội dung thích hợp vào báo cáo cuối.

---

## Kết luận
Kế hoạch theo chuẩn AI Thinking và tài liệu hóa mã nguồn đã sẵn sàng để bạn nộp đồ án hoặc làm nền tảng triển khai. Bạn muốn tôi:
- Viết bản PDF theo định dạng báo cáo (thêm hình, trích dẫn cụ thể)?
- Tạo bộ unit tests tối thiểu và chạy thử trên máy của bạn?
- Tích hợp thêm mô hình bandit nâng cao (LinUCB/Thompson) để so sánh?