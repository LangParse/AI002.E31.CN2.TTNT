# AI Medication Reminder System

A comprehensive AI-powered medication reminder system that uses machine learning and contextual bandits to optimize reminder delivery channels and timing. The system includes fairness analysis, drug interaction checking, and supports both local development and cloud training environments.

## 🚀 Features

- **Intelligent Channel Selection**: Uses contextual bandits (epsilon-greedy) to optimize reminder delivery channels (push, SMS, voice)
- **Machine Learning Models**: Supports both baseline (Logistic Regression) and advanced (TinyTemporal LSTM) models
- **Fairness Analysis**: Comprehensive bias detection across demographic groups and channels
- **Drug Interaction Checking**: Built-in DDI (Drug-Drug Interaction) validation system
- **Dual Environment Support**: Optimized for both local development and Google Colab training
- **Comprehensive Evaluation**: Includes stress testing, calibration analysis, and performance metrics
- **Modular Architecture**: Clean separation of concerns with well-defined interfaces

## 📋 Pipeline Overview

The system follows a 7-step pipeline (A.1 - A.7):

1. **A.1**: Drug Interaction Checking & Validation
2. **A.2**: Data Schema Validation
3. **A.3**: Model Evaluation & Fairness Analysis
4. **A.4**: Feature Engineering (Temporal & Behavioral)
5. **A.5**: Model Training (Baseline + TinyTemporal)
6. **A.6**: Synthetic Data Generation
7. **A.7**: Contextual Bandit Simulation

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- pip or conda

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/your-username/ai-medication-reminder.git
cd ai-medication-reminder

# Install dependencies
pip install -e .

# For development with all optional dependencies
pip install -e ".[all]"
```

### For Google Colab

**Option 1: Use the Colab Notebook (Recommended)**
1. Upload `ai_medication_reminder_colab.ipynb` to Google Colab
2. Run all cells - automatic setup included!
3. See `COLAB_SETUP.md` for detailed instructions

**Option 2: Manual Setup**
```python
# In Colab notebook
!git clone https://github.com/your-username/ai-medication-reminder.git
%cd ai-medication-reminder
!pip install -e ".[torch]"
```

## 🚀 Quick Start

### Validate Setup

```bash
python main.py --validate-setup
```

### Run Full Pipeline

```bash
# Local environment (small dataset)
python main.py --run-pipeline

# Force retrain models
python main.py --run-pipeline --force-retrain

# Use large dataset
python main.py --run-pipeline --data-scale LARGE
```

### Single User Inference

```bash
python main.py --inference --user-data '{"hour": 9, "dow": 1, "ctr7": 0.6}' --medications "aspirin" "warfarin"
```

## 📊 Usage Examples

### Python API

```python
from src import Config, Pipeline

# Initialize with default config
config = Config.from_env()
pipeline = Pipeline(config)

# Run full pipeline
results = pipeline.run_full_pipeline()

# Run inference for a user
user_context = {
    "hour": 9,           # 9 AM
    "dow": 1,            # Monday
    "ctr7": 0.6,         # 7-day CTR
    "hours_since_prev": 24
}

recommendations = pipeline.run_inference(
    user_context,
    medications=["aspirin", "lisinopril"]
)

print(f"Recommended channel: {recommendations['recommendations']['recommended_channel']}")
print(f"Response probability: {recommendations['recommendations']['response_probability']:.3f}")
```

## 🏗️ Architecture

### Module Structure

```
src/
├── config.py              # Configuration management
├── pipeline.py             # Main pipeline orchestrator
├── data/                   # Data processing
│   ├── processor.py        # Data loading & preprocessing
│   ├── generator.py        # Synthetic data generation
│   └── validator.py        # Schema validation
├── features/               # Feature engineering
│   ├── engineer.py         # Main feature orchestrator
│   ├── temporal.py         # Time-based features
│   └── behavioral.py       # User behavior features
├── models/                 # Machine learning models
│   ├── trainer.py          # Model training orchestrator
│   ├── baseline.py         # Logistic regression baseline
│   └── tiny_temporal.py    # LSTM temporal model
├── evaluation/             # Model evaluation
│   ├── evaluator.py        # Main evaluation orchestrator
│   ├── metrics.py          # Performance metrics
│   └── fairness.py         # Fairness analysis
├── bandit/                 # Contextual bandit system
│   ├── simulator.py        # Bandit simulation
│   ├── epsilon_greedy.py   # Epsilon-greedy algorithm
│   └── contextual_bandit.py # Base bandit interface
└── utils/                  # Utilities
    ├── helpers.py          # Common utilities
    └── drug_interactions.py # DDI checking
```

## 📈 Data Scales

- **SMALL**: ~10-20 users, 14-21 days, ~3k-8k records (Local development)
- **LARGE**: ~100-300 users, 30-60 days, ~100k records (Colab training)

## 🔧 Configuration

The system uses environment-aware configuration:

```python
from src import Config

config = Config.from_env()
print(f"Environment: {'Colab' if config.env.in_colab else 'Local'}")
print(f"Data Scale: {config.env.data_scale}")
print(f"GPU Available: {config.env.has_gpu}")
```

## 📝 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📚 Documentation

### Files Overview

- **`ai_medication_reminder_colab.ipynb`**: Complete interactive notebook for Google Colab
- **`COLAB_SETUP.md`**: Detailed setup guide for Google Colab
- **`main.py`**: Command-line interface for local execution
- **`src/`**: Modular source code with clean architecture
- **`demo.ipynb`**: Original demo notebook (legacy)

### Interactive Notebook Features

The Colab notebook (`ai_medication_reminder_colab.ipynb`) includes:

- 🔧 **Automatic Setup**: Environment detection and dependency installation
- 🔄 **Full Pipeline**: One-click execution of complete A.1-A.7 pipeline
- 🔍 **Component Exploration**: Step-by-step analysis of each module
- 📊 **Visualizations**: Interactive charts and performance analysis
- 🔮 **User Inference**: Test individual user scenarios with drug interaction checking
- 🛠️ **Utility Functions**: Helper functions for custom analysis

### Getting Started

1. **For Google Colab**: Upload `ai_medication_reminder_colab.ipynb` and run all cells
2. **For Local Development**: Use `python main.py --run-pipeline`
3. **For Custom Analysis**: Import modules and use the Pipeline class

For detailed documentation, see the `COLAB_SETUP.md` file and inline code documentation.

## 1) Input → Output
- **Input**: `data/logs.csv` (M2).
- **Output**:
  1) Lịch nhắc cá nhân hóa “hôm nay”.
  2) Điểm rủi ro bỏ liều 24–72h từng nhắc.
  3) Báo cáo hiệu năng: AUC, PR-AUC, F1, calibration (ECE).
  4) Báo cáo **công bằng theo trục hoạt động**: theo kênh, theo time-bucket (morning/afternoon/evening), weekday vs weekend.
  5) DDI toy: `prescription_demo.json` → cảnh báo `(pair, severity, note)`.

## 2) Schema dữ liệu (M2)
Bắt buộc: `user_pid, ts_reminder, tz_offset, channel, responded_within_2h`  
Tùy chọn: `delivered, ack_latency_sec, snooze`

| cột                 | kiểu     | mô tả                                    |
| ------------------- | -------- | ---------------------------------------- |
| user_pid            | str      | mã giả (UUID4 hoặc u_01, u_02, …)        |
| ts_reminder         | str      | UTC ISO8601 (vd: `2025-08-01T06:30:00Z`) |
| tz_offset           | str      | múi giờ offset (vd: `+07:00`)            |
| channel             | str      | `push` \| `SMS` \| `voice`               |
| delivered           | 0/1      | đã gửi thành công                        |
| responded_within_2h | 0/1      | xác nhận trong 2 giờ                     |
| ack_latency_sec     | float/NA | độ trễ xác nhận                          |
| snooze              | 0/1/NA   | hoãn nhắc                                |

**Ví dụ 3 dòng**
```csv
user_pid,ts_reminder,tz_offset,channel,delivered,responded_within_2h,ack_latency_sec,snooze
u_01,2025-08-01T06:30:00Z,+07:00,push,1,1,420,0
u_01,2025-08-01T21:00:00Z,+07:00,SMS,1,0,,0
u_02,2025-08-02T07:15:00Z,+07:00,voice,1,1,180,0