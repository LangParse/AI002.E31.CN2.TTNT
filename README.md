# AI Medication Reminder

This project is an AI-powered assistant that predicts and reminds you to take your medication based on your daily context.

## Features
- Predicts the likelihood of missing a medication dose using AI
- Provides reminders with different urgency levels
- Simple command-line interface

## Requirements
- Python 3.8 or newer
- See `requirements.txt` for Python dependencies

## Setup
1. **Clone or download this repository.**
2. **Navigate to the project directory:**
   ```sh
   cd /Users/taidotrong/Desktop/Learning/AI-thinking
   ```
3. **Install required Python libraries:**
   ```sh
   pip install -r requirements.txt
   ```
4. **Ensure the file `medication_model.pkl` is present in the project directory.**

## Usage
Run the main script:
```sh
python main.py
```

You will be prompted to enter:
- The day of the week (0=Monday, 1=Tuesday, ..., 6=Sunday)
- Time of day (0=Morning, 1=Evening)
- Your systolic blood pressure

The AI will analyze your input and provide a reminder based on the predicted risk of missing your medication.

## Notes
- The model file (`medication_model.pkl`) must be present in the same directory as `main.py`.
- If you want to retrain the model, use `train_model.py` (see script for details).

## License
This project is for educational purposes. 