#!/usr/bin/env python3
"""
Advanced Gradio UI for AI Medication Reminder System
"""

import json
import sys
from pathlib import Path

import gradio as gr

from src.config import Config

# Add src to path to import our modules
sys.path.append(str(Path(__file__).parent / "src"))


def run_inference(user_data_json, medications_json):
    """
    Run inference on user data and medications
    """
    try:
        # Import here to avoid issues at startup
        from src.pipeline import Pipeline

        # Initialize pipeline
        config = Config.from_env()
        pipeline = Pipeline(config)

        # Parse input JSON
        try:
            user_data = json.loads(user_data_json) if user_data_json.strip() else {}
        except json.JSONDecodeError as e:
            return json.dumps(
                {
                    "error": f"❌ Invalid user_data JSON: {str(e)}",
                    "recommendations": None,
                    "warnings": [],
                },
                indent=2,
            )

        try:
            medications = (
                json.loads(medications_json) if medications_json.strip() else []
            )
        except json.JSONDecodeError as e:
            return json.dumps(
                {
                    "error": f"❌ Invalid medications JSON: {str(e)}",
                    "recommendations": None,
                    "warnings": [],
                },
                indent=2,
            )

        # Run inference
        results = pipeline.run_inference(user_data, medications)

        # Add success indicator
        if results.get("recommendations") and not results["recommendations"].get(
            "error"
        ):
            results["status"] = "Inference completed successfully"

        # Return formatted JSON
        return json.dumps(results, indent=2)

    except Exception as e:
        import traceback

        return json.dumps(
            {
                "status": "Inference failed",
                "error": f"System error: {str(e)}",
                "traceback": traceback.format_exc()[-1000:],  # Last 1000 chars only
                "recommendations": {
                    "recommended_channel": "push",
                    "response_probability": 0.5,
                    "confidence": "low",
                    "error": "System failure - using defaults",
                },
                "warnings": [],
            },
            indent=2,
        )


# Example data templates
examples = {
    "morning_user": {
        "hour": 8,
        "dow": 1,  # Monday
        "age": 35,
        "ctr7": 0.85,
        "ctr14": 0.80,
        "hours_since_prev": 24,
        "ack_latency_sec": 120,
        "tz_offset_hours": -5.0,
        "channel": "push",
        "conditions": [],
        "pregnancy_status": False,
    },
    "evening_senior": {
        "hour": 20,
        "dow": 5,  # Friday
        "age": 68,
        "ctr7": 0.45,
        "ctr14": 0.42,
        "hours_since_prev": 8,
        "ack_latency_sec": 600,
        "tz_offset_hours": 2.0,
        "channel": "SMS",
        "conditions": ["diabetes", "hypertension"],
        "pregnancy_status": False,
    },
    "busy_professional": {
        "hour": 14,
        "dow": 3,  # Wednesday
        "age": 42,
        "ctr7": 0.25,
        "ctr14": 0.30,
        "hours_since_prev": 6,
        "ack_latency_sec": 1200,
        "tz_offset_hours": 8.0,
        "channel": "voice",
        "conditions": ["anxiety"],
        "pregnancy_status": False,
    },
}

medications_examples = {
    "simple": ["aspirin", "multivitamin"],
    "cardiac": ["aspirin", "lisinopril", "atorvastatin"],
    "complex": ["warfarin", "aspirin", "metformin", "lisinopril", "omeprazole"],
    "pregnancy": ["prenatal_vitamins", "folic_acid", "iron_supplement"],
}

# CSS for better styling
css = """
.gradio-container {
    max-width: 1400px !important;
    margin: auto;
}
.input-group {
    border: 2px solid #e1e5e9;
    border-radius: 12px;
    padding: 20px;
    margin: 10px 0;
    background: #f8f9fa;
}
.output-group {
    border: 2px solid #d4edda;
    border-radius: 12px;
    padding: 20px;
    margin: 10px 0;
    background: #f8fff9;
}
.example-btn {
    margin: 5px;
    font-size: 12px;
}
.main-btn {
    font-size: 16px;
    padding: 12px 24px;
    margin: 10px 0;
}
"""

# Create advanced interface
with gr.Blocks(title="🏥 AI Medication Reminder System", css=css) as demo:
    # Header
    gr.Markdown("""
    # 🏥 AI Medication Reminder System
    ### Personalized Channel Recommendation & Drug Safety Checker
    
    This system uses machine learning to recommend the optimal communication channel (Push, Email, SMS) 
    for medication reminders based on user behavior patterns and contextual factors.
    """)

    with gr.Row():
        # Input Column
        with gr.Column(scale=1):
            gr.Markdown("## 📝 Input Data")

            # User Data Section
            with gr.Group():
                gr.Markdown("### 👤 User Context Data")
                gr.Markdown("""
                **Available Fields:**
                - `hour`: Hour of day (0-23)
                - `dow`: Day of week (0=Monday, 6=Sunday)  
                - `age`: User age
                - `ctr7/ctr14`: Click-through rates (7/14 days)
                - `hours_since_prev`: Hours since last reminder
                - `ack_latency_sec`: Response time in seconds
                - `tz_offset_hours`: Timezone offset from UTC
                - `channel`: Current channel preference
                - `conditions`: Medical conditions list
                - `pregnancy_status`: Boolean
                """)

                user_data_input = gr.Code(
                    label="User Data (JSON)",
                    value=json.dumps(examples["morning_user"], indent=2),
                    language="json",
                    lines=12,
                )

                # User example buttons
                gr.Markdown("**Quick Examples:**")
                with gr.Row():

                    def load_example(example_name):
                        return json.dumps(examples[example_name], indent=2)

                    morning_btn = gr.Button("🌅 Morning User", size="sm")
                    evening_btn = gr.Button("🌃 Evening Senior", size="sm")
                    busy_btn = gr.Button("💼 Busy Professional", size="sm")

                    morning_btn.click(
                        lambda: load_example("morning_user"), outputs=user_data_input
                    )
                    evening_btn.click(
                        lambda: load_example("evening_senior"), outputs=user_data_input
                    )
                    busy_btn.click(
                        lambda: load_example("busy_professional"),
                        outputs=user_data_input,
                    )

            # Medications Section
            with gr.Group():
                gr.Markdown("### 💊 Medications List")
                gr.Markdown("List of current medications for drug interaction checking")

                medications_input = gr.Code(
                    label="Medications (JSON Array)",
                    value=json.dumps(medications_examples["simple"], indent=2),
                    language="json",
                    lines=6,
                )

                # Medication example buttons
                gr.Markdown("**Medication Examples:**")
                with gr.Row():

                    def load_meds(med_type):
                        return json.dumps(medications_examples[med_type], indent=2)

                    simple_meds_btn = gr.Button("💊 Simple", size="sm")
                    cardiac_meds_btn = gr.Button("❤️ Cardiac", size="sm")
                    complex_meds_btn = gr.Button("⚠️ Complex", size="sm")

                    simple_meds_btn.click(
                        lambda: load_meds("simple"), outputs=medications_input
                    )
                    cardiac_meds_btn.click(
                        lambda: load_meds("cardiac"), outputs=medications_input
                    )
                    complex_meds_btn.click(
                        lambda: load_meds("complex"), outputs=medications_input
                    )

            # Submit Button
            submit_btn = gr.Button(
                "🚀 Run AI Inference",
                variant="primary",
                size="lg",
                elem_classes="main-btn",
            )

        # Output Column
        with gr.Column(scale=1):
            gr.Markdown("## 📊 AI Results")

            with gr.Group():
                output = gr.Code(
                    label="Inference Results",
                    language="json",
                    lines=20,
                    interactive=False,
                )

    # Information Section
    with gr.Row():
        with gr.Column():
            gr.Markdown("""
            ### 📋 Result Interpretation
            
            **🎯 Recommendations:**
            - `recommended_channel`: Optimal channel (push/email/sms)
            - `response_probability`: Likelihood of user responding (0-1)
            - `confidence`: Model confidence (high/medium/low)
            
            **⚠️ Warnings:**
            - Drug-drug interaction alerts
            - Contraindication warnings
            - Safety recommendations
            
            **🔍 Features Used:**
            - Temporal patterns (time of day, day of week)
            - Behavioral history (CTR, response latency)
            - User preferences and demographics
            - Contextual factors (timezone, device)
            """)

        with gr.Column():
            gr.Markdown("""
            ### 🧠 Model Information
            
            **🤖 ML Pipeline:**
            - **Feature Engineering**: 13 engineered features
            - **Model**: Logistic Regression (Baseline)
            - **Bandit**: Epsilon-Greedy for channel selection
            - **Accuracy**: ~77.6% on test data
            
            **📈 Performance:**
            - **Precision**: 77.6%
            - **Recall**: 100%
            - **F1-Score**: 87.4%
            - **ROC-AUC**: 61.7%
            
            **🎲 Bandit Results:**
            - **Epsilon-Greedy**: 78% response rate
            - **Random Baseline**: 75% response rate  
            - **Improvement**: +3% over random selection
            """)

    # Connect the inference function
    submit_btn.click(
        fn=run_inference, inputs=[user_data_input, medications_input], outputs=output
    )

if __name__ == "__main__":
    print("🚀 Starting AI Medication Reminder System...")
    print("📊 Loading ML models...")
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=True,  # Set to True for public sharing
        debug=True,
        show_api=True,
    )
