"""
Main entry point for AI Medication Reminder system.

This script provides a command-line interface for running the complete pipeline
or individual components.
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import Config
from src.pipeline import Pipeline


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AI Medication Reminder System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
            python main.py --run-pipeline                    # Run full pipeline
            python main.py --run-pipeline --force-retrain   # Force retrain models
            python main.py --inference --user-data '{"hour": 9, "dow": 1}'  # Run inference
            python main.py --data-scale LARGE               # Use large dataset
        """,
    )

    # Pipeline options
    parser.add_argument(
        "--run-pipeline", action="store_true", help="Run the complete pipeline"
    )

    parser.add_argument(
        "--force-retrain", action="store_true", help="Force retraining of models"
    )

    # Configuration options
    parser.add_argument(
        "--data-scale",
        choices=["SMALL", "LARGE"],
        help="Data scale to use (overrides environment detection)",
    )

    # Inference options require must have user_data
    parser.add_argument(
        "--inference", action="store_true", help="Run inference for a single user"
    )

    parser.add_argument(
        "--user-data",
        type=str,
        help="JSON string with user context data for inference",
    )

    parser.add_argument(
        "--medications",
        nargs="+",
        help="List of medications for drug interaction checking",
    )

    args = parser.parse_args()

    # Initialize pipeline
    config = Config.from_env()
    pipeline = Pipeline(config)

    try:
        if args.run_pipeline:
            # Run full pipeline
            print("Starting AI Medication Reminder Pipeline...")
            results = pipeline.run_full_pipeline(force_retrain=args.force_retrain)
            print("\nPipeline completed successfully!")

        elif args.inference:
            # Run inference
            if not args.user_data:
                print("Error: --user-data required for inference")
                sys.exit(1)

            import json

            try:
                user_data = json.loads(args.user_data)
            except json.JSONDecodeError as e:
                print(f"Error parsing user data JSON: {e}")
                sys.exit(1)

            results = pipeline.run_inference(user_data, medications=args.medications)

            print("\nInference Results:")
            print("=" * 50)

            if results["recommendations"]["error"]:
                print(f"Error: {results['recommendations']['error']}")
                sys.exit(1)

            print(
                f"Recommended Channel: {results['recommendations']['recommended_channel']}"
            )
            print(
                f"Response Probability: {results['recommendations']['response_probability']:.3f}"
            )
            print(f"Confidence: {results['recommendations']['confidence']}")

            if results["warnings"]:
                print(f"\nWarnings ({len(results['warnings'])}):")
                for warning in results["warnings"]:
                    print(f"  - {warning.get('description', warning)}")

        elif args.data_scale:
            generate_synthetic_data(args.data_scale)
            print("\nData generation completed successfully!")

        else:
            # Show help if no action specified
            parser.print_help()

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


def generate_synthetic_data(data_scale: str = "SMALL"):
    """Generate synthetic data from command line arguments."""
    from src.data.generator import SyntheticDataGenerator

    config = Config.from_env()
    synthetic_generator = SyntheticDataGenerator(config.data, config.env.seed)
    data_dir_path = config.paths.data_dir or (config.paths.base_dir / "data")
    df = synthetic_generator.save_synthetic_data(
        data_dir_path / f"logs_{data_scale.lower()}.csv", data_scale
    )
    return df


if __name__ == "__main__":
    main()
