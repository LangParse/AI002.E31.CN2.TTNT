#!/usr/bin/env python3
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

from src import Pipeline


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

    parser.add_argument(
        "--config-file", type=Path, help="Path to custom configuration file"
    )

    # Inference options
    parser.add_argument(
        "--inference", action="store_true", help="Run inference for a single user"
    )

    parser.add_argument(
        "--user-data", type=str, help="JSON string with user context data for inference"
    )

    parser.add_argument(
        "--medications",
        nargs="+",
        help="List of medications for drug interaction checking",
    )

    # Utility options
    parser.add_argument(
        "--validate-setup",
        action="store_true",
        help="Validate system setup and dependencies",
    )

    args = parser.parse_args()

    # Validate setup if requested
    if args.validate_setup:
        validate_setup()
        return

    # Create configuration
    config = create_config(args)

    # Initialize pipeline
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

        else:
            # Show help if no action specified
            parser.print_help()

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


def create_config(args):
    """Create configuration from command line arguments."""
    from src import Config

    if args.config_file and args.config_file.exists():
        # Load from file (not implemented in this version)
        print(f"Loading config from {args.config_file}")
        config = Config.from_env()
    else:
        # Create from environment
        config = Config.from_env()

    # Override with command line arguments
    if args.data_scale:
        config.env.data_scale = args.data_scale

    return config


def validate_setup():
    """Validate system setup and dependencies."""
    print("Validating AI Medication Reminder Setup...")
    print("=" * 50)

    # Check Python version
    import sys

    print(f"Python Version: {sys.version}")

    # Check required packages
    required_packages = ["pandas", "numpy", "scikit-learn", "matplotlib"]

    optional_packages = [
        ("torch", "PyTorch (for TinyTemporal model)"),
    ]

    missing_required = []
    missing_optional = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (REQUIRED)")
            missing_required.append(package)

    for package, description in optional_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"- {package} (optional: {description})")
            missing_optional.append(package)

    # Check environment
    from src import Config

    config = Config.from_env()
    print(f"\nEnvironment: {'Colab' if config.env.in_colab else 'Local'}")
    print(f"Data Scale: {config.env.data_scale}")
    print(f"GPU Available: {config.env.has_gpu}")

    # Summary
    print("\n" + "=" * 50)
    if missing_required:
        print(
            f"❌ Setup incomplete. Missing required packages: {', '.join(missing_required)}"
        )
        print("Install with: pip install " + " ".join(missing_required))
        sys.exit(1)
    else:
        print("✅ Setup validation passed!")
        if missing_optional:
            print(
                f"Note: Optional packages not installed: {', '.join(missing_optional)}"
            )


if __name__ == "__main__":
    main()
