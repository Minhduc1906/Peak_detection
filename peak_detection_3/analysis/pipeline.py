import subprocess
import sys
import os

def run_script(script_path):
    """Run a Python script and handle potential errors."""
    print(f"Running {script_path}...")
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print(f"Successfully completed {script_path}")
        print(result.stdout)
        if result.stderr:
            print(f"Warnings/Errors from {script_path}:\n{result.stderr}")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_path}:")
        print(e.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print(f"Script not found: {script_path}")
        sys.exit(1)

def main():
    """Main pipeline to run all scripts in order."""
    scripts = [
        "analysis/processes/log_processor.py",
        "analysis/processes/feature_extraction.py",
        "analysis/processes/adaptive_feature_extraction.py",
        "analysis/visualization/param_analysis.py",
        "analysis/visualization/param_analysis_enhanced.py",
        "analysis/LSTM/LSTM_EWC.py",
        # "analysis/processes/baseline_models.py", # Run this file if also want to compare against other ML models
        # "analysis/visualization/accuracy_comparison_baseline.py", # Run this file if also want to compare against other ML models
        "analysis/visualization/accuracy_comparison.py",
        "analysis/visualization/shap_analysis.py",
    ]

    # Verify all scripts exist before starting
    for script in scripts:
        if not os.path.isfile(script):
            print(f"Error: Script {script} does not exist")
            sys.exit(1)

    # Run each script in sequence
    for script in scripts:
        run_script(script)

    print("Pipeline completed successfully!")

if __name__ == "__main__":
    main()