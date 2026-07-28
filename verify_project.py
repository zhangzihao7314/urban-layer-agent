"""One-command engineering verification for the thesis prototype."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COMMANDS = [
    [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
    [sys.executable, "evaluation/run_baseline.py"],
    [sys.executable, "evaluation/run_agent_evaluation.py"],
    [sys.executable, "evaluation/run_dialogue_evaluation.py"],
    [sys.executable, "evaluation/run_elicitation_evaluation.py"],
    [sys.executable, "evaluation/compare_versions.py"],
]


def main() -> int:
    for command in COMMANDS:
        print(f"\n> {' '.join(command)}")
        completed = subprocess.run(command, cwd=ROOT)
        if completed.returncode:
            print("Verification stopped because this step failed.")
            return completed.returncode
    print("\nAll engineering verification steps passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
