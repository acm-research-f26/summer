import subprocess
import json
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
prolog_file = os.path.join(current_dir, "test.pl")

print(prolog_file)

def run_scasp(query: str):
    result = subprocess.run(
        ["swipl", prolog_file, query],
        capture_output=True,
        text=True,
        timeout=30
    )
    if result.returncode != 0:
        raise RuntimeError(f"swipl failed: {result.stderr}")
    return json.loads(result.stdout.strip())


if __name__ == "__main__":
    # 1. Ask "is enemy dangerous, and why" — finds X that satisfies danger(X)
    data = run_scasp("chosen_action(X)")
    print(data)