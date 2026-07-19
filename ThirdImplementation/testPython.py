import subprocess
import json

def run_scasp(query: str, pl_file="test.pl", swipl_path="swipl"):
    result = subprocess.run(
        [swipl_path, pl_file, query],
        capture_output=True,
        text=True,
        timeout=30
    )
    if result.returncode != 0:
        raise RuntimeError(f"swipl failed: {result.stderr}")
    return json.loads(result.stdout.strip())

if __name__ == "__main__":
    # 1. Ask "is enemy dangerous, and why" — finds X that satisfies danger(X)
    data = run_scasp("danger(X)")
    print(data)
    # {'bindings': {'X': 'enemy'}, 'model': ['nearby(enemy)', 'armed(enemy)', 'danger(enemy)'], 'ok': True}

    # 2. Ask a fully ground query — no variables, just true/false
    data = run_scasp("danger(enemy)")
    print(data)
    # {'bindings': {}, 'model': [...], 'ok': True}   <- succeeds, bindings empty since no vars

    # 3. Ask something that should fail
    data = run_scasp("danger(civilian)")
    print(data)
    # {'ok': False, 'error': 'no solution'}   <- no facts support this

    # 4. Ask about a sub-fact directly
    data = run_scasp("armed(X)")
    print(data)
    # {'bindings': {'X': 'enemy'}, 'model': [...], 'ok': True}