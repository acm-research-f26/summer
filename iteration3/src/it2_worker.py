
import json
import os
import sys

if __name__ == "__main__":
    out_path = sys.argv[1]

    # run with cwd=iteration2 so its config/src win module resolution
    it2_root = os.getcwd()
    sys.path.insert(0, it2_root)

    from src.dataset_contextdp import load_contextdp
    from src.dataset_own import load_own

    payload = {
        "contextdp": load_contextdp().to_dict(orient="records"),
        "own": load_own().to_dict(orient="records"),
    }

    with open(out_path, "w") as f:
        json.dump(payload, f)

    print(
        f"it2_worker: dumped {len(payload['contextdp'])} contextdp + "
        f"{len(payload['own'])} own truth rows",
        file=sys.stderr,
    )
