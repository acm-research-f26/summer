
import json
import os
import sys

if __name__ == "__main__":
    snippets_path, predictions_path = sys.argv[1], sys.argv[2]

    # iteration1 root must win module resolution (its `config`, its `src`).
    it1_root = os.getcwd()
    sys.path.insert(0, it1_root)

    from src.inference import predict_batch  # iteration1's frozen contract

    with open(snippets_path) as f:
        texts = json.load(f)["texts"]

    preds = predict_batch(texts)

    with open(predictions_path, "w") as f:
        json.dump({"predictions": preds}, f)

    print(f"it1_worker: predicted {len(preds)} snippets", file=sys.stderr)
