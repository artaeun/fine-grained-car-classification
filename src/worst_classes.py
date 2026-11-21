"""List the make classes with the lowest test F1 for one trained model.

Reads only the committed results, so it runs without the dataset:

    python src/worst_classes.py hierarchical 10

writes results/<model>/worst_classes.md and prints the same table.
"""
import json
import sys
from pathlib import Path

from sklearn.metrics import precision_recall_fscore_support

ROOT = Path(__file__).resolve().parent.parent


def main(model="hierarchical", n=10):
    data = json.loads((ROOT / "results" / model / "evaluation_data.json").read_text())
    names = json.loads((ROOT / "results" / "visualization_metadata.json").read_text())["make_idx_to_name"]
    names = {k: v.strip() for k, v in names.items()}  # names as CompCars spells them
    y_true, y_pred = data["make_labels"], data["make_predictions"]
    labels = sorted(set(y_true))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)

    rows = sorted(zip(labels, p, r, f, s), key=lambda x: x[3])[:n]
    lines = [
        f"# The {n} make classes with the lowest test F1, model `{model}`",
        "",
        "| Make | Test images | Precision | Recall | F1 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for lab, pi, ri, fi, si in rows:
        lines.append(f"| {names[str(lab)]} | {si} | {pi:.2f} | {ri:.2f} | {fi:.2f} |")
    text = "\n".join(lines) + "\n"
    (ROOT / "results" / model / "worst_classes.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "hierarchical",
         int(sys.argv[2]) if len(sys.argv) > 2 else 10)
