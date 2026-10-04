"""Point the dashboard and alert rule at the real PSI metric.

Usage: python monitoring/set_psi_metric.py <metric_name> <label_name>
Example: python monitoring/set_psi_metric.py arabic_sentiment_psi feature

Run it once. The files ship with the placeholders psi_score and feature.
"""

import sys
from pathlib import Path

MON = Path(__file__).resolve().parent
FILES = [
    MON / "alert_rules.yml",
    MON / "grafana" / "dashboards" / "arabic_sentiment.json",
]


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    metric, label = sys.argv[1], sys.argv[2]
    for path in FILES:
        text = path.read_text(encoding="utf-8")
        text = text.replace("psi_score", metric)
        text = text.replace("{{feature}}", "{{" + label + "}}")
        text = text.replace("$labels.feature", "$labels." + label)
        path.write_text(text, encoding="utf-8")
        print(f"updated {path.name}")


if __name__ == "__main__":
    main()
