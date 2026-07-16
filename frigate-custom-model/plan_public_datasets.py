#!/usr/bin/env python3
"""Print a conservative public-dataset acquisition plan for the DIY Frigate model.

This script intentionally does not download data. It records what to fetch, why,
and what license checks/class mappings must happen before import.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PLAN = [
    {
        "id": "coco2017_subset",
        "source": "COCO 2017 / Ultralytics COCO YAML",
        "url": "https://docs.ultralytics.com/datasets/detect/coco/",
        "classes": ["person", "car", "truck", "dog", "cat", "bird", "bicycle", "motorcycle", "backpack", "suitcase"],
        "missing": ["package", "van", "waste_bin"],
        "use": "Baseline common-object grounding and validation examples; optionally sample rather than import all 118k train images.",
        "license_check": "Preserve COCO/source metadata. COCO annotations are widely used; individual image licenses vary.",
        "status": "recommended_v0_base",
    },
    {
        "id": "openimages_v7_targeted_subset",
        "source": "Open Images V7 bounding boxes",
        "url": "https://storage.googleapis.com/openimages/web/factsfigures_v7.html",
        "classes": ["person", "car", "truck", "van", "dog", "cat", "bird", "bicycle", "motorcycle", "backpack", "suitcase", "package_candidate", "waste_bin_candidate"],
        "missing": [],
        "use": "Targeted import for package/box-like and waste-bin-like classes after explicit class mapping review.",
        "license_check": "Filter/record image licenses; do not redistribute without respecting source licenses.",
        "status": "recommended_after_mapping",
    },
    {
        "id": "roboflow_package_permissive",
        "source": "Roboflow Universe package/parcel datasets",
        "url": "https://universe.roboflow.com/search?q=package%20detection",
        "classes": ["package"],
        "missing": [],
        "use": "Fill package-specific gap if dataset license is permissive and labels are clean.",
        "license_check": "Manual license review required per dataset. Reject unknown/no-license datasets.",
        "status": "manual_review_required",
    },
    {
        "id": "local_frigate_active_learning",
        "source": "FrontDoor and Backyard Frigate snapshots/events",
        "url": "local:/mnt/user/media/frigate_custom_model/review",
        "classes": ["all target classes when present", "empty negatives"],
        "missing": [],
        "use": "Most important domain adaptation data. Add corrections from false positives/false negatives over time.",
        "license_check": "Private local data; do not publish without explicit approval.",
        "status": "active_now",
    },
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON output file")
    args = parser.parse_args()
    print(json.dumps(PLAN, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(PLAN, indent=2) + "\n", encoding="utf-8")
        print(f"wrote={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
