import json
from pathlib import Path
from functools import lru_cache

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_FILE = PROJECT_ROOT / "data" / "cit_public.json"

REQUIRED_SECTIONS = [
    "dataset_id",
    "institute",
    "calendar",
    "priority_order",
    "role_category_cap",
    "rooms",
    "room_unavailability",
    "faculty",
    "student_groups",
    "courses",
    "sessions",
    "booking_requests",
    "cancellations",
    "exams",
]


@lru_cache(maxsize=1)
def load_dataset() -> dict:
    """Load and validate the official CIT dataset."""

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Official dataset not found: {DATA_FILE}"
        )

    with DATA_FILE.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    missing = [
        section
        for section in REQUIRED_SECTIONS
        if section not in dataset
    ]

    if missing:
        raise ValueError(
            f"Dataset is missing sections: {missing}"
        )

    for section in REQUIRED_SECTIONS:
        if section not in ("dataset_id", "institute") and not isinstance(
            dataset[section], (dict, list)
        ):
            raise ValueError(
                f"Invalid data type for section: {section}"
            )

    return dataset


def get_dataset_summary() -> dict:
    """Return a summary without exposing the entire dataset."""

    dataset = load_dataset()

    collection_sections = [
        "rooms",
        "room_unavailability",
        "faculty",
        "student_groups",
        "courses",
        "sessions",
        "booking_requests",
        "cancellations",
        "exams",
    ]

    counts = {
        section: len(dataset[section])
        for section in collection_sections
    }

    return {
        "dataset_id": dataset["dataset_id"],
        "institute": dataset["institute"],
        "sections": len(dataset),
        "counts": counts,
        "status": "loaded",
    }