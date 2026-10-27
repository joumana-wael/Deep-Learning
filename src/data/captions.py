"""Parse and clean image captions while preserving the original text."""

import re


def normalize_spacing(text):
    """Replace runs of recognized whitespace with one ordinary space."""
    return " ".join(text.split())


def propose_encoding_repair(text):
    """Propose replacing U+FFFD runs with spaces.

    This assumes the damaged characters represent word separators.
    Review the proposal before using it.
    """
    repaired = re.sub(r"\ufffd+", " ", text)
    return normalize_spacing(repaired)


def preprocess_description(description, approved_repairs=()):
    """Parse one image's description and record preprocessing changes.

    Args:
        description: Original newline-separated caption text.
        approved_repairs: One-based line numbers whose encoding repairs
            have been reviewed and approved for this description.

    Returns:
        A dictionary containing the original description, processed
        captions, a change log, and unresolved encoding-review items.
    """
    if not isinstance(description, str):
        raise TypeError("description must be a string.")

    lines = description.splitlines()
    approved_repairs = set(approved_repairs)

    if not approved_repairs.issubset(set(range(1, len(lines) + 1))):
        raise ValueError("An approved repair refers to a nonexistent line.")

    captions = []
    changes = []
    encoding_review = []

    for caption_number, original in enumerate(lines, start=1):
        cleaned = normalize_spacing(original)
        actions = []

        if cleaned != original:
            actions.append("Normalized whitespace")

        if not cleaned:
            changes.append({
                "caption_number": caption_number,
                "original": original,
                "processed": None,
                "actions": ["Removed empty caption entry"],
            })
            continue

        if "\ufffd" in cleaned:
            proposed = propose_encoding_repair(cleaned)

            if caption_number in approved_repairs:
                cleaned = proposed
                actions.append("Applied reviewed encoding repair")
            else:
                encoding_review.append({
                    "caption_number": caption_number,
                    "original": original,
                    "proposed": proposed,
                    "replacement_count": original.count("\ufffd"),
                })

        # Preserve duplicate and short captions.
        captions.append(cleaned)

        if actions:
            changes.append({
                "caption_number": caption_number,
                "original": original,
                "processed": cleaned,
                "actions": actions,
            })

    return {
        "raw_description": description,
        "captions": captions,
        "changes": changes,
        "encoding_review": encoding_review,
    }
if __name__ == "__main__":
    import json
    from pathlib import Path

    from datasets import load_dataset
    

    dataset_name = "Multimodal-Fatima/CUB_train"
    approvals_path = (
        Path(__file__).resolve().parent
        / "caption_encoding_approvals.json"
    )
    with approvals_path.open("r", encoding="utf-8") as file:
        approval_config = json.load(file)

    if approval_config["dataset"] != dataset_name:
        raise ValueError("The approvals belong to a different dataset.")

    approved_repairs_by_image = approval_config["approved_repairs"]
    applied_encoding_repairs = 0
    

    print("Loading dataset...", flush=True)
    dataset = load_dataset(dataset_name, split="train")

    # Read metadata only; caption processing does not need image decoding.
    records = dataset.select_columns([
        "id", "file_name", "label", "description"
    ])

    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    captions_path = output_dir / "captions_official_train.jsonl"
    review_path = output_dir / "caption_encoding_review.jsonl"

    image_count = 0
    caption_count = 0
    review_count = 0
    changed_caption_count = 0

    with (
        captions_path.open("w", encoding="utf-8") as captions_file,
        review_path.open("w", encoding="utf-8") as review_file,
    ):
        for record in records:
           
            approved_numbers = approved_repairs_by_image.get(
                str(record["id"]),
                [],
            )

            result = preprocess_description(
                record["description"],
                approved_repairs=approved_numbers,
            )

            applied_encoding_repairs += sum(
                "Applied reviewed encoding repair" in change["actions"]
                for change in result["changes"]
            )
            identity = {
                "dataset": dataset_name,
                "image_id": record["id"],
                "file_name": record["file_name"],
            }

            output_record = {
                **identity,
                "label": record["label"],
                **result,
            }

            captions_file.write(
                json.dumps(output_record, ensure_ascii=False) + "\n"
            )

            for issue in result["encoding_review"]:
                review_record = {
                    **identity,
                    **issue,
                }
                review_file.write(
                    json.dumps(review_record, ensure_ascii=False) + "\n"
                )

            image_count += 1
            caption_count += len(result["captions"])
            review_count += len(result["encoding_review"])
            changed_caption_count += len(result["changes"])

    print("Images processed:", image_count)
    print("Caption entries:", caption_count)
    print("Caption entries changed:", changed_caption_count)
    print("Encoding issues awaiting review:", review_count)

    print("\nCaptions saved to:", captions_path.resolve())
    print("Review items saved to:", review_path.resolve())
    print("Encoding repairs applied:", applied_encoding_repairs)