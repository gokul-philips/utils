#!/usr/bin/env python3
"""Extract embedded debug images from Inference_Frame_*.json files.

Each Inference JSON captures a `debugImages` map containing the *actual*
inputs the mr-camera-workflow-algos pipeline consumed for that frame -- not
the raw camera source JPEG (`Image_Frame_*.jpg`). This script decodes those
payloads and writes them out as PNGs, split by kind:

  - patientImage : the resized/preprocessed RGB frame that was fed to the
                    AI (Triton) body/coil inference -- this is the image the
                    algorithm actually "sees", which can differ in size and
                    content from the native camera source image.
  - resizedDepth  : the corresponding depth frame used by the algorithm
                    (table depth estimate, collision mask, etc.).

Since a full session capture (e.g. a volunteer scan) can contain many
thousands of frames spanning multiple, unrelated use cases, `--time-start`/
`--time-end` (and `--frame-min`/`--frame-max`) let callers restrict
extraction to only the frames relevant to the time window/frame range of the
issue under investigation, instead of extracting the entire capture.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
from pathlib import Path
from typing import Any

# Matches the `..._<frameIndex>_<YYYYMMDD>_<HHMM>` suffix used by
# Inference_Frame_*.json / Image_Frame_*.jpg file names.
FRAME_NAME_RE = re.compile(r"_(?P<frame>\d+)_(?P<date>\d{8})_(?P<time>\d{4})$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract images from debug image fields in JSON files."
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing JSON files")
    parser.add_argument("output_dir", type=Path, help="Directory to write extracted images")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search for JSON files recursively",
    )
    parser.add_argument(
        "--time-start",
        help="Only process frames with embedded HHMM timestamp >= this value (e.g. 0525)",
    )
    parser.add_argument(
        "--time-end",
        help="Only process frames with embedded HHMM timestamp <= this value (e.g. 0532)",
    )
    parser.add_argument(
        "--frame-min",
        type=int,
        help="Only process frames with embedded frame index >= this value",
    )
    parser.add_argument(
        "--frame-max",
        type=int,
        help="Only process frames with embedded frame index <= this value",
    )
    return parser.parse_args()


def parse_frame_name(stem: str) -> tuple[int, str] | None:
    """Extract (frameIndex, HHMM) from a file stem, or None if it doesn't match."""
    match = FRAME_NAME_RE.search(stem)
    if not match:
        return None
    return int(match.group("frame")), match.group("time")


def matches_filters(
    stem: str,
    time_start: str | None,
    time_end: str | None,
    frame_min: int | None,
    frame_max: int | None,
) -> bool:
    if not any((time_start, time_end, frame_min is not None, frame_max is not None)):
        return True

    parsed = parse_frame_name(stem)
    if parsed is None:
        # Can't evaluate the filter against an unrecognized name; skip it so
        # that only frames we can confidently place in range are extracted.
        return False

    frame_index, time_str = parsed
    if time_start is not None and time_str < time_start:
        return False
    if time_end is not None and time_str > time_end:
        return False
    if frame_min is not None and frame_index < frame_min:
        return False
    if frame_max is not None and frame_index > frame_max:
        return False
    return True


def detect_extension(blob: bytes, mime_type: str | None, file_name: str | None) -> str:
    if file_name:
        ext = Path(file_name).suffix
        if ext:
            return ext.lower()

    if mime_type:
        mime_type = mime_type.lower()
        if "png" in mime_type:
            return ".png"
        if "jpeg" in mime_type or "jpg" in mime_type:
            return ".jpg"
        if "webp" in mime_type:
            return ".webp"
        if "gif" in mime_type:
            return ".gif"
        if "bmp" in mime_type:
            return ".bmp"

    # Lightweight magic-number checks.
    if blob.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if blob.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if blob.startswith(b"GIF87a") or blob.startswith(b"GIF89a"):
        return ".gif"
    if blob.startswith(b"RIFF") and blob[8:12] == b"WEBP":
        return ".webp"
    if blob.startswith(b"BM"):
        return ".bmp"

    return ".bin"


def decode_data_url(value: str) -> tuple[bytes, str | None]:
    # Handles strings like: data:image/png;base64,AAAA...
    if value.startswith("data:") and "," in value:
        meta, encoded = value.split(",", 1)
        mime_type = None
        if ";base64" in meta:
            mime_type = meta[5:].split(";", 1)[0] or None
            return base64.b64decode(encoded, validate=False), mime_type
    return base64.b64decode(value, validate=False), None


def parse_image_entry(entry: Any, index: int) -> tuple[str, bytes] | None:
    if isinstance(entry, str):
        blob, mime_type = decode_data_url(entry.strip())
        ext = detect_extension(blob, mime_type, None)
        return f"debug_{index:03d}{ext}", blob

    if not isinstance(entry, dict):
        return None

    data_value = None
    for data_key in ("data", "base64", "image", "image_base64", "encoded"):
        if data_key in entry and isinstance(entry[data_key], str):
            data_value = entry[data_key].strip()
            break

    if not data_value:
        return None

    file_name = None
    for name_key in ("file_name", "filename", "name", "id"):
        if name_key in entry and isinstance(entry[name_key], str) and entry[name_key].strip():
            file_name = entry[name_key].strip()
            break

    mime_type = None
    for mime_key in ("mime_type", "mime", "content_type"):
        if mime_key in entry and isinstance(entry[mime_key], str) and entry[mime_key].strip():
            mime_type = entry[mime_key].strip()
            break

    blob, data_url_mime = decode_data_url(data_value)
    if not mime_type:
        mime_type = data_url_mime

    if not file_name:
        ext = detect_extension(blob, mime_type, None)
        file_name = f"debug_{index:03d}{ext}"
    else:
        ext = Path(file_name).suffix
        if not ext:
            file_name = f"{file_name}{detect_extension(blob, mime_type, file_name)}"

    return file_name, blob


def parse_debug_images_map(section: Any) -> list[tuple[str, bytes]]:
    """Parse the expected schema: debugImages.{name}.pngBase64."""
    if not isinstance(section, dict):
        return []

    extracted: list[tuple[str, bytes]] = []
    for key, entry in section.items():
        if not isinstance(entry, dict):
            continue

        data_value = entry.get("pngBase64")
        if not isinstance(data_value, str) or not data_value.strip():
            continue

        blob, data_url_mime = decode_data_url(data_value.strip())
        ext = detect_extension(blob, data_url_mime or "image/png", f"{key}.png")
        extracted.append((f"{key}{ext}", blob))

    return extracted


def extract_debug_images(payload: dict[str, Any]) -> list[tuple[str, bytes]]:
    # Primary schema used by the dataset.
    images = parse_debug_images_map(payload.get("debugImages"))
    if images:
        return images

    # Backward-compatible fallback for legacy payloads.
    section = payload.get("debug_images")
    if section is None:
        return []

    entries: list[Any]
    if isinstance(section, list):
        entries = section
    elif isinstance(section, dict):
        entries = list(section.values())
    else:
        return []

    extracted: list[tuple[str, bytes]] = []
    for index, entry in enumerate(entries, start=1):
        parsed = parse_image_entry(entry, index)
        if parsed is not None:
            extracted.append(parsed)

    return extracted


def output_subdirectory_for(image_name: str) -> str | None:
    """Return target output subdirectory for known debug image names."""
    normalized = Path(image_name).stem.lower()

    if normalized == "patientimage":
        return "patientImage"
    if normalized in {"resizeddepth", "depth", "depthimage"}:
        return "resizedDepth"

    return None


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Input directory not found or not a directory: {input_dir}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    pattern = "**/*.json" if args.recursive else "*.json"
    json_files = sorted(input_dir.glob(pattern))

    total_scanned = 0
    total_skipped_by_filter = 0
    files_with_debug_images = 0
    total_written = 0

    for json_path in json_files:
        if not matches_filters(
            json_path.stem,
            args.time_start,
            args.time_end,
            args.frame_min,
            args.frame_max,
        ):
            total_skipped_by_filter += 1
            continue

        total_scanned += 1
        try:
            with json_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:
            print(f"Skipping {json_path} (invalid JSON): {exc}")
            continue

        images = extract_debug_images(payload)
        if not images:
            continue

        files_with_debug_images += 1
        for name, blob in images:
            subdir = output_subdirectory_for(name)
            if subdir is None:
                continue

            out_path = output_dir / subdir / f"{json_path.stem}.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(blob)
            total_written += 1
            print(f"Wrote {out_path}")

    print(
        "\nSummary: "
        f"scanned={total_scanned}, "
        f"skipped_by_filter={total_skipped_by_filter}, "
        f"with_debug_images={files_with_debug_images}, "
        f"images_written={total_written}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())