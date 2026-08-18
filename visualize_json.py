#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np


def get_object_member(value: Any, key: str) -> Optional[Any]:
    if not isinstance(value, dict):
        return None
    return value.get(key)


def get_first_object_member(value: Any, keys: list[str]) -> Optional[Any]:
    if not isinstance(value, dict):
        return None
    for key in keys:
        if key in value:
            return value[key]
    return None


def get_string_member(value: Any, key: str) -> Optional[str]:
    member = get_object_member(value, key)
    return member if isinstance(member, str) else None


def get_bool_member(value: Any, key: str) -> Optional[bool]:
    member = get_object_member(value, key)
    return member if isinstance(member, bool) else None


def get_number_member(value: Any, key: str) -> Optional[float]:
    member = get_object_member(value, key)
    if isinstance(member, (int, float)):
        return float(member)
    return None


def parse_normalized_point(point_value: Any) -> Optional[tuple[float, float]]:
    x = get_number_member(point_value, "x")
    y = get_number_member(point_value, "y")
    if x is None or y is None:
        return None
    return x, y


def parse_point3(point_value: Any) -> Optional[tuple[float, float, float]]:
    x = get_number_member(point_value, "x")
    y = get_number_member(point_value, "y")
    z = get_number_member(point_value, "z")
    if x is None or y is None or z is None:
        return None
    return x, y, z


def format_point3(point: tuple[float, float, float]) -> str:
    x, y, z = point
    return f"({x:.3f}, {y:.3f}, {z:.3f})"


def format_pixel_point(point: tuple[int, int]) -> str:
    return f"({point[0]}, {point[1]})"


def to_pixel(normalized_point: tuple[float, float], image_shape: tuple[int, int, int]) -> tuple[int, int]:
    h, w = image_shape[:2]
    return int(normalized_point[0] * w), int(normalized_point[1] * h)


def sanitize_label(label: str) -> str:
    return label.replace("_", " ")


def create_visualized_image_from_schema(
    image: np.ndarray,
    root: Any,
    show_anatomy_names: bool = True,
) -> np.ndarray:
    vis_image = image.copy()

    patient = get_object_member(root, "patient")
    marker_count = 0
    if isinstance(patient, dict):
        all_markers = get_object_member(patient, "allMarkers")
        if isinstance(all_markers, dict):
            for marker_name, marker_value in all_markers.items():
                point = parse_normalized_point(marker_value)
                if point is None:
                    continue
                pixel_point = to_pixel(point, vis_image.shape)
                cv2.circle(vis_image, pixel_point, 2, (0, 245, 0), -1)
                marker_count += 1

                if show_anatomy_names and marker_count <= 25:
                    cv2.putText(
                        vis_image,
                        sanitize_label(marker_name),
                        (pixel_point[0] + 4, pixel_point[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        (220, 245, 255),
                        1,
                    )

    coil_labels: list[str] = []
    coils = get_object_member(root, "coils")
    if isinstance(coils, list):
        for i, coil in enumerate(coils):
            if not isinstance(coil, dict):
                continue

            coil_type = get_string_member(coil, "type")
            is_unwanted = get_bool_member(coil, "isUnwantedCoil") or False

            label = f"Coil {i + 1}"
            if coil_type:
                label += f": {coil_type}"
            if is_unwanted:
                label += " (unwanted)"
            coil_labels.append(label)

            coil_color = (0, 0, 255) if is_unwanted else (0, 255, 0)
            position_value = get_object_member(coil, "position")
            if not isinstance(position_value, list) or not position_value:
                continue

            points: list[tuple[int, int]] = []
            for corner in position_value:
                point = parse_normalized_point(corner)
                if point is None:
                    continue
                points.append(to_pixel(point, vis_image.shape))

            if not points:
                continue

            for point in points:
                cv2.circle(vis_image, point, 3, coil_color, 2)

            if len(points) >= 2:
                for p in range(len(points) - 1):
                    cv2.line(vis_image, points[p], points[p + 1], coil_color, 1)
                cv2.line(vis_image, points[-1], points[0], coil_color, 1)

    collision_mask_value = get_object_member(root, "collisionmask")
    if isinstance(collision_mask_value, list) and collision_mask_value:
        first_row = collision_mask_value[0]
        if isinstance(first_row, list) and first_row:
            height = len(collision_mask_value)
            width = len(first_row)
            collision_mask = np.zeros((height, width), dtype=np.uint8)

            for r, row in enumerate(collision_mask_value):
                if not isinstance(row, list):
                    continue
                max_cols = min(width, len(row))
                for c in range(max_cols):
                    value = row[c]
                    if isinstance(value, (int, float)) and value > 0.0:
                        collision_mask[r, c] = 255

            collision_mask_resized = cv2.resize(
                collision_mask,
                (vis_image.shape[1], vis_image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

            red_overlay = np.full_like(vis_image, (0, 0, 255))
            blended = cv2.addWeighted(vis_image, 0.75, red_overlay, 0.25, 0.0)
            vis_image[collision_mask_resized > 0] = blended[collision_mask_resized > 0]

            collision_pixels = int(np.count_nonzero(collision_mask_resized))
            if collision_pixels > 0:
                warning = "COLLISION DETECTED"
                (text_w, text_h), baseline = cv2.getTextSize(
                    warning,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    2,
                )
                text_org = ((vis_image.shape[1] - text_w) // 2, 55)
                cv2.rectangle(
                    vis_image,
                    (text_org[0] - 10, text_org[1] - text_h - 8),
                    (text_org[0] + text_w + 10, text_org[1] + baseline + 6),
                    (0, 0, 200),
                    -1,
                )
                cv2.putText(
                    vis_image,
                    warning,
                    text_org,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

    if isinstance(patient, dict):
        orientation = get_string_member(patient, "orientation")
        if orientation:
            text = f"Patient Orientation: {orientation}"
            (text_w, text_h), baseline = cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                2,
            )
            text_org = ((vis_image.shape[1] - text_w) // 2, vis_image.shape[0] - 18)
            cv2.rectangle(
                vis_image,
                (text_org[0] - 6, text_org[1] - text_h - 6),
                (text_org[0] + text_w + 6, text_org[1] + baseline + 4),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                vis_image,
                text,
                text_org,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

    if coil_labels:
        line_height = 24
        legend_top = 40
        max_width = 0
        for coil_label in coil_labels:
            (text_w, _), _ = cv2.getTextSize(coil_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            max_width = max(max_width, text_w)

        cv2.rectangle(
            vis_image,
            (8, legend_top),
            (8 + max_width + 24, legend_top + line_height * len(coil_labels) + 12),
            (0, 0, 0),
            -1,
        )

        for i, coil_label in enumerate(coil_labels):
            cv2.putText(
                vis_image,
                coil_label,
                (16, legend_top + 20 + i * line_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

    if marker_count > 0:
        marker_summary = f"Markers: {marker_count}"
        cv2.putText(
            vis_image,
            marker_summary,
            (12, vis_image.shape[0] - 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 245, 255),
            1,
        )

    workflow = get_object_member(root, "workflow")
    anatomy_position_value = None
    anatomy_position_on_image_value = None
    iso_center_on_image_value = None

    if isinstance(workflow, dict):
        anatomy_position_value = get_first_object_member(
            workflow,
            ["anatomyPosition", "anatomyposition", "AnatomyPosition"],
        )
        anatomy_position_on_image_value = get_first_object_member(
            workflow,
            ["anatomyPositionOnImage", "anatomypositiononimage", "AnatomyPositionOnImage"],
        )
        iso_center_on_image_value = get_first_object_member(
            workflow,
            ["isoCenterPositionOnImage", "isocenterpositiononimage", "IsoCenterPositionOnImage"],
        )

    if anatomy_position_value is None and isinstance(patient, dict):
        anatomy_position_value = get_first_object_member(
            patient,
            ["anatomyPosition", "anatomyposition", "AnatomyPosition"],
        )

    if anatomy_position_on_image_value is None and isinstance(patient, dict):
        anatomy_position_on_image_value = get_first_object_member(
            patient,
            ["anatomyPositionOnImage", "anatomypositiononimage", "AnatomyPositionOnImage"],
        )

    if anatomy_position_value is None:
        anatomy_position_value = get_first_object_member(
            root,
            ["anatomyPosition", "anatomyposition", "AnatomyPosition"],
        )

    if anatomy_position_on_image_value is None:
        anatomy_position_on_image_value = get_first_object_member(
            root,
            ["anatomyPositionOnImage", "anatomypositiononimage", "AnatomyPositionOnImage"],
        )

    if iso_center_on_image_value is None:
        iso_center_on_image_value = get_first_object_member(
            root,
            ["isoCenterPositionOnImage", "isocenterpositiononimage", "IsoCenterPositionOnImage"],
        )

    top_right_lines: list[str] = []
    anatomy_point3_for_overlay = None
    if anatomy_position_value is not None:
        anatomy_point3_for_overlay = parse_point3(anatomy_position_value)

    anatomy_point_for_marker = None
    if anatomy_position_on_image_value is not None:
        anatomy_point_for_marker = parse_normalized_point(anatomy_position_on_image_value)
    if anatomy_point_for_marker is None and anatomy_position_value is not None:
        anatomy_point_for_marker = parse_normalized_point(anatomy_position_value)

    anatomy_pixel_for_overlay = None
    if anatomy_point_for_marker is not None:
        anatomy_pixel = to_pixel(anatomy_point_for_marker, vis_image.shape)
        anatomy_pixel_for_overlay = anatomy_pixel
        cv2.circle(vis_image, anatomy_pixel, 8, (0, 0, 255), 2)
        cv2.circle(vis_image, anatomy_pixel, 2, (220, 220, 255), -1)

    if anatomy_pixel_for_overlay is not None:
        top_right_lines.append(f"anatomyPosition: {format_pixel_point(anatomy_pixel_for_overlay)}")
    elif anatomy_point3_for_overlay is not None:
        top_right_lines.append(f"anatomyPosition: {format_point3(anatomy_point3_for_overlay)}")

    if iso_center_on_image_value is not None:
        iso_point = parse_normalized_point(iso_center_on_image_value)
        if iso_point is not None:
            iso_pixel = to_pixel(iso_point, vis_image.shape)
            top_right_lines.append(
                f"isoCenterPositionOnImage: {format_pixel_point(iso_pixel)}"
            )
            cv2.circle(vis_image, iso_pixel, 8, (139, 0, 0), 2)
            cv2.circle(vis_image, iso_pixel, 2, (255, 230, 200), -1)

    if top_right_lines:
        overlay_font_scale = 0.55
        overlay_thickness = 2
        overlay_margin = 12
        line_gap = 8

        max_width = 0
        total_height = 0
        line_sizes: list[tuple[int, int, int]] = []

        for line in top_right_lines:
            (text_w, text_h), baseline = cv2.getTextSize(
                line,
                cv2.FONT_HERSHEY_SIMPLEX,
                overlay_font_scale,
                overlay_thickness,
            )
            line_sizes.append((text_w, text_h, baseline))
            max_width = max(max_width, text_w)
            total_height += text_h + baseline + line_gap

        if line_sizes:
            total_height -= line_gap

        box_x = max(0, vis_image.shape[1] - max_width - (overlay_margin * 2))
        box_y = overlay_margin
        box_w = min(vis_image.shape[1] - box_x, max_width + (overlay_margin * 2))
        box_h = min(vis_image.shape[0] - box_y, total_height + (overlay_margin * 2))

        cv2.rectangle(
            vis_image,
            (box_x, box_y),
            (box_x + box_w, box_y + box_h),
            (0, 0, 0),
            -1,
        )

        y = box_y + overlay_margin
        for line, (text_w, text_h, baseline) in zip(top_right_lines, line_sizes):
            text_x = vis_image.shape[1] - overlay_margin - text_w
            text_y = y + text_h
            cv2.putText(
                vis_image,
                line,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                overlay_font_scale,
                (255, 255, 255),
                overlay_thickness,
            )
            y += text_h + baseline + line_gap

    return vis_image


def find_json_for_image(image_path: Path) -> Optional[Path]:
    direct_json = image_path.with_suffix(".json")
    if direct_json.exists():
        return direct_json

    filename = direct_json.name
    image_token = filename.find("Image")
    if image_token != -1:
        inference_name = filename[:image_token] + "Inference" + filename[image_token + 5 :]
        inference_json = direct_json.parent / inference_name
        if inference_json.exists():
            return inference_json

    return None


def load_json_file(json_path: Path) -> Optional[Any]:
    try:
        with json_path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return None


def view_images_with_json_overlay(
    folder_path: Path,
    show_anatomy_names: bool,
    auto_play: bool,
    output_directory: Optional[Path],
    show_window: bool,
) -> int:
    image_files = [
        p for p in folder_path.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ]

    if not image_files:
        print(f"No image files found in: {folder_path}")
        return -1

    image_files.sort()

    write_output = output_directory is not None

    window_name = "Schema Visualizer - N: next, P: previous, Q: quit"
    window_created = False
    if show_window:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            window_created = True
        except cv2.error:
            print("OpenCV GUI backend is unavailable. Switching to headless mode.")
            show_window = False

    if not show_window and output_directory is None:
        output_directory = folder_path / "visualized_output"
        print(f"No output directory provided. Writing images to: {output_directory}")

    write_output = output_directory is not None
    if write_output and output_directory is not None:
        output_directory.mkdir(parents=True, exist_ok=True)

    index = 0
    while 0 <= index < len(image_files):
        image_path = image_files[index]
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Failed to load image: {image_path}")
            index += 1
            continue

        display = image.copy()
        json_path = find_json_for_image(image_path)
        if json_path is not None:
            root = load_json_file(json_path)
            if root is not None:
                try:
                    display = create_visualized_image_from_schema(
                        image,
                        root,
                        show_anatomy_names,
                    )
                except Exception as ex:
                    print(f"Failed to render JSON ('{json_path}'): {ex}")

        frame_label = f"{image_path.name} [{index + 1}/{len(image_files)}]"
        cv2.putText(
            display,
            frame_label,
            (12, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

        if write_output and output_directory is not None:
            out_name = output_directory / f"{image_path.stem}_schema_visualized.png"
            cv2.imwrite(str(out_name), display)

        if not show_window:
            index += 1
            continue

        cv2.imshow(window_name, display)

        if auto_play:
            cv2.waitKey(100)
            index += 1
            continue

        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        if key in (ord("p"), ord("P")):
            index -= 1
        else:
            index += 1

        if index < 0:
            index = 0
        if index >= len(image_files):
            index = len(image_files) - 1

    if window_created:
        cv2.destroyAllWindows()
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize schema JSON overlays for images in a folder."
    )
    parser.add_argument("folder_path", help="Folder containing images and JSON files")
    parser.add_argument(
        "-a",
        "--auto-play",
        action="store_true",
        help="Automatically advance every 100ms",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_directory",
        help="Write visualized images to output directory",
    )
    parser.add_argument(
        "--hide-marker-names",
        action="store_true",
        help="Show marker points without text labels",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable GUI window and only write output images",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    folder_path = Path(args.folder_path)
    if not folder_path.exists() or not folder_path.is_dir():
        print(f"Invalid folder path: {folder_path}")
        return -1

    output_directory = Path(args.output_directory) if args.output_directory else None
    show_anatomy_names = not args.hide_marker_names

    return view_images_with_json_overlay(
        folder_path=folder_path,
        show_anatomy_names=False,
        auto_play=args.auto_play,
        output_directory=output_directory,
        show_window=not args.no_display,
    )


if __name__ == "__main__":
    raise SystemExit(main())
