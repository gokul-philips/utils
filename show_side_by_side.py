import argparse
import cv2
import os
import re
import numpy as np
import time

# Matches the "..._Frame_<frame-number>_<date>_<timestamp>.<ext>" suffix, e.g.
# "Image_Frame_3441700_20260729_1616.jpg" -> key "3441700_20260729_1616"
FRAME_KEY_RE = re.compile(r"_frame_(\d+_\d+_\d+)", re.IGNORECASE)
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Show (or save) images from two directories side by side."
    )
    parser.add_argument(
        "--path1",
        required=True,
        help="First directory of images.",
    )
    parser.add_argument(
        "--path2",
        required=True,
        help="Second directory of images.",
    )
    parser.add_argument(
        "--mode",
        choices=["display", "save"],
        default="display",
        help="'display' shows an interactive window (Windows only, needs a monitor). "
        "'save' writes the combined side-by-side images to --output-dir instead.",
    )
    parser.add_argument(
        "--output-dir",
        default="side_by_side_output",
        help="Directory to save combined images to when --mode=save.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1920,
        help="Canvas width to use when --mode=save (ignored in display mode).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1080,
        help="Canvas height to use when --mode=save (ignored in display mode).",
    )
    return parser.parse_args()


# Function to resize images while maintaining aspect ratio
def resize_image(image, max_width, max_height):
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h)
    return cv2.resize(image, (int(w * scale), int(h * scale)))


def build_canvas(img1, img2, canvas_width, canvas_height):
    img1_resized = resize_image(img1, canvas_width // 2, canvas_height)
    img2_resized = resize_image(img2, canvas_width // 2, canvas_height)

    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    canvas[: img1_resized.shape[0], : img1_resized.shape[1]] = img1_resized
    canvas[
        : img2_resized.shape[0], canvas_width // 2 : canvas_width // 2 + img2_resized.shape[1]
    ] = img2_resized
    return canvas


def extract_frame_key(filename):
    """Extract the '<frame-number>_<date>_<timestamp>' key from a filename like
    '..._Frame_3441700_20260729_1616.jpg', ignoring the rest of the path/name
    and the file extension (png/jpg/jpeg, case-insensitive)."""
    match = FRAME_KEY_RE.search(filename)
    return match.group(1) if match else None


def index_images_by_frame_key(path):
    """Map frame key -> filename for all png/jpg images in a directory."""
    index = {}
    for fn in os.listdir(path):
        if not fn.lower().endswith(IMAGE_EXTENSIONS):
            continue
        key = extract_frame_key(fn)
        if key is None:
            print(f"Skipping (no frame/date/timestamp match): {fn}")
            continue
        index[key] = fn
    return index


def get_common_filenames(path1, path2):
    index1 = index_images_by_frame_key(path1)
    index2 = index_images_by_frame_key(path2)
    print(f"Images in path1: {list(index1.values())}")
    print(f"Images in path2: {list(index2.values())}")

    common_keys = sorted(set(index1) & set(index2))
    pairs = [(index1[key], index2[key]) for key in common_keys]
    print(f"Common frame/date/timestamp matches: {pairs}")

    if not pairs:
        raise Exception("No matching images (by frame/date/timestamp) found in the provided directories.")
    return pairs


def run_display_mode(path1, path2, pairs):
    # Display mode works on both Windows and Linux (requires a display/GUI environment).
    from screeninfo import get_monitors

    # Get monitor information
    try:
        monitors = get_monitors()
    except Exception as exc:
        raise Exception(
            "Could not detect any monitors. Display mode requires a graphical "
            "environment (e.g. an X server on Linux). Use --mode save if running "
            "headless."
        ) from exc
    print("Available monitors:", monitors)
    if not monitors:
        raise Exception(
            "No monitors detected. Display mode requires a graphical environment. "
            "Use --mode save if running headless."
        )
    if len(monitors) < 2:
        # Use the primary monitor
        primary_monitor = monitors[0]
        monitor_x = primary_monitor.x
        monitor_y = primary_monitor.y
        monitor_width = primary_monitor.width
        monitor_height = primary_monitor.height
    else:
        # Use the second monitor
        second_monitor = monitors[1]
        monitor_x = second_monitor.x
        monitor_y = second_monitor.y
        monitor_width = second_monitor.width
        monitor_height = second_monitor.height

    # Create a named window
    cv2.namedWindow("Sample", cv2.WINDOW_NORMAL)

    # Move the window to the second monitor
    cv2.moveWindow("Sample", monitor_x, monitor_y)

    # Resize the window to fit your second monitor's resolution
    cv2.resizeWindow("Sample", monitor_width, monitor_height)

    # Wait for a short period to ensure the window is displayed correctly
    time.sleep(1)

    # Resize the window again to make sure it fills the screen
    cv2.resizeWindow("Sample", monitor_width, monitor_height)

    # Loop through the images and display them side by side
    i = 0
    while True:
        fn1_name, fn2_name = pairs[i]
        fn1 = os.path.join(path1, fn1_name)
        fn2 = os.path.join(path2, fn2_name)

        if os.path.exists(fn1) and os.path.exists(fn2):
            img1 = cv2.imread(fn1, -1)
            img2 = cv2.imread(fn2, -1)

            if img1 is None or img2 is None:
                print(f"Error loading images: {fn1}, {fn2}")
                i = (i + 1) % len(pairs)
                continue

            canvas = build_canvas(img1, img2, monitor_width, monitor_height)

            cv2.imshow("Sample", canvas)

            k = cv2.waitKey(1) & 0xFF  # Short delay and mask to handle key press detection

            if k == ord("x") or k == ord("q"):
                break
            elif k == ord("n"):
                i = (i + 1) % len(pairs)
            elif k == ord("p"):
                i = (i - 1) % len(pairs)

    cv2.destroyAllWindows()


def run_save_mode(path1, path2, pairs, output_dir, width, height):
    os.makedirs(output_dir, exist_ok=True)

    for fn1_name, fn2_name in pairs:
        fn1 = os.path.join(path1, fn1_name)
        fn2 = os.path.join(path2, fn2_name)

        if not (os.path.exists(fn1) and os.path.exists(fn2)):
            continue

        img1 = cv2.imread(fn1, -1)
        img2 = cv2.imread(fn2, -1)

        if img1 is None or img2 is None:
            print(f"Error loading images: {fn1}, {fn2}")
            continue

        canvas = build_canvas(img1, img2, width, height)

        out_path = os.path.join(output_dir, fn1_name)
        cv2.imwrite(out_path, canvas)
        print(f"Saved: {out_path}")


def main():
    args = parse_args()
    pairs = get_common_filenames(args.path1, args.path2)

    if args.mode == "display":
        run_display_mode(args.path1, args.path2, pairs)
    else:
        run_save_mode(args.path1, args.path2, pairs, args.output_dir, args.width, args.height)


if __name__ == "__main__":
    main()
