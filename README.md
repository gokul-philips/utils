# utils

Tools for turning a raw MR Camera Workflow evidence capture (e.g. `Volunteer21.zip`) into
concrete, inspectable evidence for the `analyzer`/`code-analyzer` agents. These are the canonical
scripts the agents use — prefer them over ad-hoc reprocessing.

Volunteer/system capture zips are long, multi-anatomy sessions with a lot of frames unrelated to
any single reported issue. Always scope to the time/frame window relevant to the issue (from
ISSUE.md) before running these over a capture.

- **`extract_debug_images.py`** — Extracts the images embedded in each
  `Inference_Frame_*.json`'s `debugImages` map: `patientImage` (the resized/preprocessed frame
  actually fed to the AI/Triton inference — **not** the raw `Image_Frame_*.jpg` camera source)
  and `resizedDepth` (the depth frame the algorithm used). Supports `--time-start`/`--time-end`
  (`HHMM`) and `--frame-min`/`--frame-max` to restrict extraction to a relevant window.

  ```
  python3 utils/extract_debug_images.py <json_dir> <out_dir> --recursive --time-start 0525 --time-end 0532
  ```

- **`visualize_json.py`** — Overlays each frame's markers, coil detections, collision mask, and
  iso-center/anatomy position (from its companion JSON) onto the extracted `patientImage`, so
  detection/computation failures can be visually confirmed rather than inferred from raw fields.

  ```
  python3 utils/visualize_json.py <patientImage_dir> --no-display -o <viz_out_dir>
  ```

- **`make_video.py`** — Stitches a folder of (visualized) frames into an MP4 for quick playback.

- **`show_side_by_side.py`** — Displays two folders of matching images side by side (Windows-only
  GUI) for manual comparison, e.g. algorithm image vs. depth, or before/after a fix.
