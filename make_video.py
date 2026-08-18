import os
import imageio.v3 as iio

# Define your input folder and output video name
image_folder = r"C:\Users\320329176\Downloads\MR_Clean\WorkflowCamera\WorkflowCameraVJ"
# Change this to your folder path


output_video = "visualize_json_10x.mp4"
fps = 10  # Frames per second (speed of the video)

# 1. Gather and sort all image files (supports png, jpg, jpeg)
supported_extensions = (".png", ".jpg", ".jpeg")
images = [
    os.path.join(image_folder, img)
    for img in sorted(os.listdir(image_folder))
    if img.lower().endswith(supported_extensions)
]

if not images:
    print(f"No images found in {image_folder}")
    exit()

print(f"Found {len(images)} images. Creating video...")

# 2. Read images and write them directly to a video file
# The 'fps' parameter controls how long each image stays on screen
with iio.imopen(output_video, "w", plugin="pyav") as file:
    file.init_video_stream(codec="h264", fps=fps)
    for img_path in images:
        frame = iio.imread(img_path)
        file.write_frame(frame)

print(f"Video successfully saved as {output_video}")