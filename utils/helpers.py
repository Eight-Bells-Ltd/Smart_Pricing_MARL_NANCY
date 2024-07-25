import os
import cv2

def create_video_from_pngs(images_folder):

    # Output video file name
    output_video = "outputs/output_video.mp4"

    # Get a list of all PNG files in the folder and their creation times
    image_files_with_dates = [(os.path.join(images_folder, file), os.stat(os.path.join(images_folder, file)).st_mtime) for file in os.listdir(images_folder) if file.endswith(".png")]

    # Sort the files based on creation date
    image_files_sorted = sorted(image_files_with_dates, key=lambda x: x[1])

    # Extract file paths from the sorted list
    image_files = [file[0] for file in image_files_sorted]

    # Read the first image to get dimensions
    first_image = cv2.imread(image_files[0])
    height, width, layers = first_image.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4 format
    fps = 0.8  # Decrease the frame rate to increase the delay between frames
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Iterate through each image and write it to the video
    for image_file in image_files:
        image = cv2.imread(image_file)
        video.write(image)

    # Release the video object
    video.release()

    print(f"Video created successfully: {output_video}")