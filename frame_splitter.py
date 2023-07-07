import cv2
import os
from tools import remove_files

def split_video_into_frames(video_path, output_directory):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the frame interval for 1 second
    frame_interval = int(fps) * 5

    # Read and save frames
    success, frame = video.read()
    frame_number = 1
    while success:
        # Save the frame as an image
        frame_filename = os.path.join(output_directory, f"frame_{frame_number:05d}.jpg")
        cv2.imwrite(frame_filename, frame)

        # Move to the next frame
        frame_number += frame_interval
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = video.read()

    # Release the video object
    video.release()

    print(f"Successfully split the video into {frame_number // frame_interval} frames.")

# Example usage
video_path = "flightradar24.mp4"
output_directory = "split_frames"
remove_files(output_directory)
split_video_into_frames(video_path, output_directory)
