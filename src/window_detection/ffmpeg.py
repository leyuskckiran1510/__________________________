import subprocess


# ffmpeg -f v4l2 -input_format yuyv422 -video_size 1080x720 -i /dev/video0 -c:v rawvideo -pix_fmt yuyv422 output.avi
def capture_video(device_index: int, output_file: str, video_size: str = "640x480"):
    command = [
        "ffmpeg",
        "-f",
        "v4l2",
        "-input_format",
        "yuyv422",
        "-video_size",
        video_size,
        "-i",
        f"/dev/video{device_index}",
        "-c:v",
        "rawvideo",
        "-pix_fmt",
        "yuyv422",
        output_file,
    ]

    try:
        subprocess.run(command, check=True)
        print("Video captured successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")


# Example usage:
capture_video(0, "output.avi", "1080x720")
