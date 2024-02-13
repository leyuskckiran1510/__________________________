import cv2

# can be any container format like AVI or MOV,
CAPTURED_NAME = "captured_video.avi"
# for more information https://fourcc.org/pixel-format/yuv-yuy2/
FOURCC = cv2.VideoWriter_fourcc(*"YUY2")


def capture(device_index: int, output_folder: str, intervalms: int = 100):
    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        print("Error: Could not open video device.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(CAPTURED_NAME, FOURCC, 20, (frame_width, frame_height))
    key: int = 0x00
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        screenshot_path = f"{output_folder}/screenshot_{i:03d}.png"
        cv2.imwrite(screenshot_path, frame)
        out.write(frame)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(intervalms) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    out.release()


capture(0, "frames", intervalms=100)
