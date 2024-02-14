import cv2
import numpy as np

# typing
from cv2.typing import MatLike
from typing import List


def tibia_window_detect(toprocess, tolerance=10, offset=10):
    b = toprocess[:, :, 0]
    g = toprocess[:, :, 1]
    r = toprocess[:, :, 2]

    """
        choose value of only those pixel r,g,b 
            such that:-
                if 
                    tolerance-offset <= r <= tolerance+offset &
                    tolerance-offset <= g <= tolerance+offset &
                    tolerance-offset <= b <= tolerance+offset &:
                    choose_this_pixels 
        here I am basically generating mask of it
    """
    tibia_window = (
        (r <= tolerance + offset)
        & (tolerance - offset <= r)
        & (g <= tolerance + offset)
        & (tolerance - offset <= g)
        & (b <= tolerance + offset)
        & (tolerance - offset <= b)
    )
    threshold = 10
    """
    mask out the unnecessary pixels with black color
    """
    toprocess[~tibia_window] = [0, 0, 0]
    """
    if each pixel's individual color value are smaller than threadhold then
    make them white other wise keep them whites
    """
    toprocess[:, :, 0] = np.where(toprocess[:, :, 0] > threshold, [255], [0])
    toprocess[:, :, 1] = np.where(toprocess[:, :, 1] > threshold, [255], [0])
    toprocess[:, :, 2] = np.where(toprocess[:, :, 2] > threshold, [255], [0])
    toprocess = cv2.cvtColor(toprocess, cv2.COLOR_BGR2GRAY)
    imshow("wad", toprocess)
    return toprocess


def tibia_tol_ofst_adjust(winname, img):
    cv2.namedWindow(winname)
    ts = 71  # jus perfect
    ofset = 21  # just perfect for the image I have
    cv2.createTrackbar("ts", winname, ts, 255, lambda *_: None)
    cv2.createTrackbar("ofset", winname, ofset, 255, lambda *_: None)
    while True:
        processed = tibia_window_detect(img, ts, ofset)
        cv2.imshow(winname, processed)
        ts = cv2.getTrackbarPos("ts", winname)
        ofset = cv2.getTrackbarPos("ofset", winname)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break


def imshow(winname, img):
    cv2.namedWindow(winname)
    while True:
        cv2.imshow(winname, img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break


def angle_between_points(x1, y1, x2, y2) -> int:
    angle_rad = np.arctan2(y2 - y1, x2 - x1)
    angle_deg = np.degrees(angle_rad)
    angle_deg = (angle_deg + 360) % 360
    return angle_deg


def find_rectangles_like(lines: List[List[List[int]]] | MatLike) -> List[List[List[int]]] | MatLike:
    """
    Remove Any non straight lines
    """
    rec_lines = []
    valid = [0, 90, 180, 270, 360]
    for line in lines:
        for points in line:
            if angle_between_points(*points) in valid:
                rec_lines.append(line)
    return rec_lines


def find_lines(grayed_img: MatLike, colored_base: MatLike):
    low_threshold = 10
    high_threshold = 150
    rho = 2
    theta = np.pi / 180
    threshold = 50
    line_magnitude_px = 200
    synaptics_gap = 100
    edges = cv2.Canny(grayed_img, low_threshold, high_threshold)
    line_image = np.copy(colored_base)
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), line_magnitude_px, synaptics_gap)
    lines = find_rectangles_like(lines)
    for line in lines:
        for startx, starty, endx, endy in line:
            cv2.line(line_image, (startx, starty), (endx, endy), (0, 0, 255), 3)
    imshow("those", line_image)


def main():
    img = cv2.imread("img/test1_small.png")
    imshow("wa", img)
    # tibia_tol_ofst_adjust("wad", img)
    find_lines(tibia_window_detect(img.copy(), 71, 21), img)


if __name__ == "__main__":
    main()
