# builtin imports
from fractions import Fraction


# external imports
import cv2
from cv2.typing import MatLike
import numpy as np

# type hint imports
from typing import List, Tuple, Set

# self-code imports
from fresh import tibia_window_detect


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


def create_rect(points1: List[int], points2: List[int]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    x1, y1, x2, y2 = points1
    x3, y3, x4, y4 = points2
    x_min = min(x1, x2, x3, x4)
    x_max = max(x1, x2, x3, x4)
    y_min = min(y1, y2, y3, y4)
    y_max = max(y1, y2, y3, y4)
    return (x_min, y_min), (x_max - x_min, y_max - y_min)
    # return (x_min, y_min), (x_max, y_max)


def find_rectangles(lines: List[List[List[int]]] | MatLike) -> Set[Tuple[Tuple[int, int], Tuple[int, int]]] | MatLike:
    rectangles = set()
    for line1 in lines:
        for points1 in line1:
            for line2 in lines:
                for points2 in line2:
                    if sum(points1) == sum(points2):
                        continue
                    angle_diff = abs(angle_between_points(*points1) - angle_between_points(*points2))
                    if angle_diff in [90, 270]:
                        rect = create_rect(points1, points2)
                        rectangles.add(rect)

    return rectangles


def aspect_ratio(point1, point2) -> Fraction:
    x1, y1 = point1
    x2, y2 = point2
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    gcd = 1
    for i in range(2, min(width, height) + 1):
        if width % i == 0 and height % i == 0:
            gcd = i
    if width // gcd and height // gcd:
        ratio = Fraction(width // gcd, height // gcd)
        return ratio
    return Fraction(1, 1)


def is_close_to_16_9(ratio):
    target_ratio = Fraction(16, 9)
    threshold = 0.001
    diff = abs(ratio - target_ratio)
    return float(diff) < threshold


def detect_lines(gray: MatLike, img: MatLike) -> List:
    # img = cv2.addWeighted(img, 0.8, img, 0.8, 0)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray = img
    gray = tibia_window_detect(img, 71, 21)
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    low_threshold = 10
    high_threshold = 150
    rho = 2
    theta = np.pi / 180
    threshold = 50
    line_magnitude_px = 200
    synaptics_gap = 100
    cv2.namedWindow("image")
    # cv2.createTrackbar("threshold", "image", threshold, 100, lambda *_: None)
    # cv2.createTrackbar("low_threshold", "image", low_threshold, 255, lambda *_: None)
    # cv2.createTrackbar("high_threshold", "image", high_threshold, 255, lambda *_: None)
    # cv2.createTrackbar("line_magnitude_px", "image", line_magnitude_px, 300, lambda *_: None)
    # cv2.createTrackbar("synaptics_gap", "image", synaptics_gap, 100, lambda *_: None)
    # cv2.createTrackbar("rho", "image", rho, 50, lambda *_: None)
    # cv2.createTrackbar("theta", "image", int(theta), 360, lambda *_: None)

    while True:
        edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
        line_image = np.copy(img)  # creating a blank to draw lines on
        lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), line_magnitude_px, synaptics_gap)
        points = []
        lines = find_rectangles_like(lines)
        for line in lines:
            for startx, starty, endx, endy in line:
                cv2.line(line_image, (startx, starty), (endx, endy), (0, 0, 255), 3)
        rectangles = find_rectangles(lines)
        key = 0x00
        _l = line_image.copy()
        for p1, p2 in rectangles:
            isit = is_close_to_16_9(aspect_ratio(p1, p2))
            if not isit:
                continue
            line_image = _l.copy()
            cv2.rectangle(line_image, p1, p2, (255, 100, 50), 2)
            cv2.imshow("image", line_image)
            key = cv2.waitKey(1)
            if key & 0xFF == ord("q"):
                print(p1, p2)
                break
        if key & 0xFF == ord("q"):
            break
        # break
        # threshold = cv2.getTrackbarPos("threshold", "image")
        # line_magnitude_px = cv2.getTrackbarPos("line_magnitude_px", "image")
        # synaptics_gap = cv2.getTrackbarPos("synaptics_gap", "image")
        # low_threshold = cv2.getTrackbarPos("low_threshold", "image")
        # high_threshold = cv2.getTrackbarPos("high_threshold", "image")
        # rho = cv2.getTrackbarPos("rho", "image")
        # theta = cv2.getTrackbarPos("theta", "image")

    return points


def edges_with_ycrcb(image: MatLike):
    ycbcr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y_channel = ycbcr_image[:, :, 0]
    edges = cv2.Canny(y_channel, 100, 200)
    # cv2.imshow("Edges", edges)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return edges


def edges_with_gray(img: MatLike):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    low_threshold = 10
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    # cv2.imshow("Edges", edges)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return edges


def click_Callback(ev, x, y, _, __):
    if ev == cv2.EVENT_LBUTTONUP:
        print(x, y)


def imshow(winname, img):
    while True:
        cv2.imshow(winname, img)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break


def debug_img(img: MatLike):
    cv2.namedWindow("window")
    cv2.setMouseCallback("window", click_Callback)
    imshow("window", img)


def main():
    img = cv2.imread("img/test2.png")
    _img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detect_lines(_img, img)
    # debug_img(img)
    # e1 = edges_with_ycrcb(img)
    # e2 = edges_with_gray(img)
    # diff = abs(e1 + e2 + _img)

    # while True:
    #     e1 = edges_with_ycrcb(img)
    #     e2 = edges_with_gray(img)
    #     diff = _img - abs(e1 + e2)
    #     cv2.imshow("wa", diff)
    #     key = cv2.waitKey(0) & 0xFF
    #     if key == ord("q"):
    #         break

    # rectangle = rectangle[::-1]
    # while True:
    #     cv2.rectangle(img, rectangle[0], rectangle[1], color=(255, 10, 100), thickness=-1)
    #     cv2.imshow("name", img)
    #     key = cv2.waitKey(0)
    #     if key & 0xFF == ord("q"):
    #         break


if __name__ == "__main__":
    main()
