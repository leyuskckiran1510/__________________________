# builtin imports
from fractions import Fraction


# external imports
import cv2
from cv2.typing import MatLike
import numpy as np

# type hint imports
from typing import List, Tuple, Set

# self-code imports


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


def distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


# def find_rectangles(lines: List[List[List[int]]] | MatLike) -> Set[Tuple[Tuple[int, int], Tuple[int, int]]] | MatLike:
def find_rectangles(lines: np.ndarray | MatLike):
    rectangles = set()
    tomerge = []
    for line1 in lines:
        for points1 in line1:
            for line2 in lines:
                for points2 in line2:
                    if sum(points1) == sum(points2):
                        continue
                    angle_diff = abs(angle_between_points(*points1) - angle_between_points(*points2))
                    dist = distance(points1, points2)
                    if angle_diff in [90, 270] and dist > 200:
                        rect = create_rect(points1, points2)
                        rectangles.add(rect)
                    if (
                        angle_diff == 0
                        and (points1[0] == points2[0] or points1[1] or points2[1])
                        and not angle_between_points(points1[0], points1[1], points2[0], points2[1])
                    ):
                        # print(
                        #     points1, points2, angle_diff, angle_between_points(*points1), angle_between_points(*points2)
                        # )
                        # exit(0)
                        new_line = (points1[0], points1[1], points2[0], points2[1])
                        tomerge.append(new_line)

    return rectangles, tomerge


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


def detect_lines(img: MatLike) -> List:
    # img = cv2.addWeighted(img, 0.8, img, 0.8, 0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    low_threshold = 10
    high_threshold = 150
    rho = 2
    theta = np.pi / 180
    threshold = 40
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
        rectangles, new_lines = find_rectangles(lines)
        # print(new_lines)
        for startx, starty, endx, endy in new_lines:
            cv2.line(line_image, (startx, starty), (endx, endy), (255, 0, 255), 3)
            cv2.imshow("image", line_image)
            key = cv2.waitKey(0)
            if key & 0xFF == ord("q"):
                break
        key = 0x00
        _l = line_image.copy()
        for p1, p2 in rectangles:
            isit = is_close_to_16_9(aspect_ratio(p1, p2))
            if not isit:
                continue
            line_image = _l.copy()
            cv2.rectangle(line_image, p1, p2, (255, 100, 50), 2)
            cv2.imshow("image", line_image)
            key = cv2.waitKey(0)
            if key & 0xFF == ord("q"):
                print(p1, p2)
                break
        if key & 0xFF == ord("q"):
            break
        break
        # threshold = cv2.getTrackbarPos("threshold", "image")
        # line_magnitude_px = cv2.getTrackbarPos("line_magnitude_px", "image")
        # synaptics_gap = cv2.getTrackbarPos("synaptics_gap", "image")
        # low_threshold = cv2.getTrackbarPos("low_threshold", "image")
        # high_threshold = cv2.getTrackbarPos("high_threshold", "image")
        # rho = cv2.getTrackbarPos("rho", "image")
        # theta = cv2.getTrackbarPos("theta", "image")

    return points


def best_edges(image: MatLike):
    # Convert the image to YCbCr color space
    ycbcr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    # Extract the Y channel (luma component)
    y_channel = ycbcr_image[:, :, 2]

    # Apply Canny edge detection on the Y channel
    edges = cv2.Canny(y_channel, 100, 200)  # You may need to adjust the thresholds based on your image

    cv2.imshow("Edges", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return edges


def normal_edges(img: MatLike):
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    low_threshold = 10
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    cv2.imshow("Edges", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return edges


def main():
    img = cv2.imread("img/test1_small.png")
    points = detect_lines(img)
    # points = [[[0, 733], [1364, 733]], [[24, 36], [1236, 36]]]
    # cv2.line(img, *points[0], [255, 100, 10], 10)
    # cv2.line(img, *points[1], [255, 100, 10], 10)
    # cv2.imshow("Awd", img)
    # cv2.waitKey(0)
    # while True:
    #     e1 = best_edges(img)
    #     e2 = normal_edges(img)
    #     diff = abs(e1 + e2)
    #     cv2.imshow("wa", diff)
    #     key = cv2.waitKey(0) & 0xFF
    #     if key == ord("q"):
    #         break
    # rectangle = (617, 185), (1181, 493)

    # rectangle = rectangle[::-1]
    # while True:
    #     cv2.rectangle(img, rectangle[0], rectangle[1], color=(255, 10, 100), thickness=-1)
    #     cv2.imshow("name", img)
    #     key = cv2.waitKey(0)
    #     if key & 0xFF == ord("q"):
    #         break


if __name__ == "__main__":
    main()
