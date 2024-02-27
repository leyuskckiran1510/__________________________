# internal
import cProfile
import timeit
from os import environ, listdir, path
from typing import List

# external
import cv2
import numpy as np
from line_profiler import profile

# typing
from cv2.typing import MatLike

"""
Command Ran:-
    PROFILE=1 make detect ana
Output:-
    Running detector.py
    .venv/bin/python ./src/window_detection/detector.py -O100
    Analyzing..
    .venv/bin/python analyze_profiled.py
    Thu Feb 15 19:52:59 2024    profile.txt

             992 function calls in 0.011 seconds

0.065sec to 0.011sec
"""


@profile
def tibia_window_detect(toprocess: MatLike, tolerance=10, offset=10) -> MatLike:
    l = tolerance - offset
    r = tolerance + offset
    return cv2.inRange(toprocess, (l, l, l), (r, r, r), None)


def tibia_tol_ofst_adjust(winname: str, img: MatLike) -> None:
    cv2.namedWindow(winname)
    ts = 71  # jus perfect
    ofset = 10  # just perfect for the image I have
    cv2.createTrackbar("ts", winname, ts, 255, lambda *_: None)
    cv2.createTrackbar("ofset", winname, ofset, 255, lambda *_: None)
    while True:
        processed = tibia_window_detect(img, ts, ofset)
        processed = cv2.erode(processed, np.ones((2, 100)))

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


class Rect:
    ...


@profile
def find_rectangle(grayed_img: MatLike) -> List[Rect]:
    contours, _ = cv2.findContours(grayed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    recs = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:
            recs.append(cv2.boundingRect(contour))
    return recs


def find_rectangle_draw(grayed_img: MatLike, img: MatLike):
    contours, _ = cv2.findContours(grayed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:
            cv2.rectangle(img, cv2.boundingRect(contour), (255, 200, 10), 2)
    imshow("a", grayed_img)


def main():
    if PROFILE:
        cProfile.run(
            f"[find_rectangle(tibia_window_detect(img.copy(), 71, 10)) for i in range({PROFILE})]",
            filename="profile.txt",
        )
    else:
        find_rectangle_draw(tibia_window_detect(img.copy(), 71, 10), img)
        # tibia_tol_ofst_adjust("a", img)
    # for image_name in listdir("img"):
    #     img = cv2.imread(path.join("img", image_name))
    # find_rectangle(tibia_window_detect(img.copy(), 71, 10), img)


if __name__ == "__main__":
    img = cv2.imread("img/login.png")
    PROFILE = 0
    if profile := environ.get("PROFILE", environ.get("profile", None)):
        PROFILE = int(profile)
        main()
    else:
        main()
