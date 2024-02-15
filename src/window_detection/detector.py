# internal
import cProfile
from os import environ, listdir, path

# external
import cv2
from line_profiler import profile

# typing
from cv2.typing import MatLike

"""
Command Ran
    PROFILE=1 make detect ana

Output:
    Running detector.py
    .venv/bin/python ./src/detector.py
    Analyzing..
    .venv/bin/python analyze_profiled.py
    Thu Feb 15 16:32:43 2024    profile.txt

             996 function calls in 0.065 seconds 
        ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.068    0.068 <string>:1(<listcomp>)
        .....
        970    0.001    0.000    0.001    0.000 {contourArea}
        .....
        [trancuated]

"""


@profile
def tibia_window_detect(toprocess: MatLike, tolerance=10, offset=10) -> MatLike:
    """
    detect tibia game window theme color and
    detect widgets
    Returns:-
        GrayScaled Image with detected part as white and other black
    """
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
        (r <= tolerance + offset)  # type:ignore
        & (tolerance - offset <= r)  # type:ignore
        & (g <= tolerance + offset)  # type:ignore
        & (tolerance - offset <= g)  # type:ignore
        & (b <= tolerance + offset)  # type:ignore
        & (tolerance - offset <= b)  # type:ignore
    )
    threshold = 10
    """
    mask out the unnecessary pixels with black color
    """
    # TODO: It is taking up most of the time [65%] need to speed this line up
    toprocess[~tibia_window] = [0, 0, 0]
    """
    if each pixel's individual color value are smaller than threadhold then
    make them white other wise keep them whites
    """
    value = 255
    _, b = cv2.threshold(b, threshold, value, cv2.THRESH_BINARY)
    _, g = cv2.threshold(g, threshold, value, cv2.THRESH_BINARY)
    _, r = cv2.threshold(r, threshold, value, cv2.THRESH_BINARY)
    toprocess = cv2.merge((b, g, r))
    toprocess = cv2.cvtColor(toprocess, cv2.COLOR_BGR2GRAY)
    return toprocess


def tibia_tol_ofst_adjust(winname, img):
    cv2.namedWindow(winname)
    ts = 71  # jus perfect
    ofset = 10  # just perfect for the image I have
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


@profile
def find_rectangle(grayed_img: MatLike, colored_base: MatLike, i: int = 0):
    # imshow("a", grayed_img)
    _colored_base = colored_base.copy()
    contours, _ = cv2.findContours(grayed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(_colored_base, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # cv2.imwrite(f"img_dumps/{i}.png", _colored_base)
    # imshow("awd", _colored_base)


img = cv2.imread("img/test1_small.png")
PROFILE = 0


def main():
    if PROFILE:
        cProfile.run(
            f"[find_rectangle(tibia_window_detect(img.copy(), 71, 10), img,i) for i in range({PROFILE})]",
            filename="profile.txt",
        )
    else:
        find_rectangle(tibia_window_detect(img.copy(), 71, 10), img)

    # for image_name in listdir("img"):
    #     img = cv2.imread(path.join("img", image_name))
    #     find_rectangle(tibia_window_detect(img.copy(), 71, 10), img)


if __name__ == "__main__":
    if profile := environ.get("PROFILE", environ.get("profile", None)):
        PROFILE = int(profile)
        main()
    else:
        main()
