# internals

# externals
import cv2
from tibia_ocr.hash_ocr import convert_line, convert_paragraph
from tibia_ocr.utils import crop

# self
from detector import find_rectangle, imshow, tibia_window_detect


def health_bar(image, color):
    health_bars = []
    threshed = cv2.inRange(image, color, color)
    contours, _ = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        try:
            rectangle = cv2.boundingRect(contour)
            x, y, w, h = rectangle
            is_health_bar = tuple(image[y - 1, x]) == (0, 0, 0) and tuple(image[y + 2, x]) == (0, 0, 0) and h <= 2
            if is_health_bar:
                health_bars.append(_get_enemy_object(image, x, y))
        except IndexError:
            pass


def main():
    img = cv2.imread("img/ingame2.png")
    running = True
    _img = img.copy()
    tolerence = 71
    offset = 10
    while running:
        threshed = tibia_window_detect(img.copy(), tolerence, offset)
        recs = find_rectangle(threshed)
        for rec in recs:
            # cv2.rectangle(img, rec, (0, 255, 10), 2)
            threshed = cv2.inRange(img, (43, 43, 43), (68, 68, 68), None)
            _croped = crop(threshed, rec)
            # out = convert_paragraph(_croped)
            # if len(out) > 5:
            #     print(out)
            # cv2.imshow(f"{rec}", _croped)
        cv2.imshow("a", threshed)
        img = _img.copy()
        key = cv2.waitKey(1) & 0xFF
        match key:
            case x if x == ord("q"):
                running = False
            case x if x == ord("a"):
                tolerence -= 1
            case x if x == ord("s"):
                tolerence += 1
            case x if x == ord("d"):
                offset -= 1
            case x if x == ord("w"):
                offset += 1


if __name__ == "__main__":
    main()
