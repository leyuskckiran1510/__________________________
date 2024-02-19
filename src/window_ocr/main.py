"""
Vision Moduel 
"""
import cv2
from pydantic import BaseModel
from tibia_ocr.hash_ocr import convert_line

from datetime import datetime
from typing import Literal


class Conditions:
    ...


class VisionModle(BaseModel):
    level: int
    experience: int
    xp_gain_rate: int
    hit_points: int
    mana: int
    soul_points: int
    capacity: int
    speed: int
    food: datetime  # .minutes
    stamina: datetime  # .minutes
    offline_training: datetime  # .minutes
    magic: int
    fist: int
    club: int
    sword: int
    axe: int
    distance: int
    shielding: int
    fishing: int
    critical_hit: int
    health_cur: int
    health_max: int
    mana_cur: int
    mana_max: int
    ground_level: int
    conditions_list: list[Conditions]
    combat_mode: Literal["offensive"] | Literal["balanced"] | Literal["defensive"]
    combat_movement: Literal["stand_while_fighting"] | Literal["chase_opponent"]
    expert_mode: Literal["dove_mode"] | "white_hand_mode" | "yellow_hand_mode" | "   "


def main():
    images = ["img/ingame2.png", "img/ingame3.png"]

    image = cv2.imread("data/test.png")
    threshed = cv2.inRange(image, (192, 192, 192), (192, 192, 192))
    out = convert_line(threshed)
    print(out)


if __name__ == "__main__":
    main()
