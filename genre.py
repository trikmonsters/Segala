import os
import math
import shutil
from PIL import Image, ImageDraw

WIDTH = 1080
HEIGHT = 1920

FPS = 60
DURATION = 10

TOTAL_FRAMES = FPS * DURATION

FRAME_DIR = "frames"

if os.path.exists(FRAME_DIR):
    shutil.rmtree(FRAME_DIR)

os.makedirs(FRAME_DIR, exist_ok=True)


def ease_out_cubic(t):
    return 1 - (1 - t) ** 3


def lerp(a, b, t):
    return a + (b - a) * t


class Stickman:

    def __init__(self, color, x, y):
        self.color = color
        self.x = x
        self.y = y

    def draw(self, draw, arm_angle=0, leg_angle=0):

        x = self.x
        y = self.y

        draw.ellipse(
            (x - 40, y - 40, x + 40, y + 40),
            outline=self.color,
            width=8
        )

        neck_y = y + 40
        hip_y = neck_y + 130

        draw.line(
            (x, neck_y, x, hip_y),
            fill=self.color,
            width=8
        )

        shoulder_y = neck_y + 20

        arm_len = 90

        lx = x + math.cos(math.radians(180 - arm_angle)) * arm_len
        ly = shoulder_y + math.sin(math.radians(180 - arm_angle)) * arm_len

        rx = x + math.cos(math.radians(arm_angle)) * arm_len
        ry = shoulder_y + math.sin(math.radians(arm_angle)) * arm_len

        draw.line((x, shoulder_y, lx, ly), fill=self.color, width=8)
        draw.line((x, shoulder_y, rx, ry), fill=self.color, width=8)

        leg_len = 120

        llx = x - 30 + math.cos(math.radians(90 + leg_angle)) * leg_len
        lly = hip_y + math.sin(math.radians(90 + leg_angle)) * leg_len

        rlx = x + 30 + math.cos(math.radians(90 - leg_angle)) * leg_len
        rly = hip_y + math.sin(math.radians(90 - leg_angle)) * leg_len

        draw.line((x - 30, hip_y, llx, lly), fill=self.color, width=8)
        draw.line((x + 30, hip_y, rlx, rly), fill=self.color, width=8)


black = Stickman("black", 250, 900)
red = Stickman("red", 850, 900)

for frame in range(TOTAL_FRAMES):

    t = frame / FPS

    img = Image.new(
        "RGB",
        (WIDTH, HEIGHT),
        (255, 255, 255)
    )

    draw = ImageDraw.Draw(img)

    arm_black = 20
    arm_red = 20

    leg_black = 0
    leg_red = 0

    if t < 2:

        p = ease_out_cubic(t / 2)

        black.x = lerp(250, 450, p)
        red.x = lerp(850, 650, p)

        leg_black = math.sin(t * 10) * 25
        leg_red = math.sin(t * 10) * 25

    elif t < 4:

        combo = math.sin((t - 2) * 20)

        arm_black = 120 + combo * 40

        if combo > 0.8:
            red.x += 4

    elif t < 6:

        kick = math.sin((t - 4) * 12)

        leg_red = kick * 60

        if kick > 0.9:
            black.x -= 4

    elif t < 8:

        p = (t - 6) / 2

        black.y = 900 - math.sin(p * math.pi) * 250

        arm_black = 150

    else:

        arm_black = 150
        leg_black = 50

        red.x += 2

    black.draw(draw, arm_black, leg_black)
    red.draw(draw, arm_red, leg_red)

    img.save(
        f"{FRAME_DIR}/frame_{frame:05d}.png",
        optimize=True
    )

print("Frames generated successfully")
