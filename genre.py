import math
import random
import numpy as np
from PIL import Image, ImageDraw
from moviepy.editor import ImageSequenceClip

WIDTH = 1080
HEIGHT = 1920

FPS = 60
DURATION = 12

TOTAL_FRAMES = FPS * DURATION

frames = []

# ------------------------
# EASING
# ------------------------

def ease_out_cubic(t):
    return 1 - (1 - t) ** 3

def lerp(a, b, t):
    return a + (b - a) * t

# ------------------------
# STICKMAN
# ------------------------

class Stickman:

    def __init__(self, color, x, y):
        self.color = color
        self.x = x
        self.y = y

        self.head = 40
        self.body = 130

    def draw(self, draw, arm_angle=0, leg_angle=0):

        x = self.x
        y = self.y

        # head
        draw.ellipse(
            (x-40,y-40,x+40,y+40),
            outline=self.color,
            width=8
        )

        neck_y = y + 40
        hip_y = neck_y + self.body

        # body
        draw.line(
            (x,neck_y,x,hip_y),
            fill=self.color,
            width=8
        )

        shoulder_y = neck_y + 20

        arm_len = 90

        left_arm_x = x + math.cos(math.radians(180-arm_angle))*arm_len
        left_arm_y = shoulder_y + math.sin(math.radians(180-arm_angle))*arm_len

        right_arm_x = x + math.cos(math.radians(arm_angle))*arm_len
        right_arm_y = shoulder_y + math.sin(math.radians(arm_angle))*arm_len

        draw.line(
            (x,shoulder_y,left_arm_x,left_arm_y),
            fill=self.color,
            width=8
        )

        draw.line(
            (x,shoulder_y,right_arm_x,right_arm_y),
            fill=self.color,
            width=8
        )

        leg_len = 120

        left_leg_x = x - 30 + math.cos(math.radians(90+leg_angle))*leg_len
        left_leg_y = hip_y + math.sin(math.radians(90+leg_angle))*leg_len

        right_leg_x = x + 30 + math.cos(math.radians(90-leg_angle))*leg_len
        right_leg_y = hip_y + math.sin(math.radians(90-leg_angle))*leg_len

        draw.line(
            (x-30,hip_y,left_leg_x,left_leg_y),
            fill=self.color,
            width=8
        )

        draw.line(
            (x+30,hip_y,right_leg_x,right_leg_y),
            fill=self.color,
            width=8
        )

# ------------------------
# FIGHTERS
# ------------------------

black = Stickman("black", 250, 900)
red = Stickman("red", 850, 900)

# ------------------------
# MAIN LOOP
# ------------------------

for frame in range(TOTAL_FRAMES):

    t = frame / FPS

    img = Image.new("RGB",(WIDTH,HEIGHT),(255,255,255))
    draw = ImageDraw.Draw(img)

    arm_black = 15
    arm_red = 15

    leg_black = 0
    leg_red = 0

    # ----------------
    # Scene 1
    # Walk
    # ----------------

    if t < 2:

        p = ease_out_cubic(t/2)

        black.x = lerp(250,450,p)
        red.x = lerp(850,650,p)

        leg_black = math.sin(t*10)*30
        leg_red = math.sin(t*10)*30

    # ----------------
    # Scene 2
    # Dash
    # ----------------

    elif t < 3:

        p = ease_out_cubic((t-2)/1)

        black.x = lerp(450,650,p)

        arm_black = 60

    # ----------------
    # Scene 3
    # Punch Combo
    # ----------------

    elif t < 5:

        combo = math.sin((t-3)*25)

        arm_black = 100 + combo*50

        if combo > 0.8:

            red.x += 4

    # ----------------
    # Scene 4
    # Counter Kick
    # ----------------

    elif t < 7:

        kick = math.sin((t-5)*12)

        leg_red = kick*60

        if kick > 0.9:

            black.x -= 6

    # ----------------
    # Scene 5
    # Jump Attack
    # ----------------

    elif t < 9:

        p = (t-7)/2

        black.y = 900 - math.sin(p*math.pi)*250

        arm_black = 140

    # ----------------
    # Scene 6
    # Finisher
    # ----------------

    else:

        pulse = math.sin((t-9)*15)

        arm_black = 140
        leg_black = 50

        if pulse > 0.8:

            red.x += 10

    # Camera shake

    shake_x = 0
    shake_y = 0

    if frame % 30 == 0:
        shake_x = random.randint(-5,5)
        shake_y = random.randint(-5,5)

    black.x += shake_x
    black.y += shake_y

    red.x += shake_x
    red.y += shake_y

    black.draw(draw, arm_black, leg_black)
    red.draw(draw, arm_red, leg_red)

    frames.append(np.array(img))

clip = ImageSequenceClip(frames, fps=FPS)

clip.write_videofile(
    "battle.mp4",
    codec="libx264",
    fps=FPS
)
