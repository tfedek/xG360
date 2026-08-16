import numpy as np
import pandas as pd
from football_xg.config import (
    GOAL_X, GOAL_Y, GOAL_WIDTH, LEFT_POST_Y, RIGHT_POST_Y,
    BOX_X_MIN, BOX_Y_MIN, BOX_Y_MAX
)

def shot_distance(x, y):
    return float(np.sqrt((GOAL_X - x) ** 2 + (GOAL_Y - y) ** 2))

def shot_angle(x, y):
    a = np.sqrt((GOAL_X - x) ** 2 + (LEFT_POST_Y - y) ** 2)
    b = np.sqrt((GOAL_X - x) ** 2 + (RIGHT_POST_Y - y) ** 2)
    c = GOAL_WIDTH

    denom = 2 * a * b
    if denom <= 0:
        return np.nan

    cos_val = np.clip((a * a + b * b - c * c) / denom, -1, 1)
    return float(np.arccos(cos_val))

def euclidean(x1, y1, x2, y2):
    return float(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))

def point_in_triangle(px, py, ax, ay, bx, by, cx, cy):
    def sign(x1, y1, x2, y2, x3, y3):
        return (x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3)

    b1 = sign(px, py, ax, ay, bx, by) < 0
    b2 = sign(px, py, bx, by, cx, cy) < 0
    b3 = sign(px, py, cx, cy, ax, ay) < 0

    return (b1 == b2) and (b2 == b3)

def estimated_blocked_angle(px, py, sx, sy, body_width=1.0):
    d = euclidean(px, py, sx, sy)
    if d <= 0:
        return 0.0
    return float(2 * np.arctan((body_width / 2) / d))

def distance_point_to_segment(px, py, ax, ay, bx, by):
    """
    Distance from point P to segment AB.
    Used for estimating how close a defender is to the direct shot line.
    """
    apx = px - ax
    apy = py - ay
    abx = bx - ax
    aby = by - ay

    ab_len_sq = abx * abx + aby * aby
    if ab_len_sq == 0:
        return euclidean(px, py, ax, ay)

    t = (apx * abx + apy * aby) / ab_len_sq
    t = max(0.0, min(1.0, t))

    closest_x = ax + t * abx
    closest_y = ay + t * aby

    return euclidean(px, py, closest_x, closest_y)

def in_box(x, y):
    return x >= BOX_X_MIN and BOX_Y_MIN <= y <= BOX_Y_MAX
