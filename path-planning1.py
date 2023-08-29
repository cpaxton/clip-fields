import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import tqdm
import einops

import os
import sys

from dataloaders.real_dataset import DeticDenseLabelledDataset
from dataloaders.sam_dataset import SAMLabelledDataset
from grid_hash_model import GridCLIPModel

from misc import MLP

import pandas as pd
import pyntcloud
from pyntcloud import PyntCloud
import clip

import rospy
from std_msgs.msg import Float64MultiArray, String

import heapq
import math
import time

import yaml
from pathlib import Path
metadata = yaml.safe_load(Path("map_data.yaml").read_text())
xmin, ymin, resolution = metadata['xmin'], metadata['ymin'], metadata['resolution']

def neighbors(pt):
    return [(pt[0] + dx, pt[1] + dy) for dx in range(-1, 2) for dy in range(-1, 2) if (dx, dy) != (0, 0)]

def compute_heuristic(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

def to_pt(xy):
        return (
            int((xy[0] - xmin) / resolution),
            int((xy[1] - ymin) / resolution),
        )

def to_xy(pt):
    return (
        pt[0] * resolution + xmin,
        pt[1] * resolution + ymin,
    )

def is_in_line_of_sight(start_pt, end_pt):

        dx = end_pt[0] - start_pt[0]
        dy = end_pt[1] - start_pt[1]

        if abs(dx) > abs(dy):
            if dx < 0:
                start_pt, end_pt = end_pt, start_pt
            for x in range(start_pt[0], end_pt[0] + 1):
                yf = start_pt[1] + (x - start_pt[0]) / dx * dy
                # if self.point_is_occupied(x, int(yf)):
                #     return False
                for y in list({math.floor(yf), math.ceil(yf)}):
                    if (x, y) not in valid_pts:
                        return False

        else:
            if dy < 0:
                start_pt, end_pt = end_pt, start_pt
            for y in range(start_pt[1], end_pt[1] + 1):
                xf = start_pt[0] + (y - start_pt[1]) / dy * dx
                # if self.point_is_occupied(int(x), y):
                #     return False
                for x in list({math.floor(xf), math.ceil(xf)}):
                    if (x, y) not in valid_pts:
                        return False

        return True


def clean_path(path):
        cleaned_path = [path[0]]
        i = 0
        while i < len(path) - 1:
            for j in range(len(path) - 1, i, -1):
                if is_in_line_of_sight(path[i], path[j]):
                    break
            else:
                j = i + 1
            cleaned_path.append(path[j])
            i = j
        return cleaned_path

def plan(start_xy, end_xy):
    start_pt, end_pt = to_pt(start_xy), to_pt(end_xy)
    q = [(0, start_pt)]
    came_from = {start_pt: None}
    cost_so_far = {start_pt: 0.0}
    while q:
        _, current = heapq.heappop(q)
        if current == end_pt:
            break
            
        for nxt in neighbors(current):
            if nxt not in valid_pts:
                continue
                
            new_cost = cost_so_far[current] + compute_heuristic(current, nxt)
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                priority = new_cost + compute_heuristic(end_pt, nxt)
                heapq.heappush(q, (priority, nxt))
                came_from[nxt] = current
    path = []
    current = end_pt
    while current != start_pt:
        path.append(current)
        prev = came_from[current]
        if prev is None:
            break
        current = prev
    path.append(start_pt)
    path.reverse()
    path = clean_path(path)

    return [start_xy] + [to_xy(pt) for pt in path[1:-1]] + [end_xy]

valid_xs, valid_ys = np.where(np.load('map.npy'))
valid_pts = [(valid_xs[i], valid_ys[i]) for i in range(len(valid_xs))]

os.environ["TOKENIZERS_PARALLELISM"] = "false"
path_pub = rospy.Publisher('path', Float64MultiArray, queue_size=1)
rospy.init_node('talker', anonymous=True)

#location = (0, 0)
start_xy, end_xy = (0, 0), (-2, 4)

while not rospy.is_shutdown():

    waypoints = plan(start_xy[:2], end_xy[:2])
    print(waypoints)
    paths = []
    for i in range(len(waypoints) - 1):
        paths.append(np.arctan((grid_waypoints[i + 1][1] - grid_waypoints[i][1]) / (grid_waypoints[i + 1][0] - grid_waypoints[i][0])))
        paths.append(np.linalg.norm(np.array([(grid_waypoints[i + 1][1] - grid_waypoints[i][1]), (grid_waypoints[i + 1][0] - grid_waypoints[i][0])])))

    start_xy = waypoints[-1]
    
    #for waypoint in waypoints:
    #    paths.append(waypoint[0])
    #    paths.append(waypoint[1])

    points = Float64MultiArray()
    points.data = paths
    path_pub.publish(points)
