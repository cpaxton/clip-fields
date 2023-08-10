import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain, cycle
from sentence_transformers import SentenceTransformer, util

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

# This is the cutoff used for NYU Lab
CUTOFF = 0.25

DEVICE = "cuda"
model, preprocess = clip.load("ViT-B/32", device=DEVICE)
sentence_model = SentenceTransformer("all-mpnet-base-v2")

training_data = torch.load("./cdslab_labelled_dataset.pt")
max_coords, _ = training_data._label_xyz.max(dim=0)
min_coords, _ = training_data._label_xyz.min(dim=0)

label_model = GridCLIPModel(
    image_rep_size=training_data[0]["clip_image_vector"].shape[-1],
    text_rep_size=training_data[0]["clip_vector"].shape[-1],
    mlp_depth=1,
    mlp_width=600,
    log2_hashmap_size=20,
    num_levels=18,
    level_dim=8,
    per_level_scale=2,
    max_coords=max_coords,
    min_coords=min_coords,
).to(DEVICE)

model_weights_path = "./detic_label_regularized/implicit_scene_label_model_latest.pt"
model_weights = torch.load(model_weights_path, map_location=DEVICE)
label_model.load_state_dict(model_weights["model"])
print(label_model)
print("Loaded model from", model_weights_path)

batch_size = 10_000
points_dataloader = DataLoader(
    training_data._label_xyz, batch_size=batch_size, num_workers=10,
)
print("Created data loader", points_dataloader)

def neighbors(pt):
    return [(pt[0] + dx, pt[1] + dy) for dx in range(-1, 2) for dy in range(-1, 2) if (dx, dy) != (0, 0)]

def compute_heuristic(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

def to_pt(xy):
        return (
            int((xy[0] - coordinates.x.min()) / 0.2),
            int((xy[1] - coordinates.z.min()) / 0.2),
        )

def to_xy(pt):
    return (
        pt[0] * 0.2 + coordinates.x.min(),
        pt[1] * 0.2 + coordinates.z.min(),
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


def calculate_clip_and_st_embeddings_for_queries(queries):
    all_clip_queries = clip.tokenize(queries)
    with torch.no_grad():
        all_clip_tokens = model.encode_text(all_clip_queries.to(DEVICE)).float()
        all_clip_tokens = F.normalize(all_clip_tokens, p=2, dim=-1)
        all_st_tokens = torch.from_numpy(sentence_model.encode(queries))
        all_st_tokens = F.normalize(all_st_tokens, p=2, dim=-1).to(DEVICE)
    return all_clip_tokens, all_st_tokens

# Make a top down map for visualization. This step works best if the data is axis aligned.
def get_and_show_valid_points(coordinates, cutoff=CUTOFF, library=True):
    blockers = torch.from_numpy(
        np.array([[coordinates.x[i] for i in coordinates.y.keys() if coordinates.y[i] > -0.85 and coordinates.y[i] < -0.35],
          [coordinates.z[i] for i in coordinates.y.keys() if coordinates.y[i] > -0.85 and coordinates.y[i] < -0.35]]).T
    )
    all_grid_points = torch.from_numpy(np.array(np.meshgrid(np.arange(coordinates.x.min(), coordinates.x.max(), 0.2), np.arange(coordinates.z.min(), coordinates.z.max(), 0.2))).reshape(2, -1).T)
    distance = torch.linalg.norm(blockers[None, :, :].cuda() - all_grid_points[:, None, :].cuda(), dim=2, ord=2)

    valid_points_index = distance.cpu().numpy().min(axis=1) > cutoff
    valid_points = all_grid_points[valid_points_index]
    return valid_points

def find_alignment_over_model(label_model, queries, dataloader, visual=False):
    clip_text_tokens, st_text_tokens = calculate_clip_and_st_embeddings_for_queries(queries)
    # We give different weights to visual and semantic alignment 
    # for different types of queries.
    if visual:
        vision_weight = 10.0
        text_weight = 1.0
    else:
        vision_weight = 1.0
        text_weight = 10.0
    point_opacity = []
    with torch.no_grad():
        for data in tqdm.tqdm(dataloader, total=len(dataloader)):
            # Find alignmnents with the vectors
            predicted_label_latents, predicted_image_latents = label_model(data.to(DEVICE))
            data_text_tokens = F.normalize(predicted_label_latents, p=2, dim=-1).to(DEVICE)
            data_visual_tokens = F.normalize(predicted_image_latents, p=2, dim=-1).to(DEVICE)
            text_alignment = data_text_tokens @ st_text_tokens.T
            visual_alignment = data_visual_tokens @ clip_text_tokens.T
            total_alignment = (text_weight * text_alignment) + (vision_weight * visual_alignment)
            total_alignment /= (text_weight + vision_weight)
            point_opacity.append(total_alignment)

    point_opacity = torch.cat(point_opacity).T
    print(point_opacity.shape)
    return point_opacity


merged_pcd = o3d.geometry.PointCloud()
merged_pcd.points = o3d.utility.Vector3dVector(training_data._label_xyz)
merged_pcd.colors = o3d.utility.Vector3dVector(training_data._label_rgb)
merged_downpcd = merged_pcd.voxel_down_sample(voxel_size=0.03)

print("Create pts result")
pts_result = np.concatenate((np.asarray(merged_downpcd.points), np.asarray(merged_downpcd.colors)), axis=-1)

df = pd.DataFrame(
    # same arguments that you are passing to visualize_pcl
    data=pts_result,
    columns=["x", "y", "z", "red", "green", "blue"]
)
cloud = PyntCloud(df)

print("Point cloud", cloud)

# Now figure out the points that are far enough.
coordinates = cloud.points
coordinates = coordinates[coordinates.y < 0]

valid_points = get_and_show_valid_points(coordinates)
print("Found some valid points:", valid_points.shape)

valid_pts = [to_pt(valid_point) for valid_point in valid_points]

os.environ["TOKENIZERS_PARALLELISM"] = "false"
path_pub = rospy.Publisher('path', Float64MultiArray, queue_size=1)
rospy.init_node('talker', anonymous=True)

location = (0, 0)

while not rospy.is_shutdown():
    query = input("Enter instruction:")
    print("query =", query)

    alignment_q = find_alignment_over_model(label_model, [query], points_dataloader)

    for q in alignment_q:
        alpha = q.detach().cpu().numpy()
        pts = training_data._label_xyz.detach().cpu()

        # We are thresholding the points to get the top 0.01% of points.
        # Subsample if the number of points is too large.
        threshold = torch.quantile(q[::10, ...], 0.9999).cpu().item()

        # Normalize alpha
        a_norm = (alpha - alpha.min()) / (alpha.max() - alpha.min())
        a_norm = torch.as_tensor(a_norm[..., np.newaxis])

        thres = alpha > threshold
        points = training_data._label_xyz[thres]
        max_point = pts[torch.argmax(a_norm)]
        print(f"LOOKAT {query} {max_point.tolist()}")

    target = torch.tensor([max_point[0], max_point[2]])
    target = valid_points[torch.argmin(torch.linalg.norm(target.cuda() - valid_points.cuda(), dim=1, ord=2))]

    waypoints = plan(location, (target[0].item(), target[1].item()))
    print(waypoints)

    location = waypoints[-1]
    
    paths = []
    for waypoint in waypoints:
        paths.append(waypoint[0])
        paths.append(waypoint[1])

    points = Float64MultiArray()
    points.data = paths
    path_pub.publish(points)
