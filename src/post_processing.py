import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1
import json
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from scipy.spatial.transform import Rotation as R
import quaternion
from utils_postprocessing import *
from os.path import exists
import math
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--json_filename", help="json file name to parse", type=str, default="error.json")
parser.add_argument("--scene_name", help="scene name", type=str, default="error")
parser.add_argument("--duplicacy_threshold", help="duplicacy threshold", type=float, default=0.1)
parser.add_argument("--path_to_scenes_dict", help="path to scenes_dict", type=str, default="")
args = parser.parse_args()

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def fix_outliers(pois):
    outlier_idx = 4
    min_angle = np.pi
    min_angles = []
    for vertex_idx_to_ignore in range(4):
        idxs = np.arange(4)
        idxs = np.delete(idxs, vertex_idx_to_ignore)

        # three lines
        angles = []
        for j in range(3):
            anchor = idxs[j]
            # print('anchor:', anchor)
            head_candidates = []
            for k in range(3):
                if idxs[k] != anchor:
                    head_candidates.append(idxs[k])
            head1 = head_candidates[0]
            head2 = head_candidates[1]

            line1 = pois[anchor] - pois[head1]
            line2 = pois[anchor] - pois[head2]
            angles.append(abs(angle_between(line1, line2) - np.pi/2))

        min_angles.append(min(angles))
        if min(angles) < min_angle:
            min_angle = min(angles)
            outlier_idx = vertex_idx_to_ignore

    if np.max(min_angles) < 0.1:
        return pois

    pois_new = np.delete(pois, outlier_idx, axis=0)
    med_across_col = np.median(pois_new, axis=0)
    pois_new_outliers = np.argmax(abs(pois_new - med_across_col), axis=0)

    outlier_replacement = []
    for m in range(3):
        outlier_replacement.append(pois_new[pois_new_outliers[m], m])

    pois[outlier_idx] = outlier_replacement

    return pois


def post_processing(json_filename, split):
    # load json file
    f = open(json_filename)
    data = json.load(f)

    # scene name
    scene_name = args.scene_name
    scenes_dict = np.load(f'{args.path_to_scenes_dict}scenes_dict.npy', allow_pickle=True).item()
    base_position = scenes_dict[scene_name]['base_position']

    np.save(f'dataset/{split}/{scene_name}.npy', [])
    if not exists(f'dataset/{split}/{scene_name}.npy'):
        np.save(f'dataset/{split}/{scene_name}.npy', [])

    count = 0
    for img_data in data:
        
        # check scene name
        scene_name_filename = img_data['metadata']['filename'].split('.png')[0].split('_')[0]
        if scene_name_filename != scene_name:
            continue

        # position, rotation
        midfix = img_data['metadata']['filename'].split('.png')[0].split('_')[1]
        suffix = int(img_data['metadata']['filename'].split('.png')[0].split('_')[2])
        position, rotation = get_pos_and_rot(midfix, base_position, suffix)

        # extract data
        for label in img_data['response']['annotations']:
            if label['label'] == 'Handle':
                continue

            # extract corners
            points = []
            for point in label['vertices']:
                points.append([int(point['x']), int(point['y'])])
            points = np.array(points)

            # continue if any are close to border
            padding = 5
            if points.min() <= padding or points.max() >= (1920-padding):
                continue

            # get image w/ mask,
            rgb, depth, x, y, z = get_point_cloud(scene_name=scene_name, position=position, rotation=rotation)
            mask, masked_img = get_masked_img(points, rgb)
            plt.imshow(masked_img)
            plt.show()
            
            # get image w/ transparent mask,
            transparent_masked_img = get_transparent_masked_img(rgb, mask, weight=0.75)

            # get 3d locations for coordinates, normal and centroid of plane
            w, normal, centroid = fit_plane(mask, x, y, z)

            # make y-coordinate of normal 0, then normalize
            normal[1] = 0
            normal = normal / np.sqrt(np.sum(normal**2))

            # print(w, normal, centroid)
            projected_corners = project_corners_to_plane(position, normal, centroid, x, y, z, points)
            
            # fix outliers
            projected_corners = fix_outliers(projected_corners)

            # adjust surface normal
            normal = adjust_surface_normal_direction(normal, position, centroid)

            # classify door vs drawer
            low_to_high_vert_coord = np.argsort(projected_corners[:, 1])
            height_0 = projected_corners[low_to_high_vert_coord[2]][1] - projected_corners[low_to_high_vert_coord[0]][1]
            height_1 = projected_corners[low_to_high_vert_coord[3]][1] - projected_corners[low_to_high_vert_coord[1]][1]
            height = (height_0 + height_1) / 2
            width_0 = np.linalg.norm(projected_corners[low_to_high_vert_coord[0]][::2] - projected_corners[low_to_high_vert_coord[1]][::2])
            width_1 = np.linalg.norm(projected_corners[low_to_high_vert_coord[2]][::2] - projected_corners[low_to_high_vert_coord[3]][::2])
            width = (width_0 + width_1) / 2
            if height != 0:
                aspect_ratio = width / height
            else:
                aspect_ratio = 1.0
            if aspect_ratio < 0.9:
                classification = 'door'
            elif aspect_ratio > 1.1:
                classification = 'drawer'
            else:
                classification = 'ambiguous'

            # write to file (save: 3d locations for coordinates, normal and centroid of plane)
            data_dict = list(np.load(f'dataset/{split}/{scene_name}.npy', allow_pickle=True))
            new_dict = {'vertices': projected_corners,
                        'vertices_2d': points,
                        'normal': normal,
                        'centroid': centroid,
                        'masked_img': masked_img,
                        'camera_pos': position,
                        'camera_rot': rotation,
                        'width': width,
                        'height': height,
                        'aspect_ratio': aspect_ratio,
                        'classification': classification,
                        'classification_scale': -1,
                        'img_name': img_data['metadata']['filename'],
                        'mask': mask, 
                        'transparent_masked_img': transparent_masked_img}

            # check for handles
            new_dict['handle_3d'], new_dict['handle_2d'] = check_for_handle(img_data['response']['annotations'], mask, x, y, z)

            # re-do for case where only one handle is within mask
            # therefore, if there is a compartment face with either 3 or 1 vertices, things will be correctly classified
            if new_dict['handle_3d'] is None:
                new_dict['handle_3d'], new_dict['handle_2d'] = check_for_handle(img_data['response']['annotations'], mask, x, y, z, only_one=True)

            data_dict.append(new_dict)
	    np.save(f'dataset/{split}/{scene_name}.npy', data_dict)
            count += 1


train_val_test = np.load('./../misc_data/train_val_test_split.npy', allow_pickle=True).item()
if args.scene_name in train_val_test['train']:
    split = 'train'
elif args.scene_name in train_val_test['val']:
    split = 'val'
else:
    assert args.scene_name in train_val_test['test']
    split = 'test'
post_processing(args.json_filename, split)

# de-duplicate
data_dict = list(np.load(f'dataset/{split}/{args.scene_name}.npy', allow_pickle=True))

centroids = []
for face in data_dict:
    centroids.append(face['centroid'])
centroids = np.array(centroids)

dists = np.zeros((centroids.shape[0], centroids.shape[0]))
for a_idx, a_ in enumerate(centroids):
    for b_idx, b_ in enumerate(centroids):
        if a_idx == b_idx:
            dists[a_idx, b_idx] = 1e5
        else:
            dists[a_idx, b_idx] = np.linalg.norm(a_-b_)

argsort_idxs = np.argsort(dists.flatten())


all_idxs = list(np.arange(dists.shape[0]))
duplicates = []
for argsort_idx in argsort_idxs:
    a_, b_ = np.unravel_index(argsort_idx, dists.shape)
    if dists[a_, b_] < 0.1:
        # see if either a_ or b_ exists in duplicates
        added = False
        for set_idx, set_ in enumerate(duplicates):
            if a_ in set_ and b_ not in set_:
                duplicates[set_idx].append(b_)
                added = True
                break
            elif a_ not in set_ and b_ in set_:
                duplicates[set_idx].append(a_)
                added = True
                break
            elif a_ in set_ and b_ in set_:
                added = True
                break
        if added == False:
            duplicates.append([a_, b_])


for duplicate in duplicates:
    # look at distance between highest 2 y-coords, distance between lowest 2 y_coords
    # pick one with smallest max of these two
    max_vertex_dist = []
    for elem in duplicate:
        vertices = data_dict[elem]['vertices']
        vertices_y = np.sort(vertices[:, 1])
        lower_dist = abs(vertices_y[0] - vertices_y[1])
        upper_dist = abs(vertices_y[2] - vertices_y[3])
        max_vertex_dist.append(max(lower_dist, upper_dist))
    max_vertex_dist = np.argsort(max_vertex_dist)
    to_remove = [duplicate[idx] for idx in max_vertex_dist[1:]]

    for to_remove_elem in to_remove:
        if to_remove_elem in all_idxs:
            all_idxs.remove(to_remove_elem)


data_dict_unique = np.array(data_dict)[all_idxs]
np.save(f'dataset/{split}/{args.scene_name}_unique.npy', data_dict_unique)
print(f"de-duplicated ratio: {len(all_idxs)}/{len(data_dict)} = {len(all_idxs)/len(data_dict)}")

