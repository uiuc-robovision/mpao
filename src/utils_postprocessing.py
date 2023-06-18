import os
import numpy as np
import quaternion
import matplotlib.pyplot as plt
import habitat
from matplotlib.path import Path
from scipy.spatial.transform import Rotation as R
import quaternion
from numpy.linalg import inv
import math
from os.path import exists


def get_point_cloud(scene_name='kfPV7w3FaU5', position=[0, 0, 0], rotation=[0, 0, 0, 1]):
    '''
    scene_name from train set
    '''
    # Set up the environment for testing
    config_file = "habitat-challenge/configs/challenge_objectnav2022_bigger_square_pointnav.local.rgbd.yaml"

    config = habitat.get_config(config_paths=config_file)
    config.defrost()
    train_path = f'habitat-challenge/habitat-challenge-data/objectgoal_hm3d/objectnav_hm3d_v1/train/content/{scene_name}.json.gz'
    if exists(train_path):
        config.DATASET.DATA_PATH = f'habitat-challenge/habitat-challenge-data/objectgoal_hm3d/objectnav_hm3d_v1/train/content/{scene_name}.json.gz'
    else:
        config.DATASET.DATA_PATH = f'habitat-challenge/habitat-challenge-data/objectgoal_hm3d/objectnav_hm3d_v1/val/content/{scene_name}.json.gz'
    config.DATASET.SCENES_DIR = 'habitat-challenge/data/scene_datasets'
    config.freeze()

    # Can also do directly in the config file
    config.defrost()
    config.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = False
    config.freeze()

    # Intrinsic parameters, assuming width matches height. Requires a simple refactor otherwise
    W = config.SIMULATOR.DEPTH_SENSOR.WIDTH
    H = config.SIMULATOR.DEPTH_SENSOR.HEIGHT

    assert(W == H)
    hfov = float(config.SIMULATOR.DEPTH_SENSOR.HFOV) * np.pi / 180.

    env = habitat.Env(config=config)

    depths, rgbs, cameras = [], [], []
    rotation = quaternion.from_float_array(np.array(rotation))
    position = np.array(position, dtype=np.float32)
    obs = env._sim.get_observations_at(position=position, rotation=rotation, keep_agent_at_new_pose=True)
    depths += [obs["depth"][...,0]]
    rgbs += [obs["rgb"]]
    cameras += [env._sim.get_agent_state()]
    env.close()
    
    # get pointcloud info
    K = np.array([
	[1 / np.tan(hfov / 2.), 0., 0., 0.],
	[0., 1 / np.tan(hfov / 2.), 0., 0.],
	[0., 0.,  1, 0],
	[0., 0., 0, 1]])

    # Now get an approximation for the true world coordinates -- see if they make sense
    # [-1, 1] for x and [1, -1] for y as array indexing is y-down while world is y-up
    xs, ys = np.meshgrid(np.linspace(-1,1,W), np.linspace(1,-1,H))
    depth = depths[0].reshape(1,H,W)
    xs = xs.reshape(1,H,W)
    ys = ys.reshape(1,H,W)

    # Unproject
    # negate depth as the camera looks along -Z
    xys = np.vstack((xs * depth , ys * depth, -depth, np.ones(depth.shape)))
    xys = xys.reshape(4, -1)
    xy_c0 = np.matmul(np.linalg.inv(K), xys)

    # Camera 1:
    quaternion_0 = cameras[0].sensor_states['depth'].rotation
    translation_0 = cameras[0].sensor_states['depth'].position
    rotation_0 = quaternion.as_rotation_matrix(quaternion_0)
    T_world_camera0 = np.eye(4)
    T_world_camera0[0:3,0:3] = rotation_0
    T_world_camera0[0:3,3] = translation_0

    xyz = np.matmul(T_world_camera0, xy_c0)
    xyz = xyz / xyz[3,:]
    xyz_reshaped = xyz.reshape((4, 1920, 1920))
    
    rgb = rgbs[0]
    depth = depths[0]
    x = xyz_reshaped[0]
    y = xyz_reshaped[1]
    z = xyz_reshaped[2]

    return rgb, depth, x, y, z



def convert_habitat_to_camera_surface_normal(rotation, position, point):
    quaternion_0 = rotation
    translation_0 = position
    rotation_0 = quaternion.as_rotation_matrix(quaternion_0)
    T_world_camera0 = np.eye(4)
    T_world_camera0[0:3,0:3] = rotation_0

    # make point into shape (4, 1)
    point = np.array([point[0], point[1], point[2], 1.])
    point = point.reshape((4, 1))

    xyz = np.matmul(np.linalg.inv(T_world_camera0), point)
    return xyz[:3].flatten()

def convert_camera_to_habitat_surface_normal(rotation, position, point):
    quaternion_0 = rotation
    translation_0 = position
    rotation_0 = quaternion.as_rotation_matrix(quaternion_0)
    T_world_camera0 = np.eye(4)
    T_world_camera0[0:3,0:3] = rotation_0

    # make point into shape (4, 1)
    point = np.array([point[0], point[1], point[2], 1.])
    point = point.reshape((4, 1))

    xyz = np.matmul(T_world_camera0, point)
    return xyz[:3].flatten()

def convert_habitat_to_camera(rotation, position, point):
    quaternion_0 = rotation
    translation_0 = position
    rotation_0 = quaternion.as_rotation_matrix(quaternion_0)
    T_world_camera0 = np.eye(4)
    T_world_camera0[0:3,0:3] = rotation_0
    T_world_camera0[0:3,3] = translation_0

    # make point into shape (4, 1)
    point = np.array([point[0], point[1], point[2], 1.])
    point = point.reshape((4, 1))

    xyz = np.matmul(np.linalg.inv(T_world_camera0), point)
    return xyz[:3].flatten()


def convert_camera_to_habitat(rotation, position, point):
    quaternion_0 = rotation
    translation_0 = position
    rotation_0 = quaternion.as_rotation_matrix(quaternion_0)
    T_world_camera0 = np.eye(4)
    T_world_camera0[0:3,0:3] = rotation_0
    T_world_camera0[0:3,3] = translation_0

    # make point into shape (4, 1)
    point = np.array([point[0], point[1], point[2], 1.])
    point = point.reshape((4, 1))

    xyz = np.matmul(T_world_camera0, point)
    return xyz[:3].flatten()


def convert_habitat_3d_to_pixel_coord(rotation, position, point, depth, x, y, z, hfov=1.5707963267948966, W=1920, threshold=0.03):
    pointcloud = np.stack([x, y, z], axis=-1)
point_mask = np.ones((1920, 1920, 3)) * point
    norm_ = np.linalg.norm(pointcloud - point_mask, axis=-1)
    if np.min(norm_) < threshold:
        return np.unravel_index(np.argmin(norm_), norm_.shape)
    else:
        return (W+1, W+1)


def get_masked_img(points, rgb):
    pts_ = points
    tupVerts=[(pts_[0, 0], pts_[0, 1]),
              (pts_[1, 0], pts_[1, 1]),
              (pts_[2, 0], pts_[2, 1]),
              (pts_[3, 0], pts_[3, 1])]
    
    x_, y_ = np.meshgrid(np.arange(1920), np.arange(1920)) # make a canvas with coordinates
    x_, y_ = x_.flatten(), y_.flatten()
    points = np.vstack((x_,y_)).T 

    p = Path(tupVerts) # make a polygon
    grid = p.contains_points(points)
    mask = grid.reshape(1920,1920) # now you have a mask with points inside a polygon
    
    modified_rgb = rgb.copy()
    for y_ in range(1920):
        for x_ in range(1920):
            if mask[y_, x_]:
                modified_rgb[y_, x_, 0] = 255
                modified_rgb[y_, x_, 1] = 0
                modified_rgb[y_, x_, 2] = 0
                
    return mask, modified_rgb



def get_transparent_masked_img(rgb, mask, weight=0.75):
    # mask color
    mask_ = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    quadrangle = rgb * mask_
    quadrangle_color = np.sum(quadrangle, axis=(0,1)) / (mask_.sum()/3)
    if quadrangle_color[0] > 1.5*(quadrangle_color[1]) and quadrangle_color[0] > 1.5*(quadrangle_color[2]):
        green_val = 255
    else:
        green_val = 0

    modified_rgb = rgb.copy()
    for y_ in range(1920):
        for x_ in range(1920):
            if mask[y_, x_]:
                modified_rgb[y_, x_, 0] = (weight)*rgb[y_, x_, 0] + (1-weight)*255 # 255
                modified_rgb[y_, x_, 1] = (weight)*rgb[y_, x_, 0] + (1-weight)*green_val # 0
                modified_rgb[y_, x_, 2] = (weight)*rgb[y_, x_, 0] + (1-weight)*0 # 0

                nonborder_pixel = mask[y_+1, x_] and mask[y_, x_+1] and mask[y_-1, x_] and mask[y_, x_-1]
                if not nonborder_pixel:
                    modified_rgb[y_, x_, 0] = 255
                    modified_rgb[y_, x_, 1] = green_val
                    modified_rgb[y_, x_, 2] = 0
                    
    return modified_rgb


def get_pos_and_rot(midfix, base_position, suffix):
    '''
    midfix: one of {top, cent, bot}
    suffix: [0, 9] image idx 
    '''
    if midfix == 'top':
        va = -30
        base_position_ = base_position + np.array([0, 1, 0])
    elif midfix == 'cent':
        va = 0
        base_position_ = base_position + np.array([0, 0, 0])
    elif midfix == 'bot':
        va = 30
        base_position_ = base_position + np.array([0, 0, 0])
    else:
        assert False

    angles = np.linspace(0, 360, 7)[:-1]
    ang = angles[suffix]
    rotation = R.from_euler('xyz', [180., ang, va], degrees=True).as_quat()
    rotation = np.array(rotation)
    
    return base_position_, rotation


def total_least_squares(X, y):
    # total least squares (i.e. perpendicular loss)
    pointcloud = np.concatenate((X, np.expand_dims(y, axis=1)), axis=1)
    centroid = pointcloud.mean(axis=0)
    points_centered = pointcloud - centroid
    u, _, _ = np.linalg.svd(points_centered.T)
    normal = u[:, 2]
    
    # w
    d = centroid.dot(normal)
    a = -1*normal[0] / normal[2]
    b = -1*normal[1] / normal[2]
    d = d / normal[2]
    w = np.array([d, a, b])
    
    return w, normal, centroid

def visualize_topdown_fit(drawer_pointcloud, pointcloud, w, inliers=None):
    plt.scatter(pointcloud[:, 0], pointcloud[:, 2])
    
    if inliers is not None:
        plt.scatter(inliers[:, 0], inliers[:, 2], color='yellow')
    
    min_x = min(drawer_pointcloud[:, 0])
    min_y = min(drawer_pointcloud[:, 1])
    min_z = min(drawer_pointcloud[:, 2])
    max_x = max(drawer_pointcloud[:, 0])
    max_y = max(drawer_pointcloud[:, 1])
    max_z = max(drawer_pointcloud[:, 2])
    
    xs = np.linspace(min_x, max_x, 50)
    ys = np.linspace(min_y, max_y, 50)
    zs = []
    for idx in range(50):
        z = w[0] + w[1]*xs[idx] + w[2]*ys[idx]
        zs.append(z)
    
    plt.plot(xs, zs, color='r')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.xlim(min_x-0.02, max_x+0.02)
    plt.ylim(min_z-0.1, max_z+0.1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gcf().set_size_inches(14, 10)
    plt.show()

def get_inliers(pointcloud, w, inlier_threshold=0.01):
    inliers = []
    for point_ in pointcloud:
        x_, y_, z_ = point_
        # distance to plane
        num = np.abs(-1*w[1]*x_ + -1*w[2]*y_ + z_ + -1*w[0])
        den = np.sqrt(w[1]**2 + w[2]**2 + 1)
        dist = num / den
        if dist < inlier_threshold:
            inliers.append(point_)
    inliers = np.array(inliers)
    return inliers


def fit_plane(mask, x, y, z):
    HEIGHT = 1920
    WIDTH = 1920
    # select points from pointcloud within masked region
    drawer_pointcloud = []
    for y_ in range(HEIGHT):
        for x_ in range(WIDTH):
            if mask[y_, x_]:
                point = np.array([x[y_, x_], y[y_, x_], z[y_, x_]])
                drawer_pointcloud.append(point)
    drawer_pointcloud = np.array(drawer_pointcloud)
    
    # least squares (predicting the Z-coordinate in world frame)
    X = drawer_pointcloud[:, :2]
    y = drawer_pointcloud[:, 2]
    if X.shape[0] > 1000:
        num_to_get = 1000
    else:
        num_to_get = X.shape[0]
    idxs = np.random.choice(X.shape[0], num_to_get, replace=False)
    X_ = X[idxs, :]
    y_ = y[idxs]
    
    w_total, normal_total, centroid_total = total_least_squares(X_, y_)
    
    # improve with iterations
    for iter_ in range(5):
        inliers = get_inliers(drawer_pointcloud[np.random.choice(drawer_pointcloud.shape[0], num_to_get, replace=False), :], w_total)
        if len(inliers.shape) == 1:
            break
        w_total, normal_total, _ = total_least_squares(inliers[:, :2], inliers[:, 2])

    visualize_topdown_fit(drawer_pointcloud, drawer_pointcloud, w_total)
    
    return w_total, normal_total, centroid_total


def LinePlaneCollision(planeNormal, planePoint, rayPoint0, rayPoint1, epsilon=1e-6):
    rayDirection = rayPoint1 - rayPoint0
    ndotu = planeNormal.dot(rayDirection)
    if abs(ndotu) < epsilon:
        raise RuntimeError("no intersection or line is within plane")
    w = rayPoint0 - planePoint
    si = -planeNormal.dot(w) / ndotu
    Psi = w + si * rayDirection + planePoint
    return Psi


def move_towards_centroid(points, step_len, x_pc, y_pc, z_pc):
    # find centroid in image space
    centroid_2d = np.mean(points, axis=0)

    preds = []
    for point in points:
        # direction towards centroid
        dir_to_centroid = centroid_2d - point
        step = np.sign(dir_to_centroid)

        # take 2 steps in this dir
        selected_y, selected_x = point[1], point[0]
        neighbor = np.array([x_pc[selected_y + int(step[1])*step_len, selected_x + int(step[0])*step_len],
                             y_pc[selected_y + int(step[1])*step_len, selected_x + int(step[0])*step_len],
                             z_pc[selected_y + int(step[1])*step_len, selected_x + int(step[0])*step_len]])

        pred = np.array([x_pc[selected_y, selected_x],
                         y_pc[selected_y, selected_x],
                         z_pc[selected_y, selected_x]])

        max_dist = np.max(pred - neighbor)
        if max_dist > 0.05:
            pred = neighbor
        preds.append(pred)

    return np.array(preds)

def project_corners_to_plane(camera_position, normal_total, centroid_total, x_pc, y_pc, z_pc, points):
    for step_len in [2, 5, 10]:
        preds = move_towards_centroid(points, step_len, x_pc, y_pc, z_pc)
        preds_y = preds[:, 1]
        preds_y_argsort = np.argsort(preds_y)
        if abs(preds_y_argsort[-1] - preds_y_argsort[-2]) == 1 or abs(preds_y_argsort[-1] - preds_y_argsort[-2]) == 3:
            break

    poi_1 = LinePlaneCollision(normal_total, centroid_total, camera_position, preds[0])
    poi_2 = LinePlaneCollision(normal_total, centroid_total, camera_position, preds[1])
    poi_3 = LinePlaneCollision(normal_total, centroid_total, camera_position, preds[2])
    poi_4 = LinePlaneCollision(normal_total, centroid_total, camera_position, preds[3])
    return np.array([poi_1, poi_2, poi_3, poi_4])



def adjust_surface_normal_direction(surface_normal, camera_pos, centroid):
    '''
    picks either surface_normal or -1 * surface_normal
    depending on camera_pos
    
    the higher the better
    - if angle is big, cosine is small (even goes to negative)
    - if angle is small, cosine is big (upper-bounded by 1)
    '''
    candidate_1 = surface_normal
    candidate_2 = -1. * surface_normal
    
    camera_minus_centroid = camera_pos - centroid
    camera_minus_centroid /= np.linalg.norm(camera_minus_centroid)
    
    if np.dot(candidate_1, camera_minus_centroid) > np.dot(candidate_2, camera_minus_centroid):
        return candidate_1
    else:
        return candidate_2
    
    
def check_for_handle(annotations_data, mask, x, y, z, only_one=False):
    '''
    only_one: only one of two vertices is required to be within masked region for successful correspondence
    '''
    for label_handle in annotations_data:
        if label_handle['label'] == 'Handle':
            # check if within boundary of current compartment face in image space
            # one side of handle may be outside
            # - first correspond handles for which both are inside
            # - then correspond handles for which one is inside to remaining one
            # - there should be very few cases where one vertex of handle is outside compartment face

            # extract vertices
            points_handle = []
            for point_handle in label_handle['vertices']:
                points_handle.append([int(point_handle['x']), int(point_handle['y'])])

            # see value of mask at vertices
            if points_handle[0][0] == 1920:
                p_0_0 = 1919
            else:
                p_0_0 = points_handle[0][0]
            if points_handle[0][1] == 1920:
                p_0_1 = 1919
            else:
                p_0_1 = points_handle[0][1]
            if points_handle[1][0] == 1920:
                p_1_0 = 1919
            else:
                p_1_0 = points_handle[1][0]
            if points_handle[1][1] == 1920:
                p_1_1 = 1919
            else:
                p_1_1 = points_handle[1][1]
            vertex_0 = mask[p_0_1, p_0_0]
            vertex_1 = mask[p_1_1, p_1_0]

            # if both vertices within masked region
            if not only_one:
                condition_satisfied = vertex_0 and vertex_1
            else:
                condition_satisfied = vertex_0 or vertex_1
                
            if condition_satisfied:                    
                # store handle

                # convert to 3d habitat coordinates
                vertex_0_x, vertex_0_y = points_handle[0][0], points_handle[0][1]
                vertex_1_x, vertex_1_y = points_handle[1][0], points_handle[1][1]

                vertex_0_coord = np.array([x[vertex_0_y, vertex_0_x],
                                           y[vertex_0_y, vertex_0_x],
                                           z[vertex_0_y, vertex_0_x]])
                vertex_1_coord = np.array([x[vertex_1_y, vertex_1_x],
                                           y[vertex_1_y, vertex_1_x],
                                           z[vertex_1_y, vertex_1_x]])

                handle = np.array([vertex_0_coord, vertex_1_coord])

                return handle, points_handle

    return None, None
