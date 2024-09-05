# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
# Copyright (C) 2024, Oleksandr Kashkevich.
#
# Purpose of this file : Process the point-cloud and prepare it for object detection
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# general package imports
import cv2
import numpy as np
import torch
import zlib
import open3d as o3d

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# waymo open dataset reader
from tools.waymo_reader.simple_waymo_open_dataset_reader import dataset_pb2


# visualize lidar point-cloud
def show_pcl(pcl):
    # Create instance of open3d point-cloud class
    pcd = o3d.geometry.PointCloud()

    # Set points in pcd instance by converting the point-cloud into 3d vectors
    pcd.points = o3d.utility.Vector3dVector(pcl[:, :3])

    # Visualize point cloud and keep window open until right-arrow is pressed (key-code 262)
    o3d.visualization.draw_geometries([pcd])


# visualize range image
def show_range_image(frame, lidar_name):

    # Extract lidar data and range image for the roof-mounted lidar
    lidar = [obj for obj in frame.lasers if obj.name == lidar_name][0]  # get laser data structure from frame
    ri = []
    if len(lidar.ri_return1.range_image_compressed) > 0:  # use first response
        ri = dataset_pb2.MatrixFloat()
        ri.ParseFromString(zlib.decompress(lidar.ri_return1.range_image_compressed))
        ri = np.array(ri.data).reshape(ri.shape.dims)

    # Extract the range and the intensity channel from the range image
    range = ri[:, :, 0]
    intensity = ri[:, :, 1]
    
    # step 3 : set values <0 to zero
    range[range < 0] = 0
    intensity[intensity < 0] = 0
    intensity[intensity > 1] = 1
    
    # Map the range channel onto an 8-bit scale and make sure that the full range of values is appropriately considered
    range = (range * 255 / (np.amax(range) - np.amin(range))).astype(np.uint8)

    # Map the intensity channel onto an 8-bit scale
    max_intensity = np.amax(intensity)
    min_intensity = np.amin(intensity)
    intensity = (max_intensity * intensity * 255 / (max_intensity - min_intensity)).astype(np.uint8)

    # Stack the range and intensity image vertically
    img_range_intensity = np.vstack((range, intensity)).astype(np.uint8)

    # Displaying 90 degrees of the front view only
    deg45 = int(img_range_intensity.shape[1] / 4)
    center = int(img_range_intensity.shape[1] / 2)
    img_range_intensity = img_range_intensity[:, center - deg45 : center + deg45]

    return img_range_intensity


# create birds-eye view of lidar data
def bev_from_pcl(lidar_pcl, configs):

    # remove lidar points outside detection area and with too low reflectivity
    mask = np.where((lidar_pcl[:, 0] >= configs.lim_x[0]) & (lidar_pcl[:, 0] <= configs.lim_x[1]) &
                    (lidar_pcl[:, 1] >= configs.lim_y[0]) & (lidar_pcl[:, 1] <= configs.lim_y[1]) &
                    (lidar_pcl[:, 2] >= configs.lim_z[0]) & (lidar_pcl[:, 2] <= configs.lim_z[1]))
    lidar_pcl = lidar_pcl[mask]

    #######################
    # VISUALIZE POINT CLOUD
    #######################
    # Compute bev-map discretization by dividing x-range by the bev-image height
    bev_discretization = (configs.lim_x[1] - configs.lim_x[0]) / configs.bev_height

    # Create a copy of the lidar pcl and transform all metrix x-coordinates into bev-image coordinates
    bev = np.copy(lidar_pcl)
    bev[:, 0] = np.int_(np.floor(bev[:, 0] / bev_discretization))

    # Perform the same operation as in step 2 for the y-coordinates but make sure that no negative bev-coordinates occur
    bev[:, 1] = np.int_(np.floor(bev[:, 1] / bev_discretization) + (configs.bev_width + 1) / 2)

    # Shift level of ground plane to avoid flipping from 0 to 255 for neighboring pixels
    bev[:, 2] = bev[:, 2] - configs.lim_z[0]

    # Visualize point-cloud using the function show_pcl from a previous task
    #show_pcl(bev)
    
    ############################################
    # COMPUTE THE INTENSITY LAYER OF THE BEV MAP
    ############################################
    # create a numpy array filled with zeros which has the same dimensions as the BEV map
    intensity_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1))

    # Re-arrange elements in lidar_pcl_cpy by sorting first by x, then y, then -z (use numpy.lexsort)
    idxs = np.lexsort((-bev[:, 3], bev[:, 1], bev[:, 0]))
    bev_intensity = bev[idxs]

    # Extract all points with identical x and y such that only the top-most z-coordinate is kept (use numpy.unique)
    # also, store the number of points per x,y-cell in a variable named "counts" for use in the next task
    _, idxs, intensity_counts = np.unique(bev_intensity[:, :2], axis=0, return_index=True, return_counts=True)
    bev_intensity = bev_intensity[idxs]

    # Assign the intensity value of each unique entry in lidar_top_pcl to the intensity map
    # make sure that the intensity is scaled in such a way that objects of interest (e.g. vehicles) are clearly visible
    # also, make sure that the influence of outliers is mitigated by normalizing intensity
    # on the difference between the max. and min. value within the point cloud
    bev_intensity[bev_intensity[:, 3] > 1.0, 3] = 1.0
    intensity_map[np.int_(bev_intensity[:, 0]), np.int_(bev_intensity[:, 1])] = \
        bev_intensity[:, 3] / (np.amax(bev_intensity[:, 3]) - np.amin(bev_intensity[:, 3]))
    """
    # Temporarily visualize the intensity map using OpenCV to make sure that vehicles separate well from the background
    intens_map = (intensity_map * 256).astype(np.uint8)
    while True:
        cv2.imshow('img_intensity', intens_map)
        if cv2.waitKey(10) & 0xFF == 27:
            break
    cv2.destroyAllWindows()
    """
    ########################################
    # COMPUTE THE HEIGHT LAYER OF THE BEV MAP
    ########################################
    # Create a numpy array filled with zeros which has the same dimensions as the BEV map
    height_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1))

    # Assign the height value of each unique entry in lidar_top_pcl to the height map
    # make sure that each entry is normalized between the upper and lower height defined in the config file
    # use the lidar_pcl_top data structure from the previous task to access the pixels of the height_map
    idxs = np.lexsort((-bev[:, 2], bev[:, 1], bev[:, 0]))
    bev_height = bev[idxs]

    _, idxs = np.unique(bev_height[:, :2], axis=0, return_index=True)
    bev_height = bev_height[idxs]

    height_map[np.int_(bev_height[:, 0]), np.int_(bev_height[:, 1])] = \
        bev_height[:, 2] / (configs.lim_z[1] - configs.lim_z[0])

    """
    # Temporarily visualize the height map using OpenCV to make sure that vehicles separate well from the background
    heigh_map = (height_map * 256).astype(np.uint8)
    while True:
        cv2.imshow('img_height', heigh_map)
        if cv2.waitKey(10) & 0xFF == 27:
            break
    cv2.destroyAllWindows()
    """

    # Compute density layer of the BEV map
    density_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1))
    normalizedCounts = np.minimum(1.0, np.log(intensity_counts + 1) / np.log(64))
    density_map[np.int_(bev_intensity[:, 0]), np.int_(bev_intensity[:, 1])] = normalizedCounts

    """
    dens_map = (density_map * 256).astype(np.uint8)
    #density_map[density_map > 255] = 255
    while True:
        cv2.imshow('img_density', dens_map)
        if cv2.waitKey(10) & 0xFF == 27:
            break
    cv2.destroyAllWindows()
    """

    # assemble 3-channel bev-map from individual maps
    bev_map = np.zeros((3, configs.bev_height, configs.bev_width))
    bev_map[2, :, :] = density_map[:configs.bev_height, :configs.bev_width]  # r_map
    bev_map[1, :, :] = height_map[:configs.bev_height, :configs.bev_width]  # g_map
    bev_map[0, :, :] = intensity_map[:configs.bev_height, :configs.bev_width]  # b_map

    """
    my_bev = np.array(list(zip(bev_map[0, :, :].flatten(), bev_map[1, :, :].flatten(), bev_map[2, :, :].flatten()))).reshape(configs.bev_height, configs.bev_width, 3)

    while True:
        cv2.imshow('bev_map', my_bev)
        if cv2.waitKey(10) & 0xFF == 27:
            break
    cv2.destroyAllWindows()
    """

    # expand dimension of bev_map before converting into a tensor
    s1, s2, s3 = bev_map.shape
    bev_maps = np.zeros((1, s1, s2, s3))
    bev_maps[0] = bev_map

    bev_maps = torch.from_numpy(bev_maps)  # create tensor from birds-eye view
    input_bev_maps = bev_maps.to(configs.device, non_blocking=True).float()
    return input_bev_maps
