# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
# Copyright (C) 2024, Oleksandr Kashkevich.
#
# Purpose of this file : Data association class with single nearest neighbor association and gating based on Mahalanobis distance
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

import numpy as np
from scipy.stats.distributions import chi2

# Add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import misc.params as params 

class Association:
    '''Data association class with single nearest neighbor association and gating based on Mahalanobis distance'''
    def __init__(self):
        self.association_matrix = np.matrix([])
        self.unassigned_tracks = []
        self.unassigned_meas = []
        self.limit_lidar = chi2.ppf(0.95, df=3)
        self.limit_camera = chi2.ppf(0.95, df=2)

    def associate(self, track_list, meas_list):
        N = len(track_list)
        M = len(meas_list)
        self.association_matrix = np.inf * np.ones((N, M))
        self.unassigned_tracks = list(range(N))
        self.unassigned_meas = list(range(M))

        for i, track in enumerate(track_list):
            for j, measurement in enumerate(meas_list):
                MHD = self.MHD(track, measurement)
                if self.gating(MHD, measurement.sensor):
                    self.association_matrix[i, j] = MHD
        #print(f'Association matrix: {self.association_matrix}')

    def get_closest_track_and_meas(self):
        # Find the closest track and measurement for next update
        if not np.min(self.association_matrix) == np.inf:
            indices = np.unravel_index(np.argmin(self.association_matrix), self.association_matrix.shape)

            # Remove respective row and column from the self.association_matrix
            self.association_matrix = np.delete(self.association_matrix, (indices[0]), axis=0)
            self.association_matrix = np.delete(self.association_matrix, (indices[1]), axis=1)

            track = self.unassigned_tracks.pop(indices[0])
            meas = self.unassigned_meas.pop(indices[1])
            return track, meas

        else:
            return np.nan, np.nan

    def gating(self, MHD, sensor):
        # Check if the measurement lies inside the gate
        if sensor.name == 'lidar':
            return MHD < self.limit_lidar
        else:
            return MHD < self.limit_camera
        
    def MHD(self, track, meas):
        # Calculate the Mahalanobis distance
        h_x = meas.sensor.get_hx(track.x)
        H = meas.sensor.get_H(track.x)
        gamma = meas.z - h_x
        S = H @ track.P @ H.transpose() + meas.R
        mahalanobis_dist = gamma.transpose() @ np.linalg.inv(S) @ gamma

        return mahalanobis_dist
    
    def associate_and_update(self, manager, meas_list, KF):
        # Associate measurements and tracks
        self.associate(manager.track_list, meas_list)
    
        # Update associated tracks with measurements
        while self.association_matrix.shape[0] > 0 and self.association_matrix.shape[1] > 0:
            
            # Search for next association between a track and a measurement
            ind_track, ind_meas = self.get_closest_track_and_meas()
            if np.isnan(ind_track):
                print('---no more associations---')
                break
            track = manager.track_list[ind_track]
            
            # check visibility, only update tracks in fov    
            if not meas_list[0].sensor.in_fov(track.x):
                print(f'Track {track.id} is not in FOV of {meas_list[0].sensor.name}')
                track.last_updated -= 1
                print(f'Decreased track {track.id} last_updated to {track.last_updated}')
                continue
            
            # Kalman update
            print(f'Update track {track.id} with {meas_list[ind_meas].sensor.name} measurement {ind_meas}: ')
            KF.update(track, meas_list[ind_meas])

            # Update score and track state
            manager.handle_updated_track(track)
            
            # Save updated track
            manager.track_list[ind_track] = track
            
        # Run track management
        manager.manage_tracks(self.unassigned_tracks, self.unassigned_meas, meas_list)
        
        for track in manager.track_list:            
            print(f'Track {track.id} score is {track.score}')
