# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
# Copyright (C) 2024, Oleksandr Kashkevich.
#
# Purpose of this file : Classes for track and track management
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np
from enum import IntEnum

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import misc.params as params 


class State(IntEnum):
    INITIALIZED = 0
    TENTATIVE = 1
    CONFIRMED = 2


class Track:
    '''Track class with state, covariance, id, score'''
    DIM_STATE = params.dim_state

    def __init__(self, meas, t_id):
        print(f'Creating track {t_id}')
        M_rot = meas.sensor.sens_to_veh[0:3, 0:3]   # rotation matrix from sensor to vehicle coordinates
        
        self.x = np.zeros((self.DIM_STATE, 1))
        self.x[:3] = (M_rot @ meas.z)[:3]

        self.P = np.zeros((self.DIM_STATE, self.DIM_STATE))
        self.P[:3, :3] = M_rot @ meas.R @ M_rot.transpose()
        self.P[3, 3] = params.sigma_p44**2
        self.P[4, 4] = params.sigma_p55**2
        self.P[5, 5] = params.sigma_p66**2

        self.state = State.INITIALIZED
        self.score = 1./params.window

        # Other track attributes
        self.id = t_id
        self.width = meas.width
        self.length = meas.length
        self.height = meas.height
        # Transform rotation from sensor to vehicle coordinates
        self.yaw = np.arccos(M_rot[0, 0] * np.cos(meas.yaw) + M_rot[0, 1] * np.sin(meas.yaw))
        self.t = meas.t

    def set_x(self, x):
        self.x = x
        
    def set_P(self, P):
        self.P = P  
        
    def set_t(self, t):
        self.t = t  
        
    def update_attributes(self, meas):
        # Use exponential sliding average to estimate dimensions and orientation
        if meas.sensor.name == 'lidar':
            c = params.weight_dim
            self.width = c*meas.width + (1 - c)*self.width
            self.length = c*meas.length + (1 - c)*self.length
            self.height = c*meas.height + (1 - c)*self.height
            M_rot = meas.sensor.sens_to_veh
            # Transform rotation from sensor to vehicle coordinates
            self.yaw = np.arccos(M_rot[0, 0] * np.cos(meas.yaw) + M_rot[0, 1] * np.sin(meas.yaw))


class Trackmanagement:
    '''Track manager with logic for initializing and deleting objects'''
    def __init__(self):
        self.N = 0  # current number of tracks
        self.track_list = []
        self.last_id = -1
        self.result_list = []

    def manage_tracks(self, unassigned_tracks, unassigned_meas, meas_list):
        print(f'Received unassigned_tracks: {unassigned_tracks}, unassigned_meas: {unassigned_meas}')
        # Decrease score for unassigned tracks
        unassigned_tracks = np.array(unassigned_tracks)
        for i in unassigned_tracks:
            track = self.track_list[i]
            # Check visibility
            if meas_list:
                if meas_list[0].sensor.in_fov(track.x):
                    track.score -= 1./params.window
                    print(f'Decreased track {track.id} score to {track.score}')
                else:
                    print(f'Track {track.id} is not in FOV of {meas_list[0].sensor.name}')

            # Delete old tracks:
            # - if its covariance matrix P values are above the threshold
            # - if confirmed track's score drops below 'params.c_delete_threshold' value
            if np.any(track.P[[0, 1], [0, 1]] > params.max_P) or \
                    (track.state == State.CONFIRMED and track.score < params.delete_threshold):
                print(f'Track {track.id} P: {track.P}')
                self.delete_track(track)
                # As the track has been deleted, the rest indices should be reduced by 1
                unassigned_tracks -= 1

        # Initialize new track with unassigned measurement
        for j in unassigned_meas: 
            if meas_list[j].sensor.name == 'lidar':     # only initialize with lidar measurements
                self.init_track(meas_list[j])

    def addTrackToList(self, track):
        self.track_list.append(track)
        self.N += 1
        self.last_id = track.id

    def init_track(self, meas):
        track = Track(meas, self.last_id + 1)
        self.addTrackToList(track)

    def delete_track(self, track):
        self.track_list.remove(track)
        print(f'Deleted track {track.id}')
        
    def handle_updated_track(self, track):
        if track.score < 1:
            track.score += 1./params.window

        if track.state == State.INITIALIZED:
            print(f'Track {track.id} is now TENTATIVE')
            track.state = State.TENTATIVE
        elif track.state == State.TENTATIVE and track.score > params.confirmed_threshold:
            print(f'Track {track.id} is now CONFIRMED')
            track.state = State.CONFIRMED
