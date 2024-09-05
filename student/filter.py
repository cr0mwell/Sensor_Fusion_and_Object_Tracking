# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
# Copyright (C) 2024, Oleksandr Kashkevich.
#
# Purpose of this file : Kalman filter class
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

import numpy as np

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import misc.params as params 


class Filter:
    '''Kalman filter class'''

    TIME_DELTA = params.dt
    P_NOISE = params.q
    DIM_STATE = params.dim_state

    def __init__(self):
        pass

    def F(self):
        return np.array([[1, 0, 0, self.TIME_DELTA, 0, 0],
                         [0, 1, 0, 0, self.TIME_DELTA, 0],
                         [0, 0, 1, 0, 0, self.TIME_DELTA],
                         [0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 1]])

    def Q(self):
        return np.identity(self.DIM_STATE) * self.P_NOISE**self.P_NOISE

    def predict(self, track):
        F = self.F()
        track.set_x(F @ track.x)
        track.set_P(F @ track.P @ F.transpose() + self.Q())

    def update(self, track, meas):
        h_x = meas.sensor.get_hx(track.x)
        H = meas.sensor.get_H(track.x)
        gamma = meas.z - h_x  # residual
        S = H @ track.P @ H.transpose() + meas.R  # covariance of residual
        K = track.P @ H.transpose() @ np.linalg.inv(S)  # Kalman gain
        track.set_x(track.x + K @ gamma)  # state update
        I = np.identity(self.DIM_STATE)
        track.set_P((I - K @ H) @ track.P)  # covariance update

        track.update_attributes(meas)
