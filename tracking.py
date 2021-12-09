import cv2
import numpy as np
import transforms3d as tf 

from utils import Isometry3d

class Tracking(object):
    def __init__(self, params):
        self.min_measurements = params.pnp_min_measurements
        self.max_iterations = params.pnp_max_iterations

    def refine_pose(self, pose, cam, measurements):
        if len(measurements) < self.min_measurements: 
            return pose 
        pose_pnp = self.pnp(cam, pose, measurements)
        return pose_pnp

    @staticmethod 
    def pnp(cam, pose, measurements): 
        # N x 3 points in 3D 
        model_points = np.zeros((0, 3))
        # N x 2 image points 
        image_points = np.zeros((0, 2))
        for _, m in enumerate(measurements):
            pt3d = np.reshape(m.mappoint.position, (-1, 3))
            model_points = np.vstack((model_points, pt3d))
            pt2d = np.reshape(m.xy, (-1, 2))
            image_points = np.vstack((image_points, pt2d))

        camera_matrix = np.array(
                [[cam.fx, 0, cam.cx],
                [0, cam.fy, cam.cy],
                [0, 0, 1]], dtype = "double"
                )
        dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion

        pose_inv = pose.inverse()
        direc, angle = tf.axangles.mat2axangle(pose_inv.R)
        raux = angle * direc
        taux = pose_inv.t 

        retval, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(model_points, 
            image_points, camera_matrix, dist_coeffs, 
            raux, taux, useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE)
        
        angle = np.linalg.norm(rotation_vector)
        dirc = rotation_vector / angle 
        q = tf.quaternions.axangle2quat(dirc, angle)
        t = translation_vector
        pose_out = Isometry3d(q, t)
        return pose_out.inverse()