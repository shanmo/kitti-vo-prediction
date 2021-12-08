from math import tau
import numpy as np 
import transforms3d as tf

class Isometry3d(object):
    """
    3d rigid transform.
    """
    def __init__(self, q, t):
        self.R = tf.quaternions.quat2mat(q)
        self.t = np.squeeze(t)

    def orientation(self): 
        return tf.quaternions.mat2quat(self.R)

    def position(self): 
        return self.t 

    def matrix(self):
        m = np.eye(4)
        m[:3, :3] = self.R
        m[:3, 3] = self.t
        return m

    def inverse(self):
        return Isometry3d(tf.quaternions.mat2quat(self.R.T), -self.R.T @ self.t)

    def mul_trans(self, t_in):
        t_out = self.R @ np.squeeze(t_in) + self.t 
        return t_out

    def __mul__(self, T1):
        R = self.R @ T1.R
        t = self.R @ T1.t + self.t
        return Isometry3d(tf.quaternions.mat2quat(R), t)