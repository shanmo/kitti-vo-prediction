import numpy as np 
import transforms3d as tf

class Isometry3d(object):
    """
    3d rigid transform.
    """
    def __init__(self, q, t):
        self.R = tf.quat2mat(q)
        self.t = t

    def orientation(self): 
        return tf.mat2quat(self.R)

    def position(self): 
        return self.t 

    def matrix(self):
        m = np.eye(4)
        m[:3, :3] = self.R
        m[:3, 3] = self.t
        return m

    def inverse(self):
        return Isometry3d(self.R.T, -self.R.T @ self.t)

    def __mul__(self, T1):
        R = self.R @ T1.R
        t = self.R @ T1.t + self.t
        return Isometry3d(R, t)