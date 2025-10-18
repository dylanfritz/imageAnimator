import numpy as np

def IDENTITY_MATRIX(_=None):
    return np.array([
        [1,0],
        [0,1]
        ])
def ROTATION_MATRIX(deg):
    theta = np.radians(deg)
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
        ])

def HORIZONTAL_SHEAR(shear):
    return np.array([
        [1, shear],
        [0, 1]
        ])


def VERTICAL_SHEAR(shear):
    return np.array([
        [1, 0],
        [shear, 1]
        ])

def HORIZONTAL_REFLECTION(_=None):
    return np.array([
        [1, 0],
        [0, -1]
        ])

def VERTICAL_REFLECTION(_=None):
    return np.array([
        [-1, 0],
        [0, 1]
        ])

def ARBITRARY_REFLECTION(deg):
    theta = np.radians(deg)
    return np.array([
        [np.cos(2*theta), np.sin(2*theta)],
        [np.sin(2*theta), -np.cos(2*theta)]
        ])