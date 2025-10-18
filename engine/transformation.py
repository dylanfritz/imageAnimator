import numpy as np


def fast_matrix_trans(img, M, *args):
    h, w = img.shape
    center = np.array([w / 2, h / 2])

    # Combine all matrices into one composite transform
    combined = M
    for m in args:
        combined = m @ combined  #ABx is the same as AB:=C Cx

    y, x = np.indices((h, w))


    coords = np.stack((x.ravel(), y.ravel()), axis=0)
    coords_centered = coords - center[:, None]

    new_coords = combined @ coords_centered + center[:, None]

    new_x = np.round(new_coords[0]).astype(int)
    new_y = np.round(new_coords[1]).astype(int)

    mask = (0 <= new_x) & (new_x < w) & (0 <= new_y) & (new_y < h)

    transformed = np.zeros_like(img)
    transformed[new_y[mask], new_x[mask]] = img[coords[1][mask], coords[0][mask]]

    return transformed