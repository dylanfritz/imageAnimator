import numpy as np


def fast_matrix_trans(img, M, *args):
    h, w = img.shape
    center = np.array([w / 2, h / 2]) # to be used to perform transfromations around center of image instead of top corner

    # Combine all matrices into one composite transform
    combined = M
    for m in args:
        combined = m @ combined  #ABx is the same as AB:=C Cx

    y, x = np.indices((h, w)) # y is height index, x is width index


    coords = np.stack((x.ravel(), y.ravel()), axis=0) # make array of x,y vectors to be transformed
    coords_centered = coords - center[:, None] # subtract center (in the shape (2,1)) from each x,y vector in coords to scale each in terms of the center

    new_coords_centered = combined @ coords_centered # perform the transformation on all x,y vectors at once. combinedTransformationMatrix[eT_1, eT_2] X cords_centered[(x,y)_1, ..., (x,y)_pixelCount]
    # [a b] @ [[x1 x2 ... xN],  → [[a*x1 + b*y1, ..., a*xN + b*yN],
    # [c d]    [y1 y2 ... yN]]     [c*x1 + d*y1, ..., c*xN + d*yN]]

    new_coords_image = new_coords_centered + center[:, None] # reframe transformed centered coords to image coords


    new_x = np.round(new_coords_image[0]).astype(int) # round transformed coordinates to nearest integer pixel indices for indexing.
    new_y = np.round(new_coords_image[1]).astype(int)

    mask = (0 <= new_x) & (new_x < w) & (0 <= new_y) & (new_y < h) # only include transformed coords in the image bounds

    transformed = np.zeros_like(img) #initialize new image to all 0s in shape (dimensions) of origial
    transformed[new_y[mask], new_x[mask]] = img[coords[1][mask], coords[0][mask]] #transformed image is the transformed coordinates, only the ones in the mask

    return transformed