# -*- coding: utf-8 -*-
"""
Created on Sat May 29 13:14:22 2021

@author: Ammar
"""
import numpy as np
# To pad image into a square shape
def padimages(image,file_name, ratio):
    [length, width] = np.shape(image)
    if length/width>ratio:#1024/800
        print('This image needs padding.')
        add_wid = round(length*(1/ratio)-width)
        pad = np.zeros((length,add_wid))
        pad = pad.astype(image.dtype)
        if '_R_' in file_name:
        #                pad on the left
            pad_image = np.concatenate((pad,image),axis=1)
        else:
            pad_image = np.concatenate((image,pad),axis=1)
            
    return pad_image



from skimage.draw import polygon
import numpy as np
import plistlib

def load_inbreast_mask(mask_path, imshape=(4084, 3328)):
    """
    This function loads a osirix xml region as a binary numpy array for INBREAST
    dataset
    @mask_path : Path to the xml file
    @imshape : The shape of the image as an array e.g. [4084, 3328]
    return: numpy array where positions in the roi are assigned a value of 1.
    """

    def load_point(point_string):
        x, y = tuple([float(num) for num in point_string.strip('()').split(',')])
        return y, x

    mask = np.zeros(imshape)
    with open(mask_path, 'rb') as mask_file:
        plist_dict = plistlib.load(mask_file, fmt=plistlib.FMT_XML)['Images'][0]
        numRois = plist_dict['NumberOfROIs']
        rois = plist_dict['ROIs']
        assert len(rois) == numRois
        for roi in rois:
            
            if(roi['Name']=='Mass'):
              numPoints = roi['NumberOfPoints']
              points = roi['Point_px']
              assert numPoints == len(points)
              points = [load_point(point) for point in points]
              if len(points) <= 2:
                  for point in points:
                      mask[int(point[0]), int(point[1])] = 1
              else:
                  x, y = zip(*points)
                  x, y = np.array(x), np.array(y)
                  poly_x, poly_y = polygon(x, y, shape=imshape)
                  mask[poly_x, poly_y] = 1
    return mask