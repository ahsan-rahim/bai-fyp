# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 18:59:06 2021

@author: Ammar
"""

#%%

import numpy as np 
import pydicom
import png
import os
import re

#%%
findfiles = os.listdir(
'C:/Users/Ammar/Desktop/IBA/8-Spring 21/FYP II/FYP Dataset/AllDICOMs')
dcm = []

for file in findfiles:
  if file.endswith(".dcm"):
    dcm.append(os.path.join(file))

print(dcm)

print(len(dcm))


files = []

for name in dcm:
  files.append(re.findall( r"^([^.]*).*" , name)[0])

print(files)



#%%


for file in files:
    ds = pydicom.dcmread('AllDICOMs/%s.dcm'%file)
    shape = ds.pixel_array.shape
    
    # Convert to float to avoid overflow or underflow losses.
    image_2d = ds.pixel_array.astype(float)

    # Rescaling grey scale between 0-255
    image_2d_scaled = (np.maximum(image_2d,0) / image_2d.max()) * 256

    # Convert to uint
    image_2d_scaled = np.uint8(image_2d_scaled)

    with open('AIIPNG/%s.png'%file, 'wb') as png_file:
        w = png.Writer(shape[1], shape[0], greyscale=True)
        w.write(png_file, image_2d_scaled)




#%%






