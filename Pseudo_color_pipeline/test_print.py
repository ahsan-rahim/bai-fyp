# -*- coding: utf-8 -*-
"""
Created on Sat May 29 20:29:39 2021

@author: Ammar
"""
import cv2
import os
import matplotlib.pyplot as plt
import numpy
import sys
numpy.set_printoptions(threshold=sys.maxsize)

image_path = "C:/Users/Ammar/Desktop/IBA/8-Spring 21/FYP II/FYP Dataset/INbreast Release 1.0/pseudo_color_image/"   
ann_path = 'C:/Users/Ammar/Desktop/IBA/8-Spring 21/FYP II/FYP Dataset/INbreast Release 1.0/preprocessed_mask1/'
        


file_names = os.listdir(image_path)    

i=0
for img_name in file_names:
    
    if i < 1:
        
        img = cv2.imread(image_path+img_name)
        
        plt.imshow(img, cmap='gray')
        
        for ann_name in os.listdir(ann_path):
            
            
            if(ann_name[0:-5] == img_name[0:-4]):
                ann= plt.imread(ann_path+ann_name)
                print(ann)                
#                plt.imshow(ann,cmap='jet',alpha=0.5)
        
        plt.show()
        i+=1