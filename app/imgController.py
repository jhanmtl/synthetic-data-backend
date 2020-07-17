# -*- coding: utf-8 -*-
"""
Created on Thu May  7 16:20:26 2020

@author: jhans
"""


import numpy as np
import os
import cv2
import copy
import pathlib
full_path=pathlib.Path(__file__).parent.absolute()
import sys
sys.path.append(str(full_path))
import imgModel as model


class Generator:
    
    def __init__(self,h,w):
        
        bg_dir=os.path.join(str(pathlib.Path(full_path)), r'resources/sat_imgs2')
        alpha_dir=os.path.join(str(pathlib.Path(full_path)), r'resources/masks/alphanumerics')
        shape_dir=os.path.join(str(pathlib.Path(full_path)), r'resources/masks/shapes')
        
        alpha_collection=model.ImgCollection(alpha_dir)
        shape_collection=model.ImgCollection(shape_dir)
        sat_collection=model.ImgCollection(bg_dir)
        
        self.extractor=model.AerialExtractor(sat_collection)
        self.ov=model.Overlay(shape_collection,alpha_collection)
        self.uv=model.Underlay()

        self.h=h
        self.w=w
        self.composite=None
        self.info=self.ov.info
        self.bbox=None
    
    def choose_color(self,scolor,tcolor):
        self.uv.create(shape_color=scolor,text_color=tcolor)
        self.info.update(self.uv.info)
        
    def choose_masks(self,shape=None,text=None):
        self.ov.choose_mask(specified_shape=shape,specified_text=text)
    
    def scale(self,scale_range):
        self.ov.scale(scale_range[0],scale_range[1])
       
    def rotate(self,angle=None):
        self.ov.rotate(angle)
    
    def preplace(self,x=None,y=None):
        self.ov.place(x=x,y=y)
    
    def feather(self,edge_lim_range,shape_blur_range,text_blur_range):
        edge_weight=np.random.randint(edge_lim_range[0],edge_lim_range[1])
        self.ov.edge_decay(edge_weight)
        
        shape_blur=np.random.randint(shape_blur_range[0],shape_blur_range[1])
        text_blur=np.random.randint(text_blur_range[0],text_blur_range[1])
        self.ov.blur(shape_blur,text_blur)
    
    def set_text_opacity(self,alpha):
        
        shape_color=self.uv.shape_underlay
        
        text_mask=self.ov.text/np.max(self.ov.text)
        text_color=self.uv.text_underlay
        
        shape_color=(1-text_mask*alpha)*shape_color+text_mask*alpha*text_color
        shape_color=shape_color.astype(np.uint8)
        
        self.shape_underlay=shape_color
    
    def apply_mask(self,alpha_range):
        
        alpha=np.random.uniform(alpha_range[0],alpha_range[1])
        aerial=self.extractor.extract_single(self.h,self.w,zoom_min=1,zoom_max=4,margin=10,write_path=None)
        shape_mask=self.ov.shape/np.max(self.ov.shape)

        aerial=(1-shape_mask*alpha)*aerial+shape_mask*alpha*self.shape_underlay
        aerial=aerial.astype(np.uint8)
        
        self.composite=aerial
        
        xmin=self.info['x1']
        ymin=self.info['y1']
        xmax=self.info['x2']
        ymax=self.info['y2']
        self.bbox=np.array([xmin,ymin,xmax,ymax])
    
    def draw_bbox(self):
        
        xmin=self.bbox[0]
        ymin=self.bbox[1]
        xmax=self.bbox[2]
        ymax=self.bbox[3]
        original=copy.deepcopy(self.composite)
        boxed_img=cv2.rectangle(original,(xmin,ymin),(xmax,ymax),color=(0,255,0),thickness=1)
        return boxed_img