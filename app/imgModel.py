# -*- coding: utf-8 -*-
"""
Created on Wed May  6 13:59:03 2020

@author: jhans
"""

import os
import cv2
import numpy as np
import random
import colorsys as cs


def apply_mask(ov, uv, aerial, alpha):
    shape_mask = ov.shape / np.max(ov.shape)
    shape_color = uv.shape_underlay

    text_mask = ov.text / np.max(ov.text)
    text_color = uv.text_underlay

    shape_color = (1 - text_mask * alpha) * shape_color + text_mask * alpha * text_color
    shape_color = shape_color.astype(np.uint8)

    aerial = (1 - shape_mask * alpha) * aerial + shape_mask * alpha * shape_color
    aerial = aerial.astype(np.uint8)
    return aerial


def extract_roi(mask):
    roi = np.argwhere(mask[:, :, 0] != 0)
    c1 = np.min(roi, axis=0)
    c2 = np.max(roi, axis=0)
    cen = ((c1 + c2) / 2).astype(int)
    hw = c2 - c1
    c_len = max(hw)
    return roi, cen, hw, c_len


def place_masks(shape_mask, text_mask, bg_h=224, bg_w=224, cen_x=None, cen_y=None):
    shape_mask_h = shape_mask.shape[0]
    shape_mask_w = shape_mask.shape[1]

    dx = -1 * shape_mask_w
    dy = -1 * shape_mask_h

    padded_h = bg_h + shape_mask_h * 2
    padded_w = bg_w + shape_mask_w * 2
    shape_bg = np.zeros(shape=(padded_h, padded_w, 3)).astype(np.uint8)
    text_bg = np.zeros(shape=(padded_h, padded_w, 3)).astype(np.uint8)

    crop_x1 = shape_mask_w
    crop_y1 = shape_mask_h
    crop_x2 = crop_x1 + bg_w
    crop_y2 = crop_y1 + bg_h

    xmin = crop_x1 + shape_mask_w // 8  # want at least 2/3 of the target to be in the img when cropped
    ymin = crop_y1 + shape_mask_h // 8
    xmax = crop_x2 - shape_mask_w // 8
    ymax = crop_y2 - shape_mask_h // 8

    if cen_x is None:
        cen_x = np.random.randint(xmin, xmax)
    else:
        cen_x -= dx
        cen_x = np.clip(cen_x, xmin, xmax)

    if cen_y is None:
        cen_y = np.random.randint(ymin, ymax)
    else:
        cen_y -= dy
        cen_y = np.clip(cen_y, ymin, ymax)

    replace_x1 = cen_x - shape_mask_w // 2
    replace_y1 = cen_y - shape_mask_w // 2
    replace_x2 = replace_x1 + shape_mask_w
    replace_y2 = replace_y1 + shape_mask_h

    shape_bg[replace_y1:replace_y2, replace_x1:replace_x2, :] = shape_mask
    text_bg[replace_y1:replace_y2, replace_x1:replace_x2, :] = text_mask

    _, mask_cen, mask_hw, _ = extract_roi(shape_bg)
    corner1 = mask_cen - mask_hw // 2
    corner2 = mask_cen + mask_hw // 2
    bbox = np.concatenate((corner1, corner2))
    bbox = np.clip(bbox, crop_x1, crop_x2)

    bbox[0] += dx
    bbox[1] += dy
    bbox[2] += dx
    bbox[3] += dy

    shape_cropped = shape_bg[crop_y1:crop_y2, crop_x1:crop_y2, :]
    text_cropped = text_bg[crop_y1:crop_y2, crop_x1:crop_y2, :]

    return shape_cropped, text_cropped, bbox


def extract_edge(img):
    h, w, _ = img.shape
    edge_map = cv2.Canny(img, h, w)
    edges = np.argwhere(edge_map > 0)
    return edges


def reduce_edge(img, w):
    for i in range(w):
        edge_pixels = extract_edge(img)
        n = np.random.randint(0, len(edge_pixels))
        p = np.unique(np.random.randint(0, len(edge_pixels), n))
        pixels_to_drop = edge_pixels[p, :]
        img[pixels_to_drop[:, 0], pixels_to_drop[:, 1], :] = (0, 0, 0)
    return img


def shape_rethreshold(mask):
    return (mask // np.max(mask)) * 255


def text_rethreshold(mask):
    roi, _, _, _ = extract_roi(mask)
    for pair in roi:
        mask[pair[0], pair[1], :] = (255, 255, 255)
    return mask


class ImgCollection:
    """
     Essentially a container for holding the sequence of images in a directory in memory for 
     rapid access. Beware of the the total size of images in the directory. Not recommended
     for large number or large memory images.
     
    Parameters
    ----------
    img_dir : string 
        directory of the images to be put in memory
    
    Returns
    ----------
    None.
        
    """

    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.short_fnames = list(os.listdir(img_dir))
        self.short_fnames.sort()
        self.fnames = [os.path.join(img_dir, i) for i in self.short_fnames]
        self.imgs = [cv2.imread(i) for i in self.fnames]


class AerialExtractor:
    """
     A quasi-utility class to extract smaller (sub_h x sub_w) images from a collection of larger
     images. Used for extracting from large aerial/satellite photos of the competition site. 
     Performs random cropping, rotation, and downsampling to generate variety.
     
    Parameters
    ----------
    ImgCollection : ImgCollection 
        an ImgCollection object that holds the sequence of aerial/sat background images
    
    Returns
    ----------
    None.
        
    """

    def __init__(self, ImgCollection):

        self.collection = ImgCollection

    def extract_write_to_file(self, num_extracted, write_dir, sub_h, sub_w, margin=10):
        """
        method to extract a number of sub images of a certain size and write them to a directory.

        Parameters
        ----------
        num_extracted : int
            number of sub images to extract.
        write_dir : string
            path to the directory to write to.
        sub_h : int
            height of the extracted images.
        sub_w : int
            width of the extracted images.
        margin : int, optional
            the number of pixels as buffer between the extracted img and the source img's borders. The default is 10.

        Returns
        -------
        None.

        """

        file_seed = len(os.listdir(write_dir))

        for i in range(num_extracted):
            file_num = str(file_seed + i)
            write_path = os.path.join(write_dir, file_num + "." + 'jpg')

            print('extracting {}/{} images of dimension {}x{}'.format(i, num_extracted, sub_h, sub_w))
            print('writting to location: {}'.format(write_path))

            self.extract_single(sub_h, sub_w, write_path)

    def extract_single(self, sub_h, sub_w, zoom_min=2, zoom_max=4, margin=10, write_path=None):
        """
        method to extract a single sub image from an larger source image.

        Parameters
        ----------
        sub_h : int
            height of the sub image.
        sub_w : int
            width of the sub image.
        write_path : string, optional
            path to the directory to write sub image to. The default is None.
        margin : int, optional
            the number of pixels as buffer between the extracted img and the source img's borders. The default is 10.
        zoom_min: int, optional
            minimum amount of zoom on a region. think of it as upsampling an image of dimension (sub_h/zoom, sub_w/zoom)
            to (sub_h,sub_w), where zoom_min<zoom<zoom_max. used to compensate for the grainy-ness of zooming into small 
            regions of a large arial photo.The default is 2, at least 2x zoom.
        zoom_max: int, optional
            max amount of zoom on a region. think of it as upsampling an image of dimension (sub_h/zoom, sub_w/zoom)
            to (sub_h,sub_w) where zoom_min<zoom<zoom_max. used to compensate for the grainy-ness of zooming into small 
            regions of a large arial photo.The default is 2, max 4x zoom.

        Returns
        -------
        img : ndarray
            extracted image.

        """

        k = np.random.randint(0, len(self.collection.short_fnames))

        name = self.collection.short_fnames[k]
        bg = self.collection.imgs[k]
        big_h, big_w, _ = bg.shape

        zoomed_h, zoomed_w = self._zoom(sub_h, sub_w, zoom_min, zoom_max)

        if 'fs' in name:
            # print('fs bg')
            zoomed_h = sub_h
            zoomed_w = sub_w

        circumscribe_radius = np.ceil(np.sqrt(zoomed_h ** 2 + zoomed_w ** 2))
        circumscribe_radius = int(circumscribe_radius) + margin

        cen_x = np.random.randint(circumscribe_radius, big_w - circumscribe_radius)
        cen_y = np.random.randint(circumscribe_radius, big_h - circumscribe_radius)

        x1 = int(cen_x - circumscribe_radius / 2)
        y1 = int(cen_y - circumscribe_radius / 2)

        x2 = int(cen_x + circumscribe_radius / 2)
        y2 = int(cen_y + circumscribe_radius / 2)

        raw_crop = bg[y1:y2, x1:x2]
        # cv2.imshow('raw crop',raw_crop)

        rotated = self._rotate(raw_crop, zoomed_h, zoomed_w)
        rotated = cv2.resize(rotated, (sub_w, sub_h))

        img = self._bg_augmentation(rotated)
        # cv2.imshow('rotated',rotated)
        # print(rotated.shape)

        if write_path is not None:
            cv2.imwrite(write_path, img)
        return img

    def _zoom(self, sub_h, sub_w, zoom_min, zoom_max):
        zoom_lvl = np.random.randint(zoom_min, zoom_max + 1)
        # print(zoom_lvl)
        h_reduced = int(sub_h / zoom_lvl)
        w_reduced = int(sub_w / zoom_lvl)
        return h_reduced, w_reduced

    def _rotate(self, img, cropped_h, cropped_w):

        ang = np.random.randint(0, 360)
        # print(ang)
        h, w, _ = img.shape

        cen_x = int(w / 2)
        cen_y = int(h / 2)

        M = cv2.getRotationMatrix2D((cen_x, cen_y), ang, 1)
        dst = cv2.warpAffine(img, M, (h, w))

        x1 = int(cen_x - cropped_w / 2)
        y1 = int(cen_y - cropped_h / 2)
        x2 = int(cen_x + cropped_w / 2)
        y2 = int(cen_y + cropped_h / 2)

        dst = dst[y1:y2, x1:x2]

        return dst

    def _brighten(self, img):

        brightness = np.random.uniform(0.75, 2)
        # print(brightness)
        brightened = np.clip(img * brightness, 0, 255).astype(np.uint8)

        return brightened

    def _hue_adj(self, img):

        adj = np.random.uniform(-0.1, 0.1)
        work_channel = np.random.randint(0, 3)
        # work_channel=1
        # print(adj)
        # print(work_channel)
        # print('--------------')

        changed = np.clip(img * (1 - adj), 0, 255).astype(np.uint8)
        changed[:, :, work_channel] = np.clip(changed[:, :, work_channel] * (1 + adj), 0, 255).astype(np.uint8)

        return changed

    def _bg_augmentation(self, img):

        num_augmentation = np.random.randint(0, 3)
        # print(num_augmentation)

        if num_augmentation == 0:
            # print('no aug')
            return img
        elif num_augmentation == 1:
            aug_choice = np.random.randint(0, 2)
            if aug_choice == 0:
                # print('brighten')
                return self._brighten(img)
            else:
                # print('hue adj')
                return self._hue_adj(img)
        else:
            # print('brighten and hue')
            b = self._brighten(img)
            return self._hue_adj(b)


class Underlay:
    red = {'hue': [-5, 5], 'sat': [0.9, 1], 'val': [0.3, 0.6]}
    orange = {'hue': [25, 35], 'sat': [0.8, 1], 'val': [0.7, 0.9]}
    yellow = {'hue': [50, 60], 'sat': [0.9, 1], 'val': [0.9, 1.0]}
    green = {'hue': [90, 140], 'sat': [0.7, 1], 'val': [0.2, 0.6]}
    blue = {'hue': [170, 230], 'sat': [0.5, 1], 'val': [0.3, 0.6]}
    purple = {'hue': [280, 310], 'sat': [0.3, 1], 'val': [0.3, 0.6]}
    white = {'hue': [0, 360], 'sat': [0.0, 0.01], 'val': [0.95, 1]}
    gray = {'hue': [0, 360], 'sat': [0.0, 0.05], 'val': [0.3, 0.5]}
    black = {'hue': [0, 360], 'sat': [0.0, 0.05], 'val': [0, 0.1]}
    brown = {'hue': [25, 35], 'sat': [0.5, 0.7], 'val': [0.2, 0.3]}

    color_specs = [red, orange, yellow, green, blue, purple, white, gray, black, brown]
    color_labels = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'white', 'gray', 'black', 'brown']

    def __init__(self):

        self.shape_underlay = None
        self.text_underlay = None

        self.info = dict()

    def _pick_color(self, seed=None):

        k = np.random.randint(0, len(Underlay.color_specs))
        if seed is not None:
            k = np.argwhere(np.array(Underlay.color_labels) == seed).item()

        spec = Underlay.color_specs[k]
        label = Underlay.color_labels[k]

        hue = np.random.uniform(spec['hue'][0], spec['hue'][1])
        sat = np.random.uniform(spec['sat'][0], spec['sat'][1])
        val = np.random.uniform(spec['val'][0], spec['val'][1])

        if hue < 0:
            hue += 360
        hue /= 360
        rgb = np.array(cs.hsv_to_rgb(hue, sat, val)) * 255
        rgb = np.clip(rgb, 1, 255)
        rgb = rgb.astype(np.uint8)

        # print(label)

        return rgb, label

    def _color_perturbation(self, basecolor, sub_h, sub_w, std=5):

        jitter = np.random.normal(0, std, size=(sub_h, sub_w, 3))
        color = jitter + basecolor
        color = np.clip(color, 1, 254)
        color = color.astype(np.uint8)

        return color

    def create(self, bg_h=224, bg_w=224, shape_color=None, text_color=None):

        shape_rgb, shape_rgb_label = self._pick_color(seed=shape_color)
        text_rgb, text_rgb_label = self._pick_color(seed=text_color)

        self.shape_underlay = self._color_perturbation(shape_rgb, bg_h, bg_w)
        self.text_underlay = self._color_perturbation(text_rgb, bg_h, bg_w)

        self.shape_underlay = cv2.cvtColor(self.shape_underlay, cv2.COLOR_RGB2BGR)
        self.text_underlay = cv2.cvtColor(self.text_underlay, cv2.COLOR_RGB2BGR)

        self.info['shape_color_rgb'] = shape_rgb
        self.info['shape_color_label'] = shape_rgb_label
        self.info['text_color_rgb'] = text_rgb
        self.info['text_color_label'] = text_rgb_label


class Overlay:
    """
     creates an object that contains the masks of the shape and text to be overlaid on a background
     img. A shape and an alphanumeric are chosen randomly from respective collections, scaled to a 
     desired dimension, randomly rotated, randomly placed on the an initially-black mask of the same
     dimension as the eventual photo. Random edge blurring is also applied
     
    Parameters
    ----------
    ImgCollection : ImgCollection 
        an ImgCollection object that holds the sequence of aerial/sat background images
    
    Returns
    ----------
    None.
        
    """
    shape_key = {'circle': 0,
                 'semicircle': 1,
                 'quartercircle': 2,
                 'triangle': 3,
                 'square': 4,
                 'rectangle': 5,
                 'trapezoid': 6,
                 'pentagon': 7,
                 'hexagon': 8,
                 'heptagon': 9,
                 'octagon': 10,
                 'star': 11,
                 'cross': 12,
                 'background': 13}

    text_key = {'A': 1,
                'B': 2,
                'C': 3,
                'D': 4,
                'E': 5,
                'F': 6,
                'G': 7,
                'H': 8,
                'I': 9,
                'J': 10,
                'K': 11,
                'L': 12,
                'M': 13,
                'N': 14,
                'O': 15,
                'P': 16,
                'Q': 17,
                'R': 18,
                'S': 19,
                'T': 20,
                'U': 21,
                'V': 22,
                'W': 23,
                'X': 24,
                'Y': 25,
                'Z': 26,
                '0': 27,
                '1': 28,
                '2': 29,
                '3': 30,
                '4': 31,
                '5': 32,
                '6': 33,
                '7': 34,
                '8': 35,
                '9': 36,
                }

    def __init__(self, shape_collection, alpha_collection):

        self.shape_collection = shape_collection
        self.alpha_collection = alpha_collection

        self.alpha_index = dict()
        for index, name in enumerate(self.alpha_collection.short_fnames):
            head = name.split('.')[0]
            alphanumeric = head.split('_')[1]
            if alphanumeric in self.alpha_index:
                self.alpha_index[alphanumeric].append(index)
            else:
                self.alpha_index[alphanumeric] = [index]

        self.shape = None
        self.shape_roi = None

        self.text = None
        self.text_roi = None

        self.info = {'text_label': None,
                     'text_id': None,
                     'shape_label': None,
                     'shape_id': None,
                     'x1': None,
                     'y1': None,
                     'x2': None,
                     'y2': None,
                     'text_heading': None,
                     }

    def choose_mask(self, specified_shape=None, specified_text=None):
        if specified_shape is None:
            i = np.random.randint(len(self.shape_collection.imgs))
        else:
            i = Overlay.shape_key[specified_shape]

        self.shape = self.shape_collection.imgs[i]
        shape_name = self._parse_label_from_fname(self.shape_collection.short_fnames[i])
        shape_id = Overlay.shape_key[shape_name]

        self.info['shape_label'] = shape_name
        self.info['shape_id'] = shape_id

        if specified_text is None:
            j = np.random.randint(len(self.alpha_collection.imgs))
        else:
            possible_indices = self.alpha_index[specified_text]
            j = random.choice(possible_indices)
        self.text = self.alpha_collection.imgs[j]

        text_name = self._parse_label_from_fname(self.alpha_collection.short_fnames[j])
        self.info['text_label'] = text_name
        self.info['text_id'] = Overlay.text_key[text_name]

    def edge_decay(self, weight):
        self.shape = shape_rethreshold(self.shape)
        self.text = text_rethreshold(self.text)
        self.shape = reduce_edge(self.shape, weight)

    def blur(self, shape_blur, text_blur):
        if shape_blur % 2 != 0:
            shape_blur += 1
        if text_blur % 2 != 0:
            text_blur += 1

        self.shape = cv2.blur(self.shape, (shape_blur, shape_blur))
        self.text = cv2.blur(self.text, (text_blur, text_blur))

    def place(self, bg_h=224, bg_w=224, x=None, y=None):
        self.shape, self.text, bbox = place_masks(self.shape, self.text, bg_h, bg_w, cen_x=x, cen_y=y)
        self.info['y1'] = bbox[0]
        self.info['x1'] = bbox[1]
        self.info['y2'] = bbox[2]
        self.info['x2'] = bbox[3]

    def scale(self, lower_lim, upper_lim):

        target_clen = np.random.randint(lower_lim, upper_lim)
        self.shape, self.shape_roi, _, _, _ = self._scale_one_mask(target_clen, self.shape)

        text_sf = np.random.uniform(0.2, 0.3)
        text_clen = int(text_sf * target_clen)
        self.text, self.text_roi, _, _, _ = self._scale_one_mask(text_clen, self.text)

        if self.shape.shape[0] < self.text.shape[0]:
            text_crop_x1 = self.text.shape[0] // 2 - self.shape.shape[0] // 2
            text_crop_y1 = self.text.shape[0] // 2 - self.shape.shape[0] // 2
            text_crop_x2 = text_crop_x1 + self.shape.shape[0]
            text_crop_y2 = text_crop_y1 + self.shape.shape[0]
            self.text = self.text[text_crop_y1:text_crop_y2, text_crop_x1:text_crop_x2]
            self.text_roi, _, _, _ = extract_roi(self.text)

        elif self.shape.shape[0] > self.text.shape[0]:
            pad = np.zeros(self.shape.shape).astype(np.uint8)
            text_crop_x1 = pad.shape[0] // 2 - self.text.shape[0] // 2
            text_crop_y1 = pad.shape[0] // 2 - self.text.shape[0] // 2
            text_crop_x2 = text_crop_x1 + self.text.shape[0]
            text_crop_y2 = text_crop_y1 + self.text.shape[0]
            pad[text_crop_y1:text_crop_y2, text_crop_x1:text_crop_x2] = self.text
            self.text = pad
            self.text_roi, _, _, _ = extract_roi(self.text)

    def rotate(self, ang):
        if ang is None:
            ang = np.random.randint(0, 360)
        self.info['text_heading'] = ang
        self.shape = self._rotate_one_mask(self.shape, ang)
        self.text = self._rotate_one_mask(self.text, ang)

    def _rotate_one_mask(self, mask, ang):

        w, h, _ = mask.shape
        cen_x = int(w / 2)
        cen_y = int(h / 2)

        M = cv2.getRotationMatrix2D((cen_x, cen_y), ang, 1)

        rotated_mask = cv2.warpAffine(mask, M, (h, w))

        return rotated_mask

    def _scale_one_mask(self, target_clen, mask):
        old_roi, old_cen, old_hw, old_clen = extract_roi(mask)

        sf = target_clen / old_clen

        new_h = int(mask.shape[0] * sf)
        new_w = int(mask.shape[1] * sf)

        scaled_mask = cv2.resize(mask, (new_h, new_w))
        new_roi, new_cen, new_hw, new_clen = extract_roi(scaled_mask)

        return scaled_mask, new_roi, new_cen, new_hw, new_clen

    def _parse_label_from_fname(self, fname):
        label = fname.split('_')[-1]
        label = label.split('.')
        label = label[0]
        return label
