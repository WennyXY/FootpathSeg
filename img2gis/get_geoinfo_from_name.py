import numpy as np
import cv2
import os
import shutil
import tqdm
import sys
import argparse
import random

import math
import pandas as pd
import geopandas
from shapely.geometry import Polygon
import matplotlib.pyplot as plt


#SBATCH --partition=deeplearn
#SBATCH --qos=gpgpudeeplearn

#SBATCH --partition=feit-gpu-a100
#SBATCH --qos=feit

def num2deg(xtile, ytile, zoom):
    # 
    # tiles number to lat & lon
    # 
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)


def geoinfo_to_mask(datapath, maskpath, labelpath):
    # 
    # datapath: remote sensing images
    # maskpath: annotation file
    # labelpath: to save generated mask images
    # 
    footpath = geopandas.read_file(maskpath)
    footpath_4326 = footpath.to_crs('EPSG:4326')
    imgs_all = os.listdir(datapath)
    mask = []
    i = 0
    # imgs = ['144.82830047607422,-37.86577423173838~1892262,1287229~Melbourne_2020_01_GM~21.jpg']  # os.listdir(datapath)

    for img_name in imgs_all:
        i += 1
        lon_lat, coordinates, _, zoom  = img_name.split('~')
        lon, lat = lon_lat.split(',')
        lat = float(lat)
        lon = float(lon)
        x_tile, y_tile = coordinates.split(',')
        x_tile = int(x_tile)
        y_tile = int(y_tile)
        zoom = int(zoom.split('.')[0])
        
        next_tile_lat, next_tile_lon = num2deg(x_tile+1, y_tile+1, zoom)
        # offset_lat = 0.000021
        # offset_lon = 0.000008 # + left - right
        
        dif_lat = abs(next_tile_lat - lat)
        dif_lon = abs(next_tile_lon - lon)
        # offset_lat = 0.00.14
        # offset_lon = 0.00.02
        offset_lat = 0.13 * dif_lat # + bottom - top
        offset_lon = 0.03 * dif_lon # + left - right
        min_x = min(lon, next_tile_lon) + offset_lon
        min_y = min(lat, next_tile_lat) + offset_lat
        max_x = max(lon, next_tile_lon) + offset_lon
        max_y = max(lat, next_tile_lat) + offset_lat
        
        polygon = Polygon([(min_x, max_y), (max_x, max_y), (max_x, min_y), (min_x, min_y), (min_x, max_y)])
        poly_gdf = geopandas.GeoDataFrame([1], geometry=[polygon], crs=4326)
        mask_gdf_4326 = footpath_4326.clip(poly_gdf)
        if not mask_gdf_4326.empty:
            dpi = 300
            fig, ax = plt.subplots(figsize=(256/dpi, 256/dpi), dpi=dpi)
            ax.set_axis_off()
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.margins(0,0)
            poly_gdf.plot(ax=ax,color='black')
            mask_gdf_4326.plot(ax=ax,color='white')
            plt.savefig(labelpath + img_name,pad_inches=0)  # dpi=100 和上文相对应 pixel尺寸/dpi=inch尺寸
            plt.close()
            print(i)
    print('finish')
    return

def overlap_img_out(datapath, labelpath, savepath):
    imgs = os.listdir(labelpath)
    for img_name in imgs:
        # print(datapath + img_name)
        # print(labelpath + img_name)
        img = cv2.imread(datapath + img_name.replace('out_', ''))
        mask = cv2.imread(labelpath + img_name, cv2.IMREAD_GRAYSCALE)
        # 两张图合为一个
        # _ , out = cv2.threshold(out, 180, 255, 0)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        # image_combine = cv2.hconcat([img, mask])
        overlapping = cv2.addWeighted(img, 1, mask, 0.3, 0)
        cv2.imwrite(savepath + '/' + img_name.replace('out_', ''), overlapping)
    print('finish')

def divide_data(dataset_path, new_path, rate):
    # dirs = os.listdir(dataset_path)
    print('create new directories......')
    if not os.path.exists(new_path+'/train'):
        os.mkdir(new_path+'/train')
    if not os.path.exists(new_path+'/val'):
        os.mkdir(new_path+'/val')
    # for dir_name in dirs:
    files  = os.listdir(dataset_path)
    train_size = int(rate * len(files))
    test_size = len(files) - train_size
    print('size of training_set:', train_size)
    print('size of test_set:', test_size)
    train_set = random.sample(files, train_size)
    test_set = []
    for data in files:
        if data not in train_set:
            test_set.append(data)
    # print(dataset_path+'/'+dir_name)
    copy_file(dataset_path, train_set, new_path+'/train/')
    copy_file(dataset_path, test_set, new_path+'/val/')
    # print(dir_name,'category: train_size=', len(train_set), 'val_size=', len(test_set))
    # break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 
    parser.add_argument('--img_path', default='', type=str, help="Original img path.")
    parser.add_argument('--mask_path', default='', type=str, help="Mask info path.")
    parser.add_argument('--label_path', default='', type=str, help="Mask label path.")
    parser.add_argument('--overlap_path', default='', type=str, help="Overlapping img path.")

    args = parser.parse_args()
    # geoinfo_to_mask(args.img_path, args.mask_path, args.label_path)
    overlap_img_out(args.img_path, args.label_path, args.overlap_path)
