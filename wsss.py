import torch
import numpy as np
import glob
import cv2
import os
from PIL import Image
from skimage.segmentation import felzenszwalb
import matplotlib.pyplot as plt


def graphSegment(image_path):
    img = Image.open(image_path)
    image = np.array(img)
    segmented_image = felzenszwalb(image, scale=400, sigma=0.8, min_size=20).astype(np.uint8)
    _, counts = np.unique(image.reshape(-1, 1), axis=0, return_counts=True)
    h, w = segmented_image.shape

    #colormap for the segmentated classes
    # Generate a color map using matplotlib
    cmap = plt.get_cmap('tab20')
    colors = cmap(np.linspace(0, 1, 20))[:, :3] * 255  # Scale to [0, 255]
    colored_lmask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_index in range(len(counts)):
        #colored_lmask[segmented_image == class_index] = colors[class_index%20].astype(np.uint8)
        color = np.random.random((1, 3)).tolist()[0]
        color = [x * 255 for x in color]
        colored_lmask[segmented_image == class_index] = color#colors[class_index%20].astype(np.uint8)

    return colored_lmask


def colorSam(image_path):
    img = Image.open(image_path)
    image = np.array(img)
    values, counts = np.unique(image.reshape(-1, 1), axis=0, return_counts=True)
    h, w = image.shape

    #colormap for the segmentated classes
    # Generate a color map using matplotlib
    cmap = plt.get_cmap('tab20c')
    colors = cmap(np.linspace(0, 1, 20))[:, :3] * 255  # Scale to [0, 255]
    colored_lmask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_index in range(len(counts)):
        color = np.random.random((1, 3)).tolist()[0]
        color = [x * 255 for x in color]
        colored_lmask[image == class_index] = color#colors[class_index%20].astype(np.uint8)

    return colored_lmask


net_size = (1, 128, 128)
num_classes = 4

def main():

    # files = glob.glob('./images/*.jpg')
    # for image_path in files:

    #     colored_lmask = graphSegment(image_path)
    #     cv2.imwrite(os.path.join('./out_graphseg/', image_path.split('/')[-1]).replace('.jpg', '_graph.png'),colored_lmask)

    files = glob.glob('./sam_labels/*.png')
    for image_path in files:

        colored_lmask = colorSam(image_path)
        cv2.imwrite(os.path.join('./out_sam/', image_path.split('/')[-1]).replace('.png', '_sam.png'),colored_lmask)

if __name__ == '__main__':
    main()
