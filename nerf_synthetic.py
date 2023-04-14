# a data class for synthetic data

import numpy as np
import cv2
import imageio
import torch
import random
import os
import json

class NeRFSynthetic:
    def __init__(self, image_path, split='test', white_bg=True, testskip=1):
        super().__init__()
        self.image_path = image_path
        json_file = os.path.join(image_path, f'transforms_{split}.json')
        with open(json_file, 'r') as f:
            meta = json.load(f)
        self.blender2opencv = np.array([
            [1,  0,  0, 0], 
            [0, -1,  0, 0], 
            [0,  0, -1, 0], 
            [0,  0,  0, 1]])
        self.img_paths = []
        self.imgs = []
        self.c2ws = []

        for frame in meta['frames'][::testskip]:
            img_path = os.path.join(image_path, frame['file_path'])
            self.img_paths.append(img_path)
            self.c2ws.append(np.array(frame['transform_matrix']) @ self.blender2opencv)
            img = imageio.imread(img_path + '.png')
            if white_bg and img.shape[-1] == 4:
                img = img / 255.
                img = img[:,:,:3] * img[:,:,3:4] + (1 - img[:,:,3:4])
                img = (img.clip(0, 1) * 255).astype(np.uint8)
            self.imgs.append(img)
        
        H, W = self.imgs[0].shape[:2]
        camera_angle_x = float(meta['camera_angle_x'])
        focal = 0.5 * W / np.tan(0.5 * camera_angle_x)
        self.K = np.array([[focal, 0, 0.5 * W],
                            [0, focal, 0.5 * H],
                            [0, 0, 1]])
        self.H, self.W = H, W
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = self.imgs[idx]
        c2w = self.c2ws[idx]
        img_path = self.img_paths[idx]
        return img, c2w, img_path


if __name__ == "__main__":
    data = NeRFSynthetic('data/nerf_synthetic/lego', split='test', white_bg=True, testskip=10)

    import matplotlib.pyplot as plt

    plt.imshow(data[0][0])
    plt.show()





