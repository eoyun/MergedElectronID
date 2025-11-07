import numpy as np
import cv2
import os
import pandas as pd
import tensorflow as tf

class ImageGen:
    def __init__ (self, EEImage, ES1Image, ES2Image, dPhi, dEta, isAddTrack):
        self.EEImage = EEImage
        self.ES1Image = ES1Image
        self.ES2Image = ES2Image
        self.dPhi = dPhi
        self.dEta = dEta
        self.isAddTrack = isAddTrack
        self.total_img = ImageResize()
        self.meta = makeMeta()

    def ImageResize(self):
        EEImageResize = cv2.resize(self.EEImage, (self.EEImage.shape[1] * 14, self.EEImage.shape[0] * 14), interpolation=cv2.INTER_NEAREST)
        ES1ImageResize = cv2.resize(self.ES1Image, (self.ES1Image.shape[1] * 32, self.ES1Image.shape[0]), interpolation=cv2.INTER_NEAREST)
        ES1ImageResize = np.pad(ES1ImageResize,pad_width = 1,mode='constant')
    
        ES2ImageResize = cv2.resize(self.ES2Image, (self.ES2Image.shape[1], self.ES2Image.shape[0] * 32), interpolation=cv2.INTER_NEAREST)
        ES2ImageResize = np.pad(ES2ImageResize,pad_width = 1,mode='constant')
    
        EEImageResize = (EEImageResize / EEImageResize.max()) * 255 if EEImageResize.max() != 0 else EEImageResize
        ES1ImageResize = (ES1ImageResize / ES1ImageResize.max()) * 255 if ES1ImageResize.max() != 0 else ES1ImageResize
        ES2ImageResize = (ES2ImageResize / ES2ImageResize.max()) * 255 if ES2ImageResize.max() != 0 else ES2ImageResize
        total_img = np.stack((EEImageResize, ES1ImageResize, ES2ImageResize), axis=2)
        del EEImageResize, ES1ImageResize, ES2ImageResize
        total_img = tf.convert_to_tensor(total_img, dtype=tf.float32)

        return total_img

    def makeMeta (self) :
        return np.concatenate(self.dEta,self.dPhi)

