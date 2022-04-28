import os
import cv2
import numpy as np
from tensorboardX import SummaryWriter


def createLogger(self):

    if self.manager.settingsConfig.train.useTensorboard:
        self.logger = SummaryWriter(os.path.join(self.manager.base_dir, "logs"))


def save_image(save_path, im):

    cv2.imwrite(save_path, im)


def recreate_images(im_data):
    """
    Recreates images from a torch variable, sort of reverse preprocessing
    Args:
        im_as_var (torch variable): Image to recreate
    returns:
        recreated_im (numpy arr): Recreated image in array
    """
    means = (0.485, 0.456, 0.406)
    stds = (0.229, 0.224, 0.225)

    if len(im_data.shape) == 3:
        im_data = im_data.transpose(1, 2, 0)
    elif len(im_data.shape) == 4:
        im_data = im_data.transpose(0, 2, 3, 1)

    im_data = np.array(im_data, dtype=np.float32)
    rec_im = im_data*stds + means

    if means[0] < 1:
        rec_im *= 255.0

        rec_im[rec_im > 255.0] = 255.0
        rec_im[rec_im < 0.0] = 0.0

        rec_im = np.round(rec_im)
        rec_im = np.uint8(rec_im)

        if len(im_data.shape) == 3:
            rec_im = rec_im[:,:,::-1]
        elif len(im_data.shape) == 4:
            rec_im = rec_im[:, :, :, ::-1]

    else:
        rec_im[rec_im > 255.0] = 255.0
        rec_im[rec_im < 0.0] = 0.0

        rec_im = np.round(rec_im)
        rec_im = np.uint8(rec_im)

    return rec_im

