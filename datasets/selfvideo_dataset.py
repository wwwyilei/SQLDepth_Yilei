# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
import os
import numpy as np
import PIL.Image as pil

from .mono_dataset_selfvideo import MonoDataset


class SelfvideoDataset(MonoDataset):
    """Cityscapes dataset - this expects triplets of images concatenated into a single wide image,
    which have had the ego car removed (bottom 25% of the image cropped)
    """

    RAW_WIDTH = 1280
    RAW_HEIGHT = 720

    def __init__(self, *args, **kwargs):
        super(SelfvideoDataset, self).__init__(*args, **kwargs)

    def index_to_folder_and_frame_idx(self, index):
        """Convert index in the dataset to a folder name, frame_idx and any other bits

        txt file is of format:
            ulm ulm_000064_000012
        """
        frame_name = self.filenames[index]
        side = None
        return frame_name, side

    def check_depth(self):
        return False

    def load_intrinsics(self):
        # adapted from sfmlearner
        intrinsics = np.array([[0.3458, 0, 0.5, 0],
                           [0, 0.6147, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        intrinsics[0, :] /= self.RAW_WIDTH
        intrinsics[1, :] /= self.RAW_HEIGHT
        return intrinsics

    def get_colors(self, frame_name, side, do_flip):
        if side is not None:
            raise ValueError("Cityscapes dataset doesn't know how to deal with sides")

        color = self.loader(self.get_image_path(frame_name))
        color = np.array(color)

        w = color.shape[1] // 3
        inputs = {}
        inputs[("color", -1, -1)] = pil.fromarray(color[:, :w])
        inputs[("color", 0, -1)] = pil.fromarray(color[:, w:2*w])
        inputs[("color", 1, -1)] = pil.fromarray(color[:, 2*w:])

        if do_flip:
            for key in inputs:
                inputs[key] = inputs[key].transpose(pil.FLIP_LEFT_RIGHT)

        return inputs

    def get_image_path(self, frame_name):
        return os.path.join(self.data_path, "{}.png".format(frame_name))
