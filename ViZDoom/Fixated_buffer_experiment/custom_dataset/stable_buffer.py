import os
from torchvision.io import read_image
from torch.utils.data import Dataset
import numpy as np
import cv2

class PairedImageDataset(Dataset):
    def __init__(self, is_test=False, length = None):
        self.img_labels = np.load("/home/dvasilev/doom_dataset/no_action_repeat/train/0/actions.npy")
        self.start_img_dir = "/home/dvasilev/doom_dataset/no_action_repeat/train/0/start_frames"
        self.end_img_dir = "/home/dvasilev/doom_dataset/no_action_repeat/train/0/end_frames"
        if is_test:
            self.start_img_dir = "/home/dvasilev/doom_dataset/no_action_repeat/test/0/start_frames"
            self.end_img_dir = "/home/dvasilev/doom_dataset/no_action_repeat/test/0/end_frames"
            self.img_labels = np.load("/home/dvasilev/doom_dataset/no_action_repeat/test/0/actions.npy")
        self.length = length

    def __len__(self):
        if self.length is None:
            return len(self.img_labels)
        return self.length
    def __getitem__(self, idx):
        start_frame = np.load(self.start_img_dir+"/"+str(idx)+".npy")
        end_frame = np.load(self.end_img_dir+"/"+str(idx)+".npy")
        label = self.img_labels[idx]
        return start_frame, end_frame, label

class MultiAgentDataset(Dataset):
    def __init__(self, name_list, is_test = False, length=None):
        subname = "test" if is_test else "train"
        root_folder = "/home/dvasilev/doom_dataset/no_action_repeat"
        self.labels, self.start_frames_path, self.end_frames_path = {}, {}, {}
        self.quantity_of_folders = len(name_list)
        self.names = name_list
        for name in name_list:
            path = root_folder + "/" + subname + "/" + name
            self.labels[name] = np.load(path+"/actions.npy")
            self.start_frames_path[name] = path+"/start_frames"
            self.end_frames_path[name] = path+"/end_frames"
        self.length = length

    def __len__(self):
        if self.length is None:
            length = 0
            for label in self.labels.values():
                length += len(label)
            return length
        else:
            return self.length

    def __getitem__(self, idx):
        folder_idx = idx % self.quantity_of_folders
        item_idx = idx//self.quantity_of_folders
        start_folder, end_folder = (self.start_frames_path[self.names[folder_idx]],
                                    self.end_frames_path[self.names[folder_idx]])
        start_frame = np.load(start_folder + "/" + str(item_idx) + ".npy")
        end_frame = np.load(end_folder + "/" + str(item_idx) + ".npy")
        label = self.labels[self.names[folder_idx]][item_idx]
        return start_frame, end_frame, label
