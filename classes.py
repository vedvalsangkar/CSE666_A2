# The custom datasets used for assignment 2 were built using the following site as a guide:
# https://github.com/utkuozbulak/pytorch-custom-dataset-examples
#
# Class and method structure was used with appropriate modifications as needed for this project.
#


import pandas as pd
import numpy as np

# import torchvision.transforms as transforms
# import PIL.Image as

import torch

from torchvision import transforms
from PIL import Image
from torch.utils.data.dataset import Dataset


class A2TrainDataSet(Dataset):
    """
    Custom dataset class for loading training images.
    """

    def __init__(self, csv_file, image_root_folder='IJBA_images/'):

        col_names = ['TEMPLATE_ID', 'SUBJECT_ID', 'FILE']

        self.to_tensor = transforms.ToTensor()
        self.data_info = pd.read_csv(filepath_or_buffer=csv_file,
                                     header=0,
                                     usecols=col_names,
                                     dtype=object)

        self.encoder = pd.read_csv(filepath_or_buffer="class_map.csv",
                                   header=0,
                                   index_col=1)

        # self.encoder.index = self.encoder.index.astype(object)

        self.decoder = pd.read_csv(filepath_or_buffer="class_map.csv",
                                   header=0,
                                   index_col=0,
                                   dtype=object)

        self.images = np.asarray(image_root_folder + self.data_info['SUBJECT_ID'] + '/' + self.data_info['FILE'])

        self.data_len = len(self.data_info.index)

        # self.labels = np.asarray(self.data_info['SUBJECT_ID'].replace( self.encoder['ENC']))
        self.labels = np.asarray(self.data_info['SUBJECT_ID'].astype(int))

    def __getitem__(self, index):

        # Open image
        image_name = self.images[index]
        # image_label = self.labels[index]
        image_label = self.encoder.loc[self.labels[index], 'ENC']

        # try:
        #     image = Image.open(image_name)
        #
        #     tensor_img = self.to_tensor(image)
        #
        #     return tensor_img, torch.tensor(image_label)
        #
        # except:
        #     print("Unable to access image {0} with index: {1}.".format(image_name, index))
        #     return image_name, image_label
        image = Image.open(image_name)

        tensor_img = self.to_tensor(image)

        return tensor_img, torch.tensor(image_label)
        # return tensor_img, torch.from_numpy(image_label)

    def __len__(self):
        return self.data_len

    def clean(self, split):
        del_list = []
        for i in range(self.data_len):
            try:
                image = Image.open(self.images[i])
            except:
                print("Unable to access image {0} with index: {1}. Deleting.".format(self.images[i], i))
                del_list.append(i)

        self.data_info.drop(labels=self.data_info.index[del_list],
                            inplace=True)
        self.data_info.to_csv(path_or_buf="IJBA_sets/split" + str(split) + "/train_" + str(split) + "_clean.csv",
                              index=False)


class A2VerifyDataSet(Dataset):

    def __init__(self, comp_file, meta_file, image_root_folder='IJBA_images/'):
        # col_names = ['TEMPLATE_ID', 'SUBJECT_ID', 'FILE', 'MEDIA_ID', 'SIGHTING_ID']
        meta_cols = ['TEMPLATE_ID', 'SUBJECT_ID', 'FILE']
        comp_cols = ['TEMPLATE_1', 'TEMPLATE_2']

        self.comp_data = pd.read_csv(filepath_or_buffer=comp_file,
                                     header=None,
                                     # usecols=comp_cols,
                                     dtype=object)
        self.comp_data.columns = comp_cols

        self.template_1 = self.comp_data['TEMPLATE_1'].values.tolist()
        self.template_2 = self.comp_data['TEMPLATE_2'].values.tolist()

        self.meta_data = pd.read_csv(filepath_or_buffer=meta_file,
                                     header=0,
                                     usecols=meta_cols,
                                     dtype=object)

        self.data_len = len(self.comp_data.index)

        self.image_root_folder = image_root_folder

    def __getitem__(self, index):
        template_1 = self.template_1[index]
        template_2 = self.template_2[index]

        t1 = self.meta_data[self.meta_data['TEMPLATE_ID'] == template_1].reset_index()
        t2 = self.meta_data[self.meta_data['TEMPLATE_ID'] == template_2].reset_index()

        list1 = (self.image_root_folder + t1['SUBJECT_ID'] + '/' + t1['FILE']).values.tolist()
        list2 = (self.image_root_folder + t2['SUBJECT_ID'] + '/' + t2['FILE']).values.tolist()

        label1 = t1.loc[0, 'SUBJECT_ID']
        label2 = t2.loc[0, 'SUBJECT_ID']

        return list1, list2, label1 == label2

    def __len__(self):
        return self.data_len
