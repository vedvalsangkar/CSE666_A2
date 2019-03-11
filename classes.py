import pandas as pd
import numpy as np
import csv

# import torchvision.transforms as transforms
# import PIL.Image as Image

from torchvision import transforms
from PIL import Image
from torch.utils.data.dataset import Dataset


class A2TestDataSet(Dataset):

    def __init__(self, csv_file, image_root_folder='IJBA_images/'):
        # split_pre = "IJBA_sets/split"
        # t_pre = "/train_"
        # # comp = "/verify_comparisons_"
        # # meta = "/verify_metadata_"
        # ext = ".csv"

        # col_names = ['TEMPLATE_ID', 'SUBJECT_ID', 'FILE', 'MEDIA_ID', 'SIGHTING_ID']
        col_names = ['TEMPLATE_ID', 'SUBJECT_ID', 'FILE']

        self.to_tensor = transforms.ToTensor()
        self.data_info = pd.read_csv(filepath_or_buffer=csv_file,
                                     header=0,
                                     usecols=col_names,
                                     dtype=object)

        # self.file_names = pd.concat([self.data_info['SUBJECT_ID'], '/' + self.data_info['FILE']], axis=1)
        # self.file_names = self.data_info['SUBJECT_ID'] + '/' + self.data_info['FILE']
        # self.images = np.asarray(image_root_folder + self.file_names)

        self.images = np.asarray(image_root_folder + self.data_info['SUBJECT_ID'] + '/' + self.data_info['FILE'])

        self.data_len = len(self.data_info.index)

        self.labels = np.asarray(self.data_info['SUBJECT_ID'])

    def __getitem__(self, index):
        # Open image
        image_name = self.images[index]
        image_label = self.labels[index]
        try:
            image = Image.open(image_name)

            # Open image
            # image = Image.open(self.images[index])

            tensor_img = self.to_tensor(image)

        # Get label(class) of the image based on the cropped pandas column
            return tensor_img, image_label
        except:
            print("Unable to access image", image_name, "with index: {}".format(index))

            return image_name, image_label

    def __len__(self):
        return self.data_len

    def clean(self, split):
        # image_label = self.labels[index]
        del_list = []
        for i in range(self.data_len):
            try:
                image = Image.open(self.images[i])
            except:
                print("Unable to access image", self.images[i], "with index: {}. Deleting.".format(i))
                del_list.append(i)

        # pd.DataFrame(data=np.delete(self.images, del_list)).to_csv("IJBA_sets/split"+str(split)+"/train_"+str(split)+"_clean.csv")
        self.data_info.drop(labels=self.data_info.index[del_list],
                            inplace=True)
        self.data_info.to_csv(path_or_buf="IJBA_sets/split"+str(split)+"/train_"+str(split)+"_clean.csv",
                              index=False)


