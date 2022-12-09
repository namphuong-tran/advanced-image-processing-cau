import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from pathlib import PurePosixPath

class TrafficSignData(Dataset):
    base_folder = 'GermanTS'

    def __init__(self, root_dir, train=False, transform=None):
        self.root_dir = root_dir

        self.sub_directory = 'Train' if train else 'Test'
        self.csv_file_name = 'train_data.csv' if train else 'test_data.csv'

        csv_file_path = os.path.join(
            root_dir, self.base_folder, self.sub_directory, self.csv_file_name)
        print(csv_file_path)
        self.csv_data = pd.read_csv(csv_file_path)

        self.transform = transform

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        if(self.sub_directory == 'Train'):
            img_path_str = PurePosixPath(self.csv_data.iloc[idx, 7]).relative_to('Train').as_posix()
        else:
            img_path_str = PurePosixPath(self.csv_data.iloc[idx, 7]).relative_to('Test').as_posix()

        img_path = os.path.join(self.root_dir, self.base_folder, self.sub_directory,
                                img_path_str)
        img = Image.open(img_path)

        classId = self.csv_data.iloc[idx, 6]

        if self.transform is not None:
            img = self.transform(img)

        return img, classId