import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_video
import sys
import os
from PIL import Image

class Dataset_Template(Dataset):
    def __init__(self, csv_file, transform=None, addpath=None):
        # CSVファイルからデータの読み込み
        self.file_list, self.labels = self.load_csv(csv_file)
        # 前処理の定義
        self.transform = transform
        # 画像ファイルのパスに必要であれば定義
        self.add_path = addpath

    # データセットのサイズを返す関数
    def __len__(self):
        return len(self.data_frame)

    # csvをロードする関数
    def load_csv(self, csv_file):
        file_list = []
        labels = []
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                file_list.append(row[0])
                labels.append(list(map(float, row[1:])))
        return file_list, labels

    def __getitem__(self, idx):
        # ファイルのパスを取得
        file_path = os.path.join(self.add_path, self.data_frame.iloc[idx, 0])

        # 画像を読み込み
        image = Image.open(file_path)

        # 前処理が定義されていれば適用
        if self.transform:
            image = self.transform(image)

        # ラベルを取得
        label = self.labels[idx]

        return image, label