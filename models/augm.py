import cv2
import os
import numpy as np
import albumentations as A
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
import os

# Создайте папки для сохранения изображений
os.makedirs("./res/Y", exist_ok=True)
os.makedirs("./res/X", exist_ok=True)

class CustomDataset(Dataset):
    def __init__(self, path):
        file_pairs = []
        x_folder_path = os.path.join(path, "X")
        y_folder_path = os.path.join(path, "Y")

        x_files = sorted(os.listdir(x_folder_path))
        y_files = sorted(os.listdir(y_folder_path))
        if len(x_files) != len(y_files):
            print("Количество файлов в папках X и Y не совпадает.")
        else:
            for x_file, y_file in zip(x_files, y_files):
                x_file_path = os.path.join(x_folder_path, x_file)
                y_file_path = os.path.join(y_folder_path, y_file)
                file_pairs.append((x_file_path, y_file_path))
        self.data = file_pairs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pair = self.data[idx]
        # print(pair[0], pair[1])
        x, y = self.custom_transform(pair[0], pair[1])
        return x, y

    def custom_transform(self, x, y):
        x = cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2RGB)
        x = transforms.ToTensor()(x)
        y = cv2.cvtColor(cv2.imread(y), cv2.COLOR_BGR2RGB)
        y = transforms.ToTensor()(y)
        condition = (y[0] == 1) & (y[1] == 1) & (y[2] == 1)
        y = torch.where(condition, torch.tensor(0), torch.tensor(1)).unsqueeze(0).to(torch.uint8)
        return x, y

# Определите аугментации, которые вы хотите применить
transform = A.Compose([
    A.HorizontalFlip(p=0.5),  # Горизонтальное отражение с вероятностью 50%
    A.RandomRotate90(p=0.5),  # Вращение на 90 градусов с вероятностью 50%
    A.RandomBrightnessContrast(p=0.2),  # Случайная яркость и контраст
    # Другие аугментации, которые вы хотите использовать
])

dataset = CustomDataset("../XY")

for i, (X_batch, Y_batch) in enumerate(dataset):
    X_batch = X_batch  # Оставляем тензоры без изменений
    Y_batch = Y_batch

    # Примените аугментации к изображениям X и Y одновременно
    augmented = transform(image=X_batch.detach().cpu().numpy(), mask=Y_batch.detach().cpu().numpy())

    # Получите аугментированные изображения X и Y
    X_image = augmented["image"]
    Y_image = augmented["mask"]

    # Масштабируйте значения в диапазон от 0 до 255 (если необходимо)
    X_image = (X_image * 255).astype(np.uint8)
    Y_image = (Y_image * 255).astype(np.uint8)

    # Сохраните аугментированные изображения X и Y
    if not os.path.isdir("./res"):
        os.makedirs("./res")
    cv2.imwrite(f"./XY/X2/{i}.jpg", X_image)
    cv2.imwrite(f"./XY/Y2/{i}.jpg", Y_image)
