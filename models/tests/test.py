from datetime import timedelta
import cv2
from time import time
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F

SAVING_FRAMES_PER_SECOND = 2

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

def format_timedelta(td):
    result = str(td)
    try:
        result, ms = result.split(".")
    except ValueError:
        return "-" + result + ".00".replace(":", "-")
    ms = int(ms)
    ms = round(ms / 1e4)
    return f"-{result}.{ms:02}".replace(":", "-")

def get_saving_frames_durations(cap, saving_fps):
    s = []
    clip_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    for i in np.arange(0, clip_duration, 1 / saving_fps):
        s.append(i)
    return s

def main1(video_file):
    filename, _ = os.path.splitext(video_file)
    if not os.path.isdir(filename):
        os.mkdir(filename)
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    saving_frames_per_second = min(fps, SAVING_FRAMES_PER_SECOND)
    saving_frames_durations = get_saving_frames_durations(cap, saving_frames_per_second)
    count = 0
    save_count = 0
    while True:
        is_read, frame = cap.read()
        if not is_read:
            break
        frame_duration = count / fps
        try:
            closest_duration = saving_frames_durations[0]
        except IndexError:
            break
        if frame_duration >= closest_duration:
            frame_duration_formatted = format_timedelta(timedelta(seconds=frame_duration))
            saveframe_name = os.path.join(filename, f"frame{frame_duration_formatted}.jpg")
            cv2.imwrite(saveframe_name, frame)
            save_count += 1
            try:
                saving_frames_durations.pop(0)
            except IndexError:
                pass
        count += 1
    print(f"Кол-во кадров: {save_count}")

class CustomDataset(Dataset):
    def __init__(self, path):
        file_pairs = []
        x_folder_path = os.path.join(path, "")

        x_files = sorted(os.listdir(x_folder_path))
        for x_file in x_files:
            x_file_path = os.path.join(x_folder_path, x_file)
            file_pairs.append((x_file_path))
        self.data = file_pairs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pair = self.data[idx]
        x = self.custom_transform(pair)
        return x

    def custom_transform(self, x):
        x = cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2RGB)
        x = transforms.ToTensor()(x)
        return x

def main2(path_list, epsilon):
    dataset = CustomDataset(path_list)
    batch_size = 1
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=12)
    model = UNet(3,1)
    model.load_state_dict(torch.load('/home/denis/code/dust-determination-service/models/save_model/model.pth', map_location='cpu'))
    model.eval()
    with torch.no_grad():
        df = pd.DataFrame(columns=["name", "time", "proc"])
        for i, X_batch in enumerate(dataloader):
            X_batch = X_batch
            Y_pred = model(X_batch)
            Y_pred = (torch.sigmoid(Y_pred) > epsilon).int()
            if not os.path.isdir("./res"):
                os.makedirs("./res")
                os.makedirs("./res/X")
                os.makedirs("./res/Y")
            df.loc[len(df)] = [str(i) + ".jpg", i / SAVING_FRAMES_PER_SECOND, torch.sum(Y_pred[0]).item() / Y_pred[0].shape[1] / Y_pred[0].shape[2] * 100]
            alpha = 0.2
            Y_pred = Y_pred[0].detach().numpy().transpose(1, 2, 0)
            X_batch = X_batch[0].detach().numpy().transpose(1, 2, 0)
            Y_pred = (Y_pred * 255).astype(np.uint8)
            X_batch = (X_batch * 255).astype(np.uint8)
            Y_pred_rgb = cv2.cvtColor(Y_pred, cv2.COLOR_GRAY2RGB)
            X_batch = cv2.cvtColor(X_batch, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(X_batch, 1 - alpha, Y_pred_rgb, alpha, 0)
            cv2.imwrite("./res/Y/" + str(i) + ".jpg", overlay)
            cv2.imwrite("./res/X/" + str(i) + ".jpg", X_batch)
        df.to_csv('data_time.csv')
        
        
def dust_determination(path: str, file_name: str):
    path_list = '/home/denis/code/dust-determination-service/data/video2.mp4'
    tic = time()
    main1(path_list)
    main2(path_list[:-4], 0.7)
    tac = time()
    cap = cv2.VideoCapture(path_list)
    length = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    print(f'Время прогрузки {length} секундного видеоролика на cpu: {tac - tic} секунд')

if __name__ == '__main__':
    dust_determination("", "")