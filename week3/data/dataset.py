import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class PilotNetDataset(Dataset):
    def __init__(
        self,
        dataframe,
        image_root,
        training=False,
        side_camera_prob=0.4,
        steering_correction=0.2,
    ):
        """
        dataframe:
            包含center/left/right图像路径和steering标签

        image_root:
            IMG图片所在目录

        training:
            True时启用数据增强和侧摄像头

        side_camera_prob:
            选择左右摄像头的总概率

        steering_correction:
            左右摄像头对应的转角修正量
        """
        super().__init__()

        self.dataframe = dataframe.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.training = training
        self.side_camera_prob = side_camera_prob
        self.steering_correction = steering_correction

    def __len__(self):
        return len(self.dataframe)

    def select_camera(self, row):
        """
        训练时随机选择中心、左、右摄像头。

        左摄像头表示车辆相对靠右，需要向左修正；
        右摄像头表示车辆相对靠左，需要向右修正。

        这里的正负方向需要与数据集定义保持一致。
        Udacity通常正值表示向右转。
        """
        steering = float(row["steering"])
        if not self.training:
            return row["center"], steering

        value = random.random()

        if value < self.side_camera_prob / 2:
            image_path = row["left"]
            steering -= self.steering_correction
        elif value < self.side_camera_prob:
            image_path = row["right"]
            steering += self.steering_correction
        else:
            image_path = row["center"]

        return image_path, steering

    @staticmethod
    def get_file_name(raw_path):
        return str(raw_path).strip().replace("\\", "/").split("/")[-1]

    @staticmethod
    def random_brightness(image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        brightness_scale = random.uniform(0.6, 1.4)
        # hsv原始是uint8，直接乘法可能出现溢出等问题
        hsv = hsv.astype(np.float32)
        hsv[:, :, 2] *= brightness_scale
        # clip对数值做限制，避免转换会uint8后发现数值回绕
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
        
        hsv = hsv.astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    @staticmethod
    def random_translate(image, steering):
        """
        水平平移用于模拟车辆偏离车道中心。

        图像向右移动时，标签也做轻微修正。
        0.002是经验参数，需要通过实验调整。
        """
        height, width = image.shape[:2]
        
        translate_x = random.uniform(-30, 30)
        translate_y = random.uniform(-5, 5)
        
        # 仿射变换矩阵，3个字段分别表示：缩放，旋转，位移。当前仅位移
        transform = np.float32(
            [
                [1, 0, translate_x],
                [0, 1, translate_y],
            ]
        )
        image = cv2.warpAffine(
            image,
            transform,
            (width, height),
            borderMode=cv2.BORDER_REPLICATE,
        )
        steering += translate_x * 0.002
        return image, steering
    
    @staticmethod
    def preprocess(image):
        """
        Udacity模拟器原始图像通常为:
        [160, 320, 3]

        处理后:
        [3, 66, 200]
        """
        # 去掉天空和车辆引擎盖
        image = image[60:-25, :, :]
        
        # 调整成PilotNet输入尺寸
        image = cv2.resize(
            image,
            (200, 66),
            interpolation=cv2.INTER_AREA,
        )
        
        # cv2读取是BGR格式，需要转换成YUV
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2YUV,
        )
        
        # [0, 255] -> [-1, 1]
        image = image.astype(np.float32)
        image = image / 127.5 - 1.0
        
        # 格式转换 [H, W, C] -> [C, H, W]
        image = np.transpose(image, (2, 0, 1))
        
        return torch.from_numpy(image)

    def __getitem__(self, index):
        row = self.dataframe.iloc(index)

        raw_path, steering = self.select_camera(row)
        file_name = self.get_file_name(raw_path)
        image_path = self.image_root / file_name

        image = cv2.imread(str(image_path))

        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        # 训练时对数据做augment，
        if self.training:
            if random.random() < 0.5:
                image = self.random_brightness(image)
            if random.random() < 0.5:
                image, steering = self.random_translate(
                    image,
                    steering,
                )
            
            # 随机水平翻转，水平翻转后，转角符号也要翻转
            if random.random() < 0.5:
                image = cv2.flip(image, 1)
                steering = -steering
        
        image = self.preprocess(image)
        steering = torch.tensor(steering, dtype=torch.float32)

        return {
            "image": image,
            "steering": steering,
            "image_path": str(image_path),
        }
