import numpy as np
import cv2,os

import torch
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import matplotlib.pyplot as plt


train_boarder = 112


def channel_norm(img):
    # img: ndarray, float32
    global mean,std
    mean = np.mean(img)
    std = np.std(img)
    pixels = (img - mean) / (std + 0.0000001)
    return pixels


def parse_line(line):
    line_parts = line.strip().split()
    img_name = line_parts[0]
    rect = list(map(int, list(map(float, line_parts[1:5]))))
    landmarks = list(map(float, line_parts[5: len(line_parts)]))
    return img_name, rect, landmarks


class Normalize(object):
    #第三部分 关于归一化和张量转换
    """
        Resieze to train_boarder x train_boarder. Here we use 112 x 112
        Then do channel normalization: (image - mean) / std_variation
    """
    def __call__(self, sample):
        #simple为我们的图像， landmarks为脸部关键点
        image, landmarks = sample['image'], sample['landmarks']
        image_resize = np.asarray(
                            cv2.resize(image,(train_boarder, train_boarder), Image.BILINEAR),
                            dtype=np.float32)       # Image.ANTIALIAS)
        image = channel_norm(image_resize)
        return {'image': image,
                'landmarks': landmarks
                }

class ToTensor(object):
    """
        Convert ndarrays in sample to Tensors.
        Tensors channel sequence: N x C x H x W
    """
    def __call__(self, sample):
        #转换维度
        image, landmarks = sample['image'], sample['landmarks']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose(2, 0, 1)
        # image = np.expand_dims(image, axis=0)
        landmarks = landmarks.astype("float").reshape(-1, 1).T.squeeze(0)
        # print(landmarks.shape)
        # print(f"landmarks:{landmarks}")
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}


class FaceLandmarksDataset(Dataset):
    #将变换作用与数据
    # Face Landmarks Dataset
    def __init__(self, src_lines, transform):  #类的初始化
        '''
        :param src_lines: src_lines
        :param train: whether we are training or not
        :param transform: data transform
        '''
        self.lines = src_lines
        self.transform = transform

    def __len__(self): #多少数据
        # print(f"data_length:{len(self.lines)}")
        return len(self.lines)

    def __getitem__(self, idx): #对每批数据的变化
        img_name, rect, landmarks = parse_line(self.lines[idx])
        # image
        # img = Image.open(img_name).convert('L')
        # img_crop = img.crop(tuple(rect))

        img_file = os.path.join("face", img_name)
        img=cv2.imread(img_file,1)
        # print(img.shape)

        landmarks = np.array(landmarks)
        landmarks=landmarks.astype("float").reshape(-1,2)

        # you should let your landmarks fit to the train_boarder(112)
        # please complete your code under this blank
        # your code:
        crop_height,crop_width,_=img.shape
        # print(f"image_shape:{img.shape}")

        height_ratio=train_boarder/crop_height
        width_ratio=train_boarder/crop_width

        ratio=list((width_ratio,height_ratio))
        # print(f"ratio:{ratio}")
        landmarks=landmarks*ratio

        # print(f"landmarks:{landmarks}")

        sample = {'image': img, 'landmarks': landmarks}
        sample = self.transform(sample)
        return sample


def load_data(phase):
    #第二部分
    data_file = phase + '.txt'
    file=os.path.join("face",data_file)
    # print("file",file)
    with open(file) as f:
        lines = f.readlines()
        # print(f"lines:{lines[0]}")
    if phase == 'Train' or phase == 'train':
        tsfm = transforms.Compose([
            Normalize(),                # do channel normalization
            ToTensor()]                 # convert to torch type: NxCxHxW
        )
    else:
        tsfm = transforms.Compose([
            Normalize(),
            ToTensor()
        ])
    data_set = FaceLandmarksDataset(lines, tsfm)
    # print(type(data_set))
    return data_set


def get_train_test_set():
    #第一部分 根源
    train_set = load_data('train')
    valid_set = load_data('test')
    return train_set, valid_set

if __name__ == '__main__':
    train_set = load_data('train')

    for i in range(0, len(train_set)):
        # print(train_set[i])
        sample = train_set[i]
        img = sample['image']

        # image = img.numpy().squeeze(0)
        image = img.numpy()
        image = np.transpose(image, (1, 2, 0))
        # image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

        # image=np.dot(image,(std + 0.0000001))+mean
        ## 请画出人脸crop以及对应的landmarks
        # please complete your code under this blank
        landmarks = sample['landmarks'].numpy()
        # landmarks=landmarks.astype("float").reshape(-1,2)
        #
        # plt.imshow(image)
        # plt.scatter(landmarks[:,0], landmarks[:,1], s = 30, marker = ".", c = 'g')
        # plt.show()

        ##cv2 to show

        x = list(map(int, landmarks[0:len(landmarks):2]))
        y = list(map(int, landmarks[1:len(landmarks):2]))
        landmark = list(zip(x, y))
        # print(landmark)
        for i in range(len(landmark)):
            cv2.circle(image,landmark[i], 2, (0, 0, 255), -1)
        cv2.imshow("face",image)
        key = cv2.waitKey()
        if key == 27:
            exit(0)
        cv2.destroyAllWindows()






