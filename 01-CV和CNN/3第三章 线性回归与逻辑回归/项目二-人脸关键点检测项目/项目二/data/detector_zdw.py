from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
import torchvision

import runpy
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

from data_zdw import get_train_test_set
# from predict import predict

torch.set_default_tensor_type(torch.FloatTensor)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Backbone:
        # in_channel, out_channel, kernel_size, stride, padding
        # block 1
        self.conv1_1 = nn.Conv2d(3, 8, 5, 2, 0)
        # block 2
        self.conv2_1 = nn.Conv2d(8, 64, 3, 1, 0)
        self.conv2_2 = nn.Conv2d(64, 64, 3, 1, 0)
        # block 3
        self.conv3_1 = nn.Conv2d(64, 128, 3, 1, 0)
        self.conv3_2 = nn.Conv2d(128, 128, 3, 1, 0)
        # block 4
        self.conv4_1 = nn.Conv2d(128, 256, 3, 1, 1)
        # points branch
        self.conv4_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.ip1 = nn.Linear(4 * 4 * 256, 4096)
        # self.ip2 = nn.Linear(4096, 4096)
        self.ip3 = nn.Linear(4096, 42)
        # common used
        self.prelu1_1 = nn.PReLU()
        self.prelu2_1 = nn.PReLU()
        self.prelu2_2 = nn.PReLU()
        self.prelu3_1 = nn.PReLU()
        self.prelu3_2 = nn.PReLU()
        self.prelu4_1 = nn.PReLU()
        self.prelu4_2 = nn.PReLU()
        self.preluip1 = nn.PReLU()
        self.preluip2 = nn.PReLU()
        self.ave_pool = nn.AvgPool2d(2, 2, ceil_mode=True)

    def forward(self, x):
        # block 1
        # print('x input shape: ', x.shape)
        x = self.ave_pool(self.prelu1_1(self.conv1_1(x)))
        # print('x after block1 and pool shape should be 64x8x27x27: ', x.shape)     # good
        # block 2
        x = self.prelu2_1(self.conv2_1(x))
        # print('b2: after conv2_1 and prelu shape should be 64x64x25x25: ', x.shape) # good
        x = self.prelu2_2(self.conv2_2(x))
        # print('b2: after conv2_2 and prelu shape should be 64x64x23x23: ', x.shape) # good
        x = self.ave_pool(x)
        # print('x after block2 and pool shape should be 64x64x12x12: ', x.shape)
        # block 3
        x = self.prelu3_1(self.conv3_1(x))
        # print('b3: after conv3_1 and pool shape should be 64x128x10x10: ', x.shape)
        x = self.prelu3_2(self.conv3_2(x))
        # print('b3: after conv3_2 and pool shape should be 64x128x8x8: ', x.shape)
        x = self.ave_pool(x)
        # print('x after block3 and pool shape should be 64x256x4x4: ', x.shape)
        # block 4
        x = self.prelu4_1(self.conv4_1(x))
        # print('x after conv4_1 and pool shape should be 64x256x4x4: ', x.shape)

        # points branch
        ip3 = self.prelu4_2(self.conv4_2(x))
        # print('pts: ip3 after conv4_2 and pool shape should be 64x256x4x4: ', ip3.shape)
        ip3 = ip3.view(-1, 4 * 4 * 256)
        # print('ip3 flatten shape should be 64x4096: ', ip3.shape)
        ip3 = self.preluip1(self.ip1(ip3))
        # print('ip3 after ip1 shape should be 64x4096: ', ip3.shape)
        # ip3 = self.preluip2(self.ip2(ip3))
        # print('ip3 after ip2 shape should be 64x4096: ', ip3.shape)
        ip3 = self.ip3(ip3)
        # print('ip3 after ip3 shape should be 64x42: ', ip3.shape)

        return ip3


def train(args, train_loader, valid_loader, model, criterion, optimizer, device):
    # save model
    if args.save_model:
        if not os.path.exists(args.save_directory):
            os.makedirs(args.save_directory)

    epoch = args.epochs
    pts_criterion = criterion

    train_losses = []
    valid_losses = []

    for epoch_id in range(epoch):
        # monitor training loss
        train_loss = 0.0
        valid_loss = 0.0
        ######################
        # training the model #
        ######################
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            img = batch['image']
            landmark = batch['landmarks']
            # print(f"landmark:{landmark.shape}")

            # ground truth
            input_img = img.to(device)
            target_pts = landmark.to(device)
            # print(f"target_pts:{target_pts.shape}")

            # clear the gradients of all optimized variables
            optimizer.zero_grad()

            # get output
            # print(input_img)
            output_pts = model(input_img)
            # print(f"output_pts:{output_pts.shape}")

            # get loss
            output_pts = output_pts.type(torch.float64)
            # target_pts=target_pts.type(torch.float64)

            loss = pts_criterion(output_pts, target_pts)


            # loss.grad_fn
            # do BP automatically
            loss.backward()
            optimizer.step()

            # show log info
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t pts_loss: {:.6f}'.format(
                        epoch_id,
                        batch_idx * len(img),
                        len(train_loader.dataset),
                        100. * batch_idx / len(train_loader),
                        loss.item(),

                    )
                )

        train_losses.append(loss.item())
        ######################
        # validate the model #
        ######################
        valid_mean_pts_loss = 0.0

        model.eval()  # prep model for evaluation
        with torch.no_grad():
            valid_batch_cnt = 0

            for valid_batch_idx, batch in enumerate(valid_loader):
                valid_batch_cnt += 1
                valid_img = batch['image']
                landmark = batch['landmarks']

                input_img = valid_img.to(device)
                target_pts = landmark.to(device)

                output_pts = model(input_img)
                valid_loss = pts_criterion(output_pts, target_pts)
                valid_mean_pts_loss += valid_loss.item()

            valid_mean_pts_loss /= valid_batch_cnt * 1.0

            valid_losses.append(valid_mean_pts_loss)


            print('Valid: pts_loss: {:.6f}'.format(
                    valid_mean_pts_loss
                )
            )
        print('====================================================')
        # save model
        if args.save_model:
            saved_model_name = os.path.join(args.save_directory, 'detector_epoch' + '_' + str(epoch_id) + '.pt')
            torch.save(model.state_dict(), saved_model_name)
    return train_losses,valid_losses


def test(args,test_loader,model,criterion,device):
    epoch = args.epochs
    test_losses=[]
    test_loss_mean=0.0
    i=0

    saved_model_name = os.path.join(args.save_directory, 'detector_epoch' + '_' + str(49) + '.pt')
    model.load_state_dict(torch.load(saved_model_name))
    model.eval()

    for epoch_id in range(epoch):
        i+=1
        with torch.no_grad():
            test_batch_cnt=0
            for test_batch_idx,batch in enumerate(test_loader):
                test_batch_cnt+=1
                test_img=batch["image"]
                # print(test_img)
                test_landmark=batch["landmarks"]

                input = test_img.to(device)
                target = test_landmark.to(device)

                output=model(input)
                loss=criterion(output,target)
                test_loss_mean+=loss.item()
            test_loss_mean/=test_batch_cnt
            test_losses.append(test_loss_mean)

        print(f'epoch :{i}  test loss: {round(test_loss_mean,3)}')

    return test_losses

def pred(args,test_loader,model,device):
    saved_model_name = os.path.join(args.save_directory, 'detector_epoch' + '_' + str(49) + '.pt')
    model.load_state_dict(torch.load(saved_model_name))
    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            img = batch["image"]
            landmark=batch["landmarks"]
            img = img.to(device)
            # target = test_landmark.to(device)
            print(f"i:{i}")

            output = model(img)
            output=output.cpu().numpy()[0]
            # print(f"output:{output}")
            x=list(map(int,output[0:len(output):2]))
            y=list(map(int,output[1:len(output):2]))
            landmark_generated=list(zip(x,y))

            landmark=landmark.numpy()[0]
            x=list(map(int,landmark[0:len(landmark):2]))
            y=list(map(int,landmark[1:len(landmark):2]))
            landmark_truth=list(zip(x,y))

            img = img.cpu().numpy()[0].transpose(1, 2, 0)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            for landmark_truth, landmark_generated in zip(landmark_truth, landmark_generated):
                cv2.circle(img, tuple(landmark_truth), 2, (0, 0, 255), -1)
                cv2.circle(img, tuple(landmark_generated), 2, (0, 255, 0), -1)

            cv2.imshow(str(i), img)
            key = cv2.waitKey()
            if key == 27:
                exit()
            cv2.destroyAllWindows()

def fintune(args, train_loader, criterion, device):
    model_ft=torchvision.models.resnet18(pretrained =True)
    for param in model_ft.parameters():
        param.requires_grad=False
    in_f=model_ft.fc.in_features
    model_ft.fc=nn.Linear(in_f,42)
    # print(model_ft)
    model_ft.to(device)

    optimizer = optim.SGD(model_ft.parameters(), lr = 0.001,momentum = 0.9)

    # save model
    if args.save_model:
        if not os.path.exists(args.save_directory):
            os.makedirs(args.save_directory)

    epoch = args.epochs
    pts_criterion = criterion
    train_losses = []

    for epoch_id in range(epoch):
        train_loss = 0.0
        model_ft.train()
        for batch_idx, batch in enumerate(train_loader):
            img = batch['image']
            landmark = batch['landmarks']
            input_img = img.to(device)
            target_pts = landmark.to(device)

            optimizer.zero_grad()

            output_pts = model_ft(input_img)
            output_pts = output_pts.type(torch.float64)


            loss = pts_criterion(output_pts, target_pts)
            # do BP automatically
            loss.backward()
            optimizer.step()

            # show log info
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t pts_loss: {:.6f}'.format(
                    epoch_id,
                    batch_idx * len(img),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item(),

                )
                )

        train_losses.append(loss.item())

    return train_losses

def main_test():
    ##设置参数部分，控制整个程序
    parser = argparse.ArgumentParser(description='Detector')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',				
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',		
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='save the current Model')
    parser.add_argument('--save-directory', type=str, default='trained_models',
                        help='learnt models are saving here')
    parser.add_argument('--phase', type=str, default='Finetune',   # Train/train, Predict/predict, Finetune/finetune
                        help='training, predicting or finetuning')
    args = parser.parse_args()
    ###################################################################################
    #第二部分 程序控制代码 设置GPU ，将数据传入GPU，读取数据
    torch.manual_seed(args.seed)
    # For single GPU
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")    # cuda:0
    # For multi GPUs, nothing need to change here
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    print('===> Loading Datasets')
    ########读取数据
    train_set, test_set = get_train_test_set()
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size)
    test_loader=torch.utils.data.DataLoader(test_set)

    print('===> Building Model')
    # For single GPU
    model = Net().to(device)
    ####################################################################
    #第三部分 训练控制代码 设置Loss，设置优化器
    criterion_pts = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)  #momentum=args.momentum
    ####################################################################
    #程序处于什么阶段，a训练 b测试 c迁移 d预测
    if args.phase == 'Train' or args.phase == 'train':
        print('===> Start Training')
        train_losses, valid_losses = train(args, train_loader, valid_loader, model, criterion_pts, optimizer, device)

        fig=plt.figure()
        plt.plot(range(len(train_losses)),train_losses,color="r")
        plt.plot(range(len(valid_losses)), valid_losses,color="b")
        plt.show()

        print('====================================================')
    elif args.phase == 'Test' or args.phase == 'test':
        print('===> Test')
        test_losses=test(args,test_loader,model,criterion_pts,device)

        fig = plt.figure()
        plt.plot(range(len(test_losses)), test_losses, color = "r")
        plt.show()

        print('====================================================')

    elif args.phase == 'Finetune' or args.phase == 'finetune':
        print('===> Finetune')
        train_losses=fintune(args, train_loader, criterion_pts,device)
        fig = plt.figure()
        plt.plot(range(len(train_losses)), train_losses, color = "r")
        plt.show()


    elif args.phase == 'Predict' or args.phase == 'predict':
        print('===> Predict')
        pred(args,test_loader, model, device)


if __name__ == '__main__':
    main_test()










