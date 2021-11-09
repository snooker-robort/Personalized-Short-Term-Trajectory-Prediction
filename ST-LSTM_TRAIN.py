import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from matplotlib import pyplot as plt
# from utils import plot_image, plot_curve, one_hot
import pandas as pd
import numpy as np
import pickle
from math import floor
import os
np.random.seed(0)
pre_horizon = 10

# io1 = r'F:\桌面\不同数据结构\ueina.xlsx'
# io3 = r'F:\桌面\不同数据结构\ueouta.xlsx'
# datax = np.array(pd.read_excel(io1, header=None))
# datay = np.array(pd.read_excel(io3, header=None))
# print(datax.shape)
# np.save('cadatax_A.npy',datax)
# np.save('cadatay_A.npy',datay)



trainData_X = np.load('cdatax6.npy')[:,0:14]
trainData_Y = np.load('cdatay6.npy')
print('0',trainData_X.shape)
trainData_X = trainData_X[0:trainData_X.shape[0]:5,:]
trainData_Y = trainData_Y[0:trainData_Y.shape[0]:2,:]
print('1',trainData_X.shape)
trainData_X = trainData_X.reshape(int(trainData_X.shape[0]/6),6,14)
Y_self = trainData_Y.reshape(-1,10)








# valid_set_percentage = 0.8
# _, _, X_train, Y_train = split_valid_set(trainData_X, Y_social , valid_set_percentage)
# print(X_train.shape,Y_train.shape)
X_train = trainData_X
Y_train = Y_self
lstm_encoder_size = 128
lstm_decoder_size = 64
time_steps = 5
batch_size = 256
# 每批次50样本
# n_inputs = 14  # 输入到单个 LSTM 神经元的特征数
# max_time = trainData.shape[1]  # steps的数量，固定序列长度
n_classes = pre_horizon  # 10个预测量
n_batch = X_train.shape[0] // batch_size  # 计算一共多少批次


class LSTM_encoder(nn.Module):

    def __init__(self):
        super(LSTM_encoder, self).__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=32, num_layers=1, batch_first=True)
        self.lstm1 = nn.LSTM(input_size=2, hidden_size=32, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=2, hidden_size=32, num_layers=1, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=2, hidden_size=32, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(32, 128)
        self.fc11 = nn.Linear(32, 128)
        self.fc12 = nn.Linear(32, 128)
        self.fc13 = nn.Linear(32, 128)
        self.fc2 = nn.Linear(128, pre_horizon)
        self.fc21 = nn.Linear(128, pre_horizon)
        self.fc22 = nn.Linear(128, pre_horizon)
        self.fc23 = nn.Linear(128, pre_horizon)

    def forward(self, x, x1, x2, x3, x4, x5, x6):
        out, (h, c) = self.lstm(x)
        dense = self.fc1(h[0])
        dense = F.leaky_relu_(dense)
        result = self.fc2(dense)

        out1, (h1, c1) = self.lstm1(x1)
        dense1 = self.fc11(h1[0])
        dense1 = F.leaky_relu_(dense1)
        result1 = self.fc21(dense1)

        out2, (h2, c2) = self.lstm1(x2)
        dense2 = self.fc11(h2[0])
        dense2 = F.leaky_relu_(dense2)
        result2 = self.fc21(dense2)

        out3, (h3, c3) = self.lstm2(x3)
        dense3 = self.fc12(h3[0])
        dense3 = F.leaky_relu_(dense3)
        result3 = self.fc22(dense3)

        out4, (h4, c4) = self.lstm2(x4)
        dense4 = self.fc12(h4[0])
        dense4 = F.leaky_relu_(dense4)
        result4 = self.fc22(dense4)

        out5, (h5, c5) = self.lstm3(x5)
        dense5 = self.fc13(h5[0])
        dense5 = F.leaky_relu_(dense5)
        result5 = self.fc23(dense5)

        out6, (h6, c6) = self.lstm3(x6)
        dense6 = self.fc13(h6[0])
        dense6 = F.leaky_relu_(dense6)
        result6 = self.fc23(dense6)


        return dense, result, dense1, result1, dense2, result2, dense3, result3, dense4, result4, dense5, result5, \
               dense6, result6


# surronding vehicle attention
class Attention_decoder(nn.Module):

    def __init__(self):
        super(Attention_decoder, self).__init__()
        self.model1 = nn.Sequential(nn.Linear(256, 128), nn.LeakyReLU(), nn.Linear(128, 1), nn.LeakyReLU(),
                                   nn.Softmax(dim=1))
        self.model2 = nn.Sequential(nn.Linear(256, 128), nn.LeakyReLU(), nn.Linear(128, 1), nn.LeakyReLU(),
                                   nn.Softmax(dim=1))
        self.model3 = nn.Sequential(nn.Linear(256, 128), nn.LeakyReLU(), nn.Linear(128, 1), nn.LeakyReLU(),
                                   nn.Softmax(dim=1))
        self.layer11 = nn.Linear(256, 128)
        self.layer12 = nn.Linear(256, 128)
        self.layer13 = nn.Linear(256, 128)
        self.layer2 = nn.LeakyReLU()
        self.layer31 = nn.Linear(128, pre_horizon)
        self.layer32 = nn.Linear(128, pre_horizon)
        self.layer33 = nn.Linear(128, pre_horizon)


    def forward(self, x, x1, x2, x3, x4, x5, x6):
        vocal = x.unsqueeze(dim=1)

        x1_social = x1.unsqueeze(dim=1)
        X1_forward = torch.cat([vocal, x1_social], dim=2) #BAICHSIZE, 1, 256
        attention_score = self.model1(X1_forward)
        attention_score = torch.transpose(attention_score, dim0=2, dim1=1)
        aggregation = torch.matmul(attention_score, x1_social)
        aggregation = aggregation.squeeze(dim=1)
        cc = torch.cat([aggregation, x], dim=1)
        cc1 = self.layer11(cc)
        cc21 = self.layer2(cc1)
        cc31 = self.layer31(cc21)

        x2_social = x2.unsqueeze(dim=1)
        X2_forward = torch.cat([vocal, x2_social], dim=2)  # BAICHSIZE, 1, 256
        attention_score = self.model1(X2_forward)
        attention_score = torch.transpose(attention_score, dim0=2, dim1=1)
        aggregation = torch.matmul(attention_score, x2_social)
        aggregation = aggregation.squeeze(dim=1)
        cc = torch.cat([aggregation, x], dim=1)
        cc1 = self.layer11(cc)
        cc22 = self.layer2(cc1)
        cc32 = self.layer31(cc22)

        x3_social = x3.unsqueeze(dim=1)
        X3_forward = torch.cat([vocal, x3_social], dim=2)  # BAICHSIZE, 1, 256
        attention_score = self.model1(X3_forward)
        attention_score = torch.transpose(attention_score, dim0=2, dim1=1)
        aggregation = torch.matmul(attention_score, x3_social)
        aggregation = aggregation.squeeze(dim=1)
        cc = torch.cat([aggregation, x], dim=1)
        cc1 = self.layer12(cc)
        cc23 = self.layer2(cc1)
        cc33 = self.layer32(cc23)

        x4_social = x4.unsqueeze(dim=1)
        X4_forward = torch.cat([vocal, x4_social], dim=2)  # BAICHSIZE, 1, 256
        attention_score = self.model1(X4_forward)
        attention_score = torch.transpose(attention_score, dim0=2, dim1=1)
        aggregation = torch.matmul(attention_score, x4_social)
        # print('x4_social',x4_social.shape)
        # print('aggregation',aggregation.shape)

        aggregation = aggregation.squeeze(dim=1)
        cc = torch.cat([aggregation, x], dim=1)
        cc1 = self.layer12(cc)
        cc24 = self.layer2(cc1)
        cc34 = self.layer32(cc24)

        x5_social = x5.unsqueeze(dim=1)
        X5_forward = torch.cat([vocal, x5_social], dim=2)  # BAICHSIZE, 1, 256
        attention_score = self.model1(X5_forward)
        attention_score = torch.transpose(attention_score, dim0=2, dim1=1)
        aggregation = torch.matmul(attention_score, x5_social)
        aggregation = aggregation.squeeze(dim=1)
        cc = torch.cat([aggregation, x], dim=1)
        cc1 = self.layer13(cc)
        cc25 = self.layer2(cc1)
        cc35 = self.layer33(cc25)

        x6_social = x6.unsqueeze(dim=1)
        X6_forward = torch.cat([vocal, x6_social], dim=2)  # BAICHSIZE, 1, 256
        attention_score = self.model1(X6_forward)
        attention_score = torch.transpose(attention_score, dim0=2, dim1=1)
        aggregation = torch.matmul(attention_score, x6_social)
        aggregation = aggregation.squeeze(dim=1)
        cc = torch.cat([aggregation, x], dim=1)
        cc1 = self.layer13(cc)
        cc26 = self.layer2(cc1)
        cc36 = self.layer33(cc26)


        return cc21, cc31, cc22, cc32, cc23, cc33, cc24, cc34, cc25, cc35, cc26, cc36


class Forward_attention(nn.Module):

    def __init__(self):
        super(Forward_attention, self).__init__()
        self.model = nn.Sequential(nn.Linear(256, 128), nn.LeakyReLU(), nn.Linear(128, 1), nn.LeakyReLU(),
                                   nn.Softmax(dim=1))
        self.layer1 = nn.Linear(256, 128)
        self.layer2 = nn.LeakyReLU()
        self.layer3 = nn.Linear(128, pre_horizon)

    def forward(self, x, x1, x2, x3, x4, x5, x6):
        # x=567
        x_social = torch.stack([x1, x2, x3, x4, x5, x6], dim=1)
        vocal = x.unsqueeze(dim=1)
        vocal = vocal.repeat(1, 6, 1)
        X_forward = torch.cat([vocal, x_social], dim=2)
        attention_score = self.model(X_forward)
        # print('attention_score1',attention_score)
        attention_score = torch.transpose(attention_score, dim0=2, dim1=1)
        # print('attention_score2',attention_score)
        aggregation = torch.matmul(attention_score, x_social)
        # print('aggregation',aggregation)
        # print(aggregation.shape)
        aggregation = aggregation.squeeze(dim=1)
        # print(aggregation.shape)
        cc = torch.cat([aggregation, x], dim=1)
        # print('cc',cc)
        cc1 = self.layer1(cc)
        cc2 = self.layer2(cc1)
        cc3 = self.layer3(cc2)
        return cc2, cc3


# forward traffic attention


class Final_out(nn.Module):

    def __init__(self):
        super(Final_out, self).__init__()
        self.model = nn.Sequential(nn.Linear(128, 64), nn.LeakyReLU(), nn.Linear(64, 2))

    def forward(self, x):
        # x[batch, 7, 128]
        output = self.model(x)
        return output

def DATA_split(X):
    x_self = torch.from_numpy(X[:, :, 0:2]).to(device)
    x1 = torch.from_numpy(X[:, :, 2:4]).to(device)  # 前车
    x2 = torch.from_numpy(X[:, :, 4:6]).to(device)  # 后车
    x3 = torch.from_numpy(X[:, :, 6:8]).to(device)  # 左前车
    x4 = torch.from_numpy(X[:, :, 8:10]).to(device)  # 左后车
    x5 = torch.from_numpy(X[:, :, 10:12]).to(device)  # 右前车
    x6 = torch.from_numpy(X[:, :, 12:14]).to(device)  # 右后车
    return x_self, x1, x2, x3, x4, x5, x6



device = torch.device('cuda:0')
LsTM = LSTM_encoder().to(device)
LsTM.load_state_dict(torch.load('ST_LsTM3.pt'))
attention = Attention_decoder().to(device)
attention.load_state_dict(torch.load('ST_attention3.pt'))
Final = Final_out().to(device)
# Final.load_state_dict(torch.load('Final1.pt'))
traffic = Forward_attention().to(device)
traffic.load_state_dict(torch.load('ST_Forward3.pt'))
optimizer = optim.Adam(LsTM.parameters(), lr=0.0005)
optimizer1 = optim.Adam([{'params': LsTM.parameters()},
                         {'params': attention.parameters()}],lr=0.0001)
optimizer2 = optim.Adam([{'params': LsTM.parameters()},
                         {'params': attention.parameters()},
                         {'params': traffic.parameters()}],lr=0.00005)

depoch = -1
for epoch in range(20):
    if epoch % 5 == 0:
        torch.save(LsTM.state_dict(), 'ST_LsTM3.pt')
        torch.save(attention.state_dict(), 'ST_attention3.pt')
        torch.save(traffic.state_dict(), 'ST_Forward3.pt')
    for batch in range(n_batch):
        X_data = X_train[batch * batch_size:(batch + 1) * batch_size]
        X_data = np.float32(X_data)
        Y_data = Y_train[batch * batch_size:(batch + 1) * batch_size]
        Y_data = np.float32(Y_data)
        # 现在的数据是X[BATCH, TIME_STEPS, FEATURES] 这里因为pytorch和tensorflow对LSTM input的要求不一样
        y_self = torch.from_numpy(Y_data).to(device)
        x_self, x1, x2, x3, x4, x5, x6 = DATA_split(X_data)
        h, c, h1, c1,  h2, c2, h3, c3, h4, c4, h5, c5, h6, c6 = LsTM(x_self, x1, x2, x3, x4, x5, x6)

        if epoch <= -3:
            prediction = torch.stack([c, c1, c2, c3, c4, c5, c6], dim=1)
            prediction = prediction.squeeze(dim=0)
            loss = F.mse_loss(prediction, Y_data)
            print('LsTM', loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        elif (epoch < -7) & (epoch > 0):
            hv1, cv1, hv2, cv2, hv3, cv3, hv4, cv4, hv5, cv5, hv6, cv6 = attention(h, h1, h2, h3, h4, h5, h6)
            prediction = torch.stack([ cv1, cv2, cv3, cv4, cv5, cv6], dim=1).reshape(128,60)
            out = (y_self.unsqueeze(dim=1)).repeat(1, 6, 1).reshape(-1,60)
            loss = F.mse_loss(prediction, out)
            print('attention', loss)
            optimizer1.zero_grad()
            loss.backward()
            optimizer1.step()
        else:
            hv1, cv1, hv2, cv2, hv3, cv3, hv4, cv4, hv5, cv5, hv6, cv6 = attention(h, h1, h2, h3, h4, h5, h6)
            _, out = traffic(h,hv1,hv2,hv3,hv4,hv5,hv6)
            prediction = y_self
            loss = F.mse_loss(prediction, out)
            print('optimizer2', loss)
            if batch + 1 == n_batch:
                print('optimizer2', loss)
            optimizer2.zero_grad()
            loss.backward()
            optimizer2.step()
torch.save(LsTM.state_dict(), 'ST_LsTM3.pt')
torch.save(traffic.state_dict(), 'ST_Forward3.pt')
torch.save(attention.state_dict(), 'ST_attention3.pt')
