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

# io1 = r'F:\桌面\P\ctina.xlsx'
# # io2 = r'F:\桌面\究极体模型\cein2.xlsx'
# io3 = r'F:\桌面\P\ctouta.xlsx'
# datax = np.array(pd.read_excel(io1, header=None))
# # datax2 = np.array(pd.read_excel(io2, header=None))
# datay = np.array(pd.read_excel(io3, header=None))
# # datax = np.append(datax1, datax2, axis=0)
# print(datax.shape)
# np.save('ctadatax6.npy',datax)
# np.save('ctadatay6.npy',datay)
#
# io1 = r'F:\桌面\P\ctinb.xlsx'
# # io2 = r'F:\桌面\究极体模型\cein2.xlsx'
# io3 = r'F:\桌面\P\ctoutb.xlsx'
# datax = np.array(pd.read_excel(io1, header=None))
# # datax2 = np.array(pd.read_excel(io2, header=None))
# datay = np.array(pd.read_excel(io3, header=None))
# # datax = np.append(datax1, datax2, axis=0)
# print(datax.shape)
# np.save('ctbdatax6.npy',datax)
# np.save('ctbdatay6.npy',datay)
#
# io1 = r'F:\桌面\P\ctinc.xlsx'
# # io2 = r'F:\桌面\究极体模型\cein2.xlsx'
# io3 = r'F:\桌面\P\ctoutc.xlsx'
# datax = np.array(pd.read_excel(io1, header=None))
# # datax2 = np.array(pd.read_excel(io2, header=None))
# datay = np.array(pd.read_excel(io3, header=None))
# # datax = np.append(datax1, datax2, axis=0)
# print(datax.shape)
# np.save('ctcdatax6.npy',datax)
# np.save('ctcdatay6.npy',datay)



trainData_X = np.load('tdataxa1.npy')[:,0:14]
trainData_Y = np.load('tdataya1.npy')
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
batch_size = 1
# 每批次50样本
# n_inputs = 14  # 输入到单个 LSTM 神经元的特征数
# max_time = trainData.shape[1]  # steps的数量，固定序列长度
n_classes = pre_horizon  # 10个预测量
n_batch = X_train.shape[0] // batch_size   # 计算一共多少批次
print(n_batch)


class LSTM_encoder(nn.Module):

    def __init__(self):
        super(LSTM_encoder, self).__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=256, num_layers=1, batch_first=True)
        self.lstm1 = nn.LSTM(input_size=2, hidden_size=256, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=2, hidden_size=256, num_layers=1, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=2, hidden_size=256, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(256, 128)
        self.fc11 = nn.Linear(256, 128)
        self.fc12 = nn.Linear(256, 128)
        self.fc13 = nn.Linear(256, 128)
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
        dense2 = self.fc12(h2[0])
        dense2 = F.leaky_relu_(dense2)
        result2 = self.fc22(dense2)

        out3, (h3, c3) = self.lstm2(x3)
        dense3 = self.fc13(h3[0])
        dense3 = F.leaky_relu_(dense3)
        result3 = self.fc23(dense3)

        out4, (h4, c4) = self.lstm2(x4)
        dense4 = self.fc11(h4[0])
        dense4 = F.leaky_relu_(dense4)
        result4 = self.fc21(dense4)

        out5, (h5, c5) = self.lstm3(x5)
        dense5 = self.fc12(h5[0])
        dense5 = F.leaky_relu_(dense5)
        result5 = self.fc22(dense5)

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
        self.model = nn.Sequential(nn.Linear(256, 128), nn.LeakyReLU(), nn.Linear(128, 1), nn.LeakyReLU(),
                                   nn.Softmax(dim=1))
        self.layer1 = nn.Linear(256, 64)
        self.layer2 = nn.LeakyReLU(inplace=True)
        self.cell1 = nn.LSTMCell(64, 128)

    def forward(self, x1, x2, x3, x4):
        # h1 = torch.zeors(seq, batch, hidden_state)
        # x1 input, x2 hidden_state, x3 c0
        h2 = torch.unsqueeze(x2, dim=1)
        self_veh = h2.repeat(1, 6, 1)
        x = torch.cat([x1, self_veh], dim=2)
        attention_score = self.model(x)
        attention_score = torch.transpose(attention_score, dim0=2, dim1=1)
        Aggregation1 = torch.matmul(attention_score, x1)
        Aggregation1 = Aggregation1.squeeze(dim=1)
        decoder_input = torch.cat([x4, Aggregation1], dim=1)
        decoder_input = self.layer1(decoder_input)
        xt = self.layer2(decoder_input)
        # h1[batch, hidden]
        H0, C0 = self.cell1(xt, [x2, x3])
        return attention_score, H0, C0

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
        attention_score = torch.transpose(attention_score, dim0=2, dim1=1)
        aggregation = torch.matmul(attention_score, x_social)
        aggregation = aggregation.squeeze(dim=1)
        cc = torch.cat([aggregation, x], dim=1)
        cc1 = self.layer1(cc)
        cc2 = self.layer2(cc1)
        cc3 = self.layer3(cc2)
        return cc, cc3, attention_score

class Personal_out1(nn.Module):

    def __init__(self):
        super(Personal_out1, self).__init__()
        self.layer1 = nn.Linear(256, 256)
        self.layer2 = nn.Linear(256, pre_horizon)
        self.layer3 = nn.LeakyReLU()

    def forward(self,cc2):
        out = self.layer3(self.layer1(cc2))
        out = self.layer2(out)
        return out

class Personal_out2(nn.Module):

    def __init__(self):
        super(Personal_out2, self).__init__()
        self.layer1 = nn.Linear(128, pre_horizon)

    def forward(self,cc2):
        out = self.layer1(cc2)
        return out

class Personal_out3(nn.Module):

    def __init__(self):
        super(Personal_out3, self).__init__()
        self.layer1 = nn.Linear(128, pre_horizon)

    def forward(self,cc2):
        out = self.layer1(cc2)
        return out

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
Personal1 = Personal_out1().to(device)
Personal1.load_state_dict(torch.load('Personal2_b1.pt'))
Personal2 = Personal_out2().to(device)
Personal3 = Personal_out3().to(device)
LsTM = LSTM_encoder().to(device)
LsTM.load_state_dict(torch.load('LsTM_6.2.pt'))
attention = Attention_decoder().to(device)
# attention.load_state_dict(torch.load('attention1.pt'))
Final = Final_out().to(device)
# Final.load_state_dict(torch.load('Final1.pt'))
traffic = Forward_attention().to(device)
traffic.load_state_dict(torch.load('Forward_6.2.pt'))
optimizer = optim.Adam(LsTM.parameters(), lr=0.0001)
# optimizer2 = optim.Adam(traffic.parameters(), lr=0.0001)
optimizer1 = optim.Adam([{'params': LsTM.parameters()},
                         {'params': traffic.parameters()}], lr=0.0001)
optimizera = optim.Adam(Personal1.parameters(), lr=0.0001)
optimizerb = optim.Adam(Personal2.parameters(), lr=0.0001)
optimizerc = optim.Adam(Personal3.parameters(), lr=0.0001)

attentions = 0
losssum, losssum1, losssum2, losssum3, losssum4, losssum5 = 0, 0, 0, 0, 0, 0
with torch.no_grad():
    for batch in range(n_batch):
        X_data = X_train[batch * batch_size:(batch + 1) * batch_size]
        X_data = np.float32(X_data)
        Y_data = Y_train[batch * batch_size:(batch + 1) * batch_size]
        Y_data = np.float32(Y_data)
        # 现在的数据是X[BATCH, TIME_STEPS, FEATURES] 这里因为pytorch和tensorflow对LSTM input的要求不一样
        y_self = torch.from_numpy(Y_data).to(device)
        x_self, x1, x2, x3, x4, x5, x6 = DATA_split(X_data)
        h, c, h1, c1, h2, c2, h3, c3, h4, c4, h5, c5, h6, c6 = LsTM(x_self, x1, x2, x3, x4, x5, x6)
        cc2, _ ,attention= traffic(h, h1, h2, h3, h4, h5, h6)
        attentions = attentions + attention
        out = Personal1(cc2)
        loss1 = F.mse_loss(out, y_self)

        # print(batch+1,loss1,out)

        losssum = losssum + loss1.item()

        out1 = out[:, [0, 1]]
        y_self1 = y_self[:, [0, 1]]
        loss1 = F.mse_loss(out1, y_self1).to(device)
        losssum1 = losssum1 + loss1

        out2 = out[:, [2, 3]]
        y_self2 = y_self[:, [2, 3]]
        loss2 = F.mse_loss(out2, y_self2).to(device)
        losssum2 = losssum2 + loss2

        out3 = out[:, [4, 5]]
        y_self3 = y_self[:, [4, 5]]
        loss3 = F.mse_loss(out3, y_self3).to(device)
        losssum3 = losssum3 + loss3

        out4 = out[:, [6, 7]]
        y_self4 = y_self[:, [6, 7]]
        loss4 = F.mse_loss(out4, y_self4).to(device)
        losssum4 = losssum4 + loss4

        out5 = out[:, [8, 9]]
        y_self5 = y_self[:, [8, 9]]
        loss5 = F.mse_loss(out5, y_self5).to(device)
        losssum5 = losssum5 + loss5
print('attention',attentions/n_batch)
print('loss', losssum/n_batch)
print('loss1',losssum1/n_batch)
print('loss2',losssum2/n_batch)
print('loss3',losssum3/n_batch)
print('loss4',losssum4/n_batch)
print('loss5',losssum5/n_batch)


