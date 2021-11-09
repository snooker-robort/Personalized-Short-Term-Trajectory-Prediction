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


pre_horizon = 10

# io1 = r'C:\Users\斯诺克机器人\Desktop\GA_LSTM2\test_input1.xlsx'
# io2 = r'C:\Users\斯诺克机器人\Desktop\GA_LSTM2\test_output1.xlsx'
# datax = np.array(pd.read_excel(io1, header=None))
# datay = np.array(pd.read_excel(io2, header=None))
# print(datax.shape)
# np.save('tdatax1.npy',datax)
# np.save('tdatay1.npy',datay)

# io1 = r'C:\Users\斯诺克机器人\Desktop\GA_LSTM2\test_inputa1.xlsx'
# io2 = r'C:\Users\斯诺克机器人\Desktop\GA_LSTM2\test_outputa1.xlsx'
# datax = np.array(pd.read_excel(io1, header=None))
# datay = np.array(pd.read_excel(io2, header=None))
# np.save('tdataxa1.npy',datax)
# np.save('tdataya1.npy',datay)
# io1 = r'C:\Users\斯诺克机器人\Desktop\GA_LSTM2\test_inputb1.xlsx'
# io2 = r'C:\Users\斯诺克机器人\Desktop\GA_LSTM2\test_outputb1.xlsx'
# datax = np.array(pd.read_excel(io1, header=None))
# datay = np.array(pd.read_excel(io2, header=None))
# np.save('tdataxb1.npy',datax)
# np.save('tdatayb1.npy',datay)
# io1 = r'C:\Users\斯诺克机器人\Desktop\GA_LSTM2\test_inputc1.xlsx'
# io2 = r'C:\Users\斯诺克机器人\Desktop\GA_LSTM2\test_outputc1.xlsx'
# datax = np.array(pd.read_excel(io1, header=None))
# datay = np.array(pd.read_excel(io2, header=None))
# np.save('tdataxc1.npy',datax)
# np.save('tdatayc1.npy',datay)

##a类测试
trainData_X = (np.load('tdataxc1.npy')[:,0:14]).astype('float32')
trainData_Y = np.load('tdatayc1.npy').astype('float32')



# trainData_X = (np.load('ctdatax_A.npy')[:,0:14]).astype('float32')
# trainData_Y = np.load('ctdatay_A.npy').astype('float32')
print('0',trainData_X.shape)
trainData_X = trainData_X[0:trainData_X.shape[0]:5,:]
trainData_Y = trainData_Y[0:trainData_Y.shape[0]:2,:]
print('1',trainData_X.shape)
trainData_X = trainData_X.reshape(int(trainData_X.shape[0]/6),6,14)
Y_self = trainData_Y.reshape(-1,10)

X_train = trainData_X
Y_train = Y_self
lstm_encoder_size = 128
lstm_decoder_size = 64
time_steps = 5
batch_size = 1
n_classes = pre_horizon  # 10个预测量
n_batch = X_train.shape[0] // batch_size  # 计算一共多少批次


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
        self_veh = h2.repeat(1, 3, 1)
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
        return cc2, cc3, attention_score

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

x_valid_self, x_valid1, x_valid2, x_valid3, x_valid4, x_valid5, x_valid6 = DATA_split(X_train)
print('x_valid_self',x_valid_self.shape)

n = int(Y_train.shape[0]/batch_size)
losssum, losssum1, losssum2, losssum3, losssum4, losssum5 = 0, 0, 0, 0, 0, 0
attentions=0
with torch.no_grad():
    for i in range(0,n):
        hv, cv, hv1, cv1, hv2, cv2, hv3, cv3, hv4, cv4, hv5, cv5, hv6, cv6 = LsTM(x_valid_self[i*batch_size:(i+1)*batch_size], x_valid1[i*batch_size:(i+1)*batch_size], x_valid2[i*batch_size:(i+1)*batch_size],
                                                                                  x_valid3[i*batch_size:(i+1)*batch_size], x_valid4[i*batch_size:(i+1)*batch_size], x_valid5[i*batch_size:(i+1)*batch_size],
                                                                                  x_valid6[i*batch_size:(i+1)*batch_size])
        _, out,attention = traffic(hv, hv1, hv2, hv3, hv4, hv5, hv6)
        attentions += attention
        y_self = torch.from_numpy(Y_train[batch_size*i:batch_size*(i+1),:]).to(device)
        loss0 = F.mse_loss(out, y_self).to(device)
        losssum = losssum+loss0.item()

        out1 = out[:,[0,1]]
        y_self1 = y_self[:,[0,1]]
        loss1 = F.mse_loss(out1, y_self1).to(device)
        losssum1 = losssum1 + loss1

        out2 = out[:, [2,3]]
        y_self2 = y_self[:, [2,3]]
        loss2 = F.mse_loss(out2, y_self2).to(device)
        losssum2 = losssum2 + loss2

        out3 = out[:, [4,5]]
        y_self3 = y_self[:, [4,5]]
        loss3 = F.mse_loss(out3, y_self3).to(device)
        losssum3 = losssum3 + loss3

        out4 = out[:, [6,7]]
        y_self4 = y_self[:, [6,7]]
        loss4 = F.mse_loss(out4, y_self4).to(device)
        losssum4 = losssum4 + loss4

        out5 = out[:, [8,9]]
        y_self5 = y_self[:, [8,9]]
        loss5 = F.mse_loss(out5, y_self5).to(device)
        losssum5 = losssum5 + loss5
print('attention',attentions/n)
print('loss',losssum/n)
print('loss1',losssum1/n)
print('loss2',losssum2/n)
print('loss3',losssum3/n)
print('loss4',losssum4/n)
print('loss5',losssum5/n)



