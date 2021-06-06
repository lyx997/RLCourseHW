# coding=utf-8
import socket  # socket模块
import json
import random
import numpy as np
from collections import deque
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Conv2D, Flatten, concatenate
from keras.optimizers import Adam
from math import floor, sqrt
import tensorflow as tf
import subprocess
import time
import psutil
import pyautogui
import os
import pickle
from multiprocessing import Pool
from keras.backend.tensorflow_backend import set_session
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.distributions import Categorical


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

# set device to cpu or cuda
device = torch.device('cpu')

if(torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

# torch 1.1 don't define nn.Flatten
class Flatten(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


class Critic(nn.Module):

    def __init__(self, num_inputs):
        super(Critic, self).__init__()  # ???
        self.conv1 = nn.Conv2d(num_inputs, 64, (4, 2))
        self.conv2 = nn.Conv2d(64, 64, (4, 2))
        self.conv3 = nn.Conv2d(64, 3, 1)
        self.relu = nn.ReLU()
        self.state1 = Flatten()
        self.state2 = nn.Linear(120, 256)
        self.state3 = nn.Linear(256, 64)
        self.output = nn.Linear(64, 1)
        self.output.weight.data.mul_(0.1)
        self.output.bias.data.mul_(0.0)

    def forward(self, state):
        # input1 = state[0]  input2 = state[1]
        x = self.relu(self.conv1(state[0]))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.state1(x)
        x = torch.cat((x, state[1]), dim=1)
        x = self.relu(self.state2(x))
        x = self.relu(self.state3(x))
        state_values = self.output(x)

        return state_values


class Actor(nn.Module):

    def __init__(self, num_inputs, action_size):
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 64, (4, 2))
        self.conv2 = nn.Conv2d(64, 64, (4, 2))
        self.conv3 = nn.Conv2d(64, 3, 1)
        self.relu = nn.ReLU()
        self.state1 = Flatten()
        self.state2 = nn.Linear(120, 256)
        self.state3 = nn.Linear(256, 64)

        self.action_tensor = nn.Linear(64, action_size)
        self.action_tensor.weight.data.mul_(0.1)
        self.action_tensor.bias.data.mul_(0.0)
        self.action_prob = nn.Softmax(dim=1)

    def forward(self, state):
        # input1 = state[0]  input2 = state[1]
        x = self.relu(self.conv1(state[0]))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.state1(x)
        x = torch.cat((x, state[1]), dim=1)
        x = self.relu(self.state2(x))
        x = self.relu(self.state3(x))

        action_tensor = self.action_tensor(x)
        action_prob = self.action_prob(action_tensor)

        return action_prob


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, action_size):
        super(ActorCritic, self).__init__()
        self.actor = Actor(num_inputs, action_size)
        self.critic = Critic(num_inputs)

    def act(self, state):
        action_prob = self.actor(state)
        dist = Categorical(action_prob)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    def optim_act(self, state):
        action_prob = self.actor(state)
        action = torch.argmax(action_prob, dim=1)

        return action.detach()

    def predict(self, state, action):
        action_prob = self.actor(state)
        action_index = torch.LongTensor([action]).to(device)
        action_logprob = torch.index_select(action_prob, 1, action_index)
        state_values = self.critic(state)

        return action_logprob, state_values


class PPOAgent:

    def __init__(self, num_inputs, state_height, state_width, action_size, lr_critic=0.001, lr_actor=0.0003, gamma=0.9,
                 eps_clip=0.2, K_epochs=80):
        self.state_height = state_height
        self.state_width = state_width
        self.action_size = action_size
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.lr_critic = lr_critic
        self.lr_actor = lr_critic
        self.K_epochs = K_epochs
        self.target_policylearner = ActorCritic(num_inputs, action_size).to(device)
        self.policylearner = ActorCritic(num_inputs, action_size).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policylearner.actor.parameters(), 'lr': lr_actor},
            {'params': self.policylearner.critic.parameters(), 'lr': lr_critic}
        ])

        self.policylearner.load_state_dict(self.target_policylearner.state_dict())

        self.mseloss = nn.MSELoss()
        self.memory1 = deque(maxlen=20000)
        self.memory2 = deque(maxlen=20000)

    def act(self, state):

        with torch.no_grad():
            state_ = torch.FloatTensor(state[0]).to(device)
            pos = torch.Tensor(state[1]).to(device)
            action, action_logprob = self.policylearner.act([state_, pos])

        return action.item(), action_logprob

    def optim_act(self, state):

        with torch.no_grad():
            state_ = torch.FloatTensor(state[0]).to(device)
            pos = torch.Tensor(state[1]).to(device)
            action = self.policylearner.optim_act([state_, pos])

        return action.item()

    def remember1(self, state, action, logprobs, reward, next_state):
        self.memory1.append((state, action, logprobs, reward, next_state))

    def remember2(self, state, action, logprobs, reward, next_state):
        self.memory2.append((state, action, logprobs, reward, next_state))

    def update_policylearner(self):
        self.target_policylearner.load_state_dict(self.policylearner.state_dict())

    def replay(self, batch_size):
        minibatch1 = random.sample(self.memory1, int(batch_size / 2))
        minibatch2 = random.sample(self.memory2, batch_size - int(batch_size / 2))
        minibatch = minibatch1 + minibatch2

        for i in range(self.K_epochs):

            for old_state, old_action, old_logprobs, reward, next_state in minibatch:
                logprobs, state_values = self.target_policylearner.predict(old_state, old_action)

                predict_values = reward + self.gamma * state_values
                # important sampling coefficient
                ratio = torch.exp(logprobs - old_logprobs)
                advantages = predict_values - state_values
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

                loss = - torch.min(surr1, surr2) + 0.5 * self.mseloss(state_values, predict_values)

                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()

    def save(self, checkpoint_path):
        torch.save(self.policylearner.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policylearner.load_state_dict(torch.load(checkpoint_path))
        self.target_policylearner.load_state_dict(torch.load(checkpoint_path))


def connect(ser):
    conn, addr = ser.accept()  # 接受TCP连接，并返回新的套接字与IP地址
    print('Connected by', addr)  # 输出客户端的IP地址
    return conn


def open_ter(loc):
    os.system("gnome-terminal -e 'bash -c \"cd " + loc + " && ./path_planning; exec bash\"'")
    time.sleep(1)
    # return sim


def kill_terminal():
    pids = psutil.pids()
    for pid in pids:
        p = psutil.Process(pid)
        #if p.name() == "gnome-terminal-server":
        if p.name() == "gnome-terminal-server":
            os.kill(pid, 9)


def close_all(sim):
    if sim.poll() is None:
        sim.terminate()
        sim.wait()
    time.sleep(2)
    kill_terminal()

location = "/home/lyx997/clone/Autonomous-Driving/decision-making-CarND/CarND-test/build"
HOST = '127.0.0.1'
PORT = 1234
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 定义socket类型，网络通信，TCP
server.bind((HOST, PORT))  # 套接字绑定的IP与端口
server.listen(1)  # 开始TCP监听

state_height = 45
state_width = 3
action_size = 3
agent = PPOAgent(1, state_height, state_width, action_size)
agent.load("./ppo-result/episode24.h5")
with open('ppo-result/firsttrain/exp1.pkl', 'rb') as exp1:
    agent.memory1 = pickle.load(exp1)
with open('ppo-result/firsttrain/exp2.pkl', 'rb') as exp2:
    agent.memory2 = pickle.load(exp2)

# 开启程序
pool = Pool(processes=2)
result = []
result.append(pool.apply_async(connect, (server,)))
pool.apply_async(open_ter, (location,))
pool.close()
pool.join()
conn = result[0].get()
sim = subprocess.Popen('/home/lyx997/clone/Autonomous-Driving/decision-making-CarND/term3_sim_linux/term3_sim.x86_64')
time.sleep(5)
pyautogui.click(x=1164, y=864, button='left')
time.sleep(10)
pyautogui.click(x=920, y=840, button='left')
data = conn.recv(2000)  # 把接收的数据实例化

while not data:
    try:
        data = conn.recv(2000)
    except Exception as e:
        close_all(sim)
data = bytes.decode(data)
# print(data)
j = json.loads(data)

# 初始化状态信息
# Main car's localization Data
# car_x = j[1]['x']
# car_y = j[1]['y']
car_s = j[1]['s']
car_d = j[1]['d']
car_yaw = j[1]['yaw']
car_speed = j[1]['speed']
# Sensor Fusion Data, a list of all other cars on the same side of the road.
sensor_fusion = j[1]['sensor_fusion']
grid = np.ones((51, 3))
ego_car_lane = int(floor(car_d/4))
grid[31:35, ego_car_lane] = car_speed / 100.0  #4是车长

# sensor_fusion_array = np.array(sensor_fusion)
for i in range(len(sensor_fusion)):
    vx = sensor_fusion[i][3]
    vy = sensor_fusion[i][4]
    s = sensor_fusion[i][5]
    d = sensor_fusion[i][6]
    check_speed = sqrt(vx * vx + vy * vy)
    car_lane = int(floor(d / 4))
    if 0 <= car_lane < 3:
        s_dis = s - car_s
        if -36 < s_dis < 66:
            pers = - int(floor(s_dis / 2.0)) + 30
            grid[pers:pers + 4, car_lane] = - check_speed / 100.0 * 2.237

state = np.zeros((state_height, state_width))
state[:, :] = grid[3:48, :]        #agent视野范围3:48
state = np.reshape(state, [-1, 1, state_height, state_width])
pos = [car_speed / 50, 0, 0]
if ego_car_lane == 0:
    pos = [car_speed / 50, 0, 1]
elif ego_car_lane == 1:
    pos = [car_speed / 50, 1, 1]
elif ego_car_lane == 2:
    pos = [car_speed / 50, 1, 0]
pos = np.reshape(pos, [1, 3])
# print(state)

action = torch.LongTensor([0]).to(device)
mess_out = str(action.item())
mess_out = str.encode(mess_out)
conn.sendall(mess_out)
count = 0
counts = 0
start = time.time()

# 开始训练过程
while True:
    # now = time.time()
    # if (now - start) / 60 > 15:
    #     close_all(sim)
    #     break
    try:
        data = conn.recv(2000)
    except Exception as e:
        pass
    while not data:
        try:
            data = conn.recv(2000)
        except Exception as e:
            pass
    data = bytes.decode(data)
    if data == "over":  # 此次迭代结束
        print('测试结束')
        break
    try:
        j = json.loads(data)
    except Exception as e:
        close_all(sim)
        break

    # *****************在此处编写程序*****************
    state = torch.Tensor(state).to(device)
    pos = torch.Tensor(pos).to(device)
    last_act = action
    last_state = state
    last_pos = pos
    last_lane = ego_car_lane
    # **********************************************

    # Main car's localization Data
    # car_x = j[1]['x']
    # car_y = j[1]['y']
    car_s = j[1]['s']
    car_d = j[1]['d']
    car_yaw = j[1]['yaw']
    car_speed = j[1]['speed']
    print(car_s)
    if car_speed == 0:
        mess_out = str(0)
        mess_out = str.encode(mess_out)
        conn.sendall(mess_out)
        continue
    # Sensor Fusion Data, a list of all other cars on the same side of the road.
    sensor_fusion = j[1]['sensor_fusion']
    ego_car_lane = int(floor(car_d / 4))
    if last_act == 0:
        last_reward = (2 * ((j[3] - 25.0) / 5.0))  # - abs(ego_car_lane - 1))
    else:
        last_reward = (2 * ((j[3] - 25.0) / 5.0)) - 10.0
    if grid[3:31, last_lane].sum() > 27 and last_act != 0:
        last_reward = -30.0
    grid = np.ones((51, 3))
    grid[31:35, ego_car_lane] = car_speed / 100.0
    # sensor_fusion_array = np.array(sensor_fusion)
    for i in range(len(sensor_fusion)):
        vx = sensor_fusion[i][3]
        vy = sensor_fusion[i][4]
        s = sensor_fusion[i][5]
        d = sensor_fusion[i][6]
        check_speed = sqrt(vx * vx + vy * vy)
        car_lane = int(floor(d / 4))
        if 0 <= car_lane < 3:
            s_dis = s - car_s
            if -36 < s_dis < 66:
                pers = - int(floor(s_dis / 2.0)) + 30
                grid[pers:pers + 4, car_lane] = - check_speed / 100.0 * 2.237
        if j[2] < -10:
            last_reward = float(j[2])  # reward -50, -100

    last_reward = last_reward / 10.0
    state = np.zeros((state_height, state_width))
    state[:, :] = grid[3:48, :]
    state = np.reshape(state, [-1, 1, state_height, state_width])
    # print(state)
    pos = [car_speed / 50, 0, 0]
    if ego_car_lane == 0:
        pos = [car_speed / 50, 0, 1]
    elif ego_car_lane == 1:
        pos = [car_speed / 50, 1, 1]
    elif ego_car_lane == 2:
        pos = [car_speed / 50, 1, 0]
    pos = np.reshape(pos, [1, 3])
    print("last_action:{}, last_reward:{:.4}, speed:{:.3}".format(last_act, last_reward, float(car_speed)))

    action, logprobs = agent.act([state, pos])
    #action = agent.optim_act([state, pos])
    mess_out = str(action)
    mess_out = str.encode(mess_out)
    conn.sendall(mess_out)
