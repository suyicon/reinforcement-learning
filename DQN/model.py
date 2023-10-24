import torch


class Model(torch.nn.Module):
    def __init__(self, obs_dim,act_dim):
        super().__init__()
        hid1_size = 128
        hid2_size = 128
        # 3层全连接网络
        self.relu = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(in_features=obs_dim,out_features=128)
        self.fc2 = torch.nn.Linear(in_features=128,out_features=128)
        self.fc3 = torch.nn.Linear(in_features=128,out_features=act_dim)

    def forward(self, obs):
        h1 = self.fc1(obs)
        h1 = self.relu(h1)
        h2 = self.fc2(h1)
        h2 = self.relu(h2)
        Q = self.fc3(h2)
        return Q