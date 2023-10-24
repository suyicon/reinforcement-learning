import copy
import torch

class DQN:
    def __init__(self,model,gamma=None,lr=None):
        self.model = model
        self.target_model = copy.deepcopy(model)
        self.gamma = gamma
        self.lr = lr
        self.loss = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(params=model.parameters(),lr=self.lr)

    def sync_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def predict(self,obs):
        Q_list = self.model(obs)
        return Q_list

    def learn(self,obs,action,reward,next_obs,terminal):
        #main network
        pred_values = self.model(obs)
        action_dim = pred_values.shape[-1]
        action = torch.squeeze(action, axis=-1)#[32,1]->[32]
        action_onehot = torch.eye(action_dim)[action]#[32,2]
        # 将action转onehot向量，比如：3 => [0,0,0,1,0]
        # 下面一行是逐元素相乘，拿到action对应的 Q(s,a)
        # 比如：pred_value = [[2.3, 5.7, 1.2, 3.9, 1.4]], action_onehot = [[0,0,0,1,0]]
        pred_values = pred_values * action_onehot#[32,2]
        #  ==> pred_value = [[3.9]]
        pred_value = torch.sum(pred_values,dim=1,keepdim=True)#[32,1]

        #target network
        with torch.no_grad():
            max_v = self.target_model(next_obs).max(dim=1,keepdim=True)[0]
            target = reward + (1-terminal)*self.gamma*max_v
        loss = self.loss(target, pred_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss



