import gym
import numpy as np
from pygame import display

from model import Model
from algo import DQN
from my_agent import Agent
from replay_memory import ReplayMemory
from my_wrapper import MyWrapper

LEARN_FREQ = 5  # 训练频率，不需要每一个step都learn，攒一些新增经验后再learn，提高效率
MEMORY_SIZE = 20000  # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 200  # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn
BATCH_SIZE = 32  # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
LEARNING_RATE = 0.001  # 学习率
GAMMA = 0.99  # reward 的衰减因子，一般取 0.9 到 0.999 不等


# 训练一个episode
def run_train_episode(agent, env, rpm):
    total_reward = 0
    obs = env.reset()[0]
    step = 0
    while True:
        #先采样数据预热rpm
        step += 1
        action = agent.sample(obs)  # 采样动作，所有动作都有概率被尝试到
        next_obs, reward, done, _,_ = env.step(action)
        rpm.append((obs, action, reward, next_obs, done))

        # train model
        if (len(rpm) > MEMORY_WARMUP_SIZE) and (step % LEARN_FREQ == 0):
            # s,a,r,s',done
            (batch_obs, batch_action, batch_reward, batch_next_obs,
             batch_done) = rpm.sample(BATCH_SIZE)
            train_loss = agent.learn(batch_obs, batch_action, batch_reward,
                                     batch_next_obs, batch_done)

        total_reward += reward
        obs = next_obs
        if done:
            break
    return total_reward


# 评估 agent, 跑 5 个episode，总reward求平均
def run_evaluate_episodes(agent, env, render=False):
    eval_reward = []
    for i in range(5):
        obs = env.reset()[0]
        episode_reward = 0
        while True:
            action = agent.predict(obs)  # 预测动作，只选最优动作
            obs, reward, done, _,_ = env.step(action)
            episode_reward += reward
            if render:
                display.clear_output(wait=False)
                env.show()
            if done:
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)


def main():
    # CartPole-v0: expected reward > 180
    # MountainCar-v0 : expected reward > -120
    env = MyWrapper()
    obs_dim = env.observation_space.shape[0]  # CartPole-v0: (4,)
    act_dim = env.action_space.n  # CartPole-v0: 2

    rpm = ReplayMemory(MEMORY_SIZE)  # DQN的经验回放池

    # 根据parl框架构建agent
    model = Model(obs_dim=obs_dim, act_dim=act_dim)
    algorithm = DQN(model, gamma=GAMMA, lr=LEARNING_RATE)
    agent = Agent(
        algorithm,
        act_dim=act_dim,
        e_greed=0.1,  # 有一定概率随机选取动作，探索
        e_greed_decrement=1e-6)  # 随着训练逐步收敛，探索的程度慢慢降低

    # 加载模型
    # save_path = './dqn_model.ckpt'
    # agent.restore(save_path)

    # 先往经验池里存一些数据，避免最开始训练的时候样本丰富度不够
    while len(rpm) < MEMORY_WARMUP_SIZE:
        run_train_episode(agent, env, rpm)

    max_episode = 2000

    # start train
    episode = 0
    while episode < max_episode:  # 训练max_episode个回合，test部分不计算入episode数量
        # train part
        model.train()
        for i in range(50):
            total_reward = run_train_episode(agent, env, rpm)
            episode += 1

        # test part       render=True 查看显示效果
        model.eval()
        eval_reward = run_evaluate_episodes(agent, env, render=False)
        print('episode:{}    e_greed:{}   Test reward:{} '.format(
            episode, agent.e_greed, eval_reward))

    # 训练结束，保存模型
    #save_path = './dqn_model.ckpt'
    #agent.save(save_path)


if __name__ == '__main__':
    main()