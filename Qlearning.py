import time
import gym
from Gridworld.CliffWalkingWapper import CliffWalkingWapper
from agent.qlearning_agent import QLearningAgent

def run_episode(env, agent, render=False):
    total_steps = 0  # 记录每个episode走了多少step
    total_reward = 0

    obs = env.reset()[0]   # 重置环境, 重新开一局（即开始新的一个episode）#S0

    while True:
        action =agent.sample(obs)# 根据算法选择一个动作 A0
        next_obs, reward, done, _,_ = env.step(action)  # 与环境进行一个交互#S1,R1
        # 训练 Sarsa 算法
        agent.learn(obs, action, reward, next_obs, done)
        #目标策略:生成at+1
        #行为策略:生成St,at,rt+1,St+1
        #比sarsa更大胆
        obs = next_obs  # 存储上一个观察值
        total_reward += reward
        total_steps += 1  # 计算step数
        if render:
            env.render()  #渲染新的一帧图形
        if done:
            break
    return total_reward, total_steps


def test_episode(env, agent):
    total_reward = 0
    obs = env.reset()[0]
    while True:
        action = agent.predict(obs)  # greedy
        next_obs, reward, done, _,_ = env.step(action)
        total_reward += reward
        obs = next_obs
        time.sleep(0.5)
        env.render()
        if done:
            print('test reward = %.1f' % (total_reward))
            break

def main():
    env = gym.make("CliffWalking-v0")  # 0 up, 1 right, 2 down, 3 left
    env = CliffWalkingWapper(env)

    agent = QLearningAgent(
        obs_n=env.observation_space.n,
        act_n=env.action_space.n,
    )

    is_render = False
    for episode in range(500):
        ep_reward, ep_steps = run_episode(env, agent, is_render)

        # 每隔20个episode渲染一下看看效果
        if episode % 20 == 0:
            is_render = True
            print('Episode %s: steps = %s , reward = %.1f' % (episode, ep_steps,
                                                              ep_reward))
        else:
            is_render = False
    # 训练结束，查看算法效果
    test_episode(env, agent)


if __name__ == "__main__":
    main()