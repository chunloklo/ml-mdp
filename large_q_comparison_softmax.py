from maze_generator import MazeEnv, read_maze
from value_iteration import value_iteration
from policy_iteration import policy_improvement
import numpy as np
from mdp_graph import graph_value_policy
import matplotlib.pyplot as plt
from time import time
from qlearn import QLearn
import time

maze_shape = (32, 32)
maze_file = 'maze/mazeLarge.png'
# p = 0.1
qlearn = QLearn(num_states=32 * 32, num_actions=4, alpha=0.2, gamma=0.99, epsilon=0.1, softmax=True)

env = MazeEnv(maze_file=maze_file)

total_reward_hist = []
cum_total_reward = 0
q_start_hist = []
episodes = 10000
for e in range(episodes):
    done = False
    obs = env.reset()
    if (e % 100 == 0):
        print("Episode {}".format(e))

    gamma = 0.9
    gamma_pow = 1
    total_reward = 0

    q_start_hist.append(np.max(qlearn.q[obs]))
    while not done:
        # obs = env.reset()
        # action = 1
        # new_obs, reward, done, _ = env.step(action)
        # print(obs, action, reward, new_obs)
        # time.sleep(4)
        # print(obs)
        # print(qlearn.q[obs])
        action = qlearn.chooseAction(obs)
        new_obs, reward, done, _ = env.step(action)
        total_reward += gamma_pow * reward
        gamma_pow *= gamma

        qlearn.learn(obs, action, reward, new_obs, done)
        # if (action == 1):
        # print(obs, action, reward, new_obs)
        # env.render()
        # time.sleep(0.01)
        obs = new_obs

    cum_total_reward += total_reward
    total_reward_hist.append(cum_total_reward)

q = qlearn.q
policy = np.argmax(q, axis=1)

value = np.max(q, axis=1)
policy = policy.reshape(maze_shape)
value = value.reshape(maze_shape)
maze = read_maze(maze_file)
im = graph_value_policy(value, policy, maze)
# plt.colorbar()

# plt.figure()
# plt.plot(range(episodes), reward_hist)

timestr = time.strftime("%Y%m%d-%H%M%S")

q1 = q
q_start_hist1 = q_start_hist
total_reward_hist1 = total_reward_hist


plt.figure()
plt.suptitle("Max Q value of Start")
plt.plot(range(episodes), q_start_hist)

plt.figure()
plt.suptitle("Total Reward History")
plt.plot(range(episodes), total_reward_hist)
plt.show()

# np.save("data/{}-small-q-epsilon-p1.npy".format(timestr), q)
# np.save("data/{}-small-q_start_hist-epsilon-p1.npy".format(timestr), q_start_hist)
# np.save("data/{}-small-total_reward_hist-epsilon-p1.npy".format(timestr), total_reward_hist)
# plt.show()

np.save("data/{}-large-q-softmax-100.npy".format(timestr), q1)
np.save("data/{}-large-q_start_hist-softmax-100.npy".format(timestr), q_start_hist1)
np.save("data/{}-large-total_reward_hist-softmax-100.npy".format(timestr), total_reward_hist1)

plt.show()


qlearn = QLearn(num_states=32 * 32, num_actions=4, alpha=0.2, gamma=0.9, epsilon=0.05, softmax=True)

# env = WindyMazeEnv(maze_file=maze_file, wind_prob=p)

total_reward_hist = []
cum_total_reward = 0
q_start_hist = []
for e in range(episodes):
    done = False
    obs = env.reset()
    if (e % 1000 == 0):
        print("Episode {}".format(e))

    gamma = 0.9
    gamma_pow = 1
    total_reward = 0

    q_start_hist.append(np.max(qlearn.q[obs]))
    while not done:
        # obs = env.reset()
        # action = 1
        # new_obs, reward, done, _ = env.step(action)
        # print(obs, action, reward, new_obs)
        # time.sleep(4)
        # print(obs)
        # print(qlearn.q[obs])
        action = qlearn.chooseAction(obs)
        new_obs, reward, done, _ = env.step(action)
        total_reward += gamma_pow * reward
        gamma_pow *= gamma

        qlearn.learn(obs, action, reward, new_obs, done)
        # if (action == 1):
        # print(obs, action, reward, new_obs)
        # env.render()
        # time.sleep(0.01)
        obs = new_obs

    cum_total_reward += total_reward
    total_reward_hist.append(cum_total_reward)

q = qlearn.q
policy = np.argmax(q, axis=1)

value = np.max(q, axis=1)
policy = policy.reshape(maze_shape)
value = value.reshape(maze_shape)
maze = read_maze(maze_file)
im = graph_value_policy(value, policy, maze)

# plt.figure()
# plt.plot(range(episodes), reward_hist)

timestr = time.strftime("%Y%m%d-%H%M%S")

q2 = q
q_start_hist2 = q_start_hist
total_reward_hist2 = total_reward_hist


plt.figure()
plt.suptitle("Max Q value of Start")
plt.plot(range(episodes), q_start_hist)

plt.figure()
plt.suptitle("Total Reward History")
plt.plot(range(episodes), total_reward_hist)


np.save("data/{}-small-q-epsilon-p05.npy".format(timestr), q)
np.save("data/{}-small-q_start_hist-epsilon-p05.npy".format(timestr), q_start_hist)
np.save("data/{}-small-total_reward_hist-epsilon-p05.npy".format(timestr), total_reward_hist)

plt.show()




qlearn = QLearn(num_states=32 * 32, num_actions=4, alpha=0.2, gamma=0.9, epsilon=0.2)

# env = WindyMazeEnv(maze_file=maze_file, wind_prob=p)

total_reward_hist = []
cum_total_reward = 0
q_start_hist = []
for e in range(episodes):
    done = False
    obs = env.reset()
    if (e % 1000 == 0):
        print("Episode {}".format(e))

    gamma = 0.9
    gamma_pow = 1
    total_reward = 0

    q_start_hist.append(np.max(qlearn.q[obs]))
    while not done:
        # obs = env.reset()
        # action = 1
        # new_obs, reward, done, _ = env.step(action)
        # print(obs, action, reward, new_obs)
        # time.sleep(4)
        # print(obs)
        # print(qlearn.q[obs])
        action = qlearn.chooseAction(obs)
        new_obs, reward, done, _ = env.step(action)
        total_reward += gamma_pow * reward
        gamma_pow *= gamma

        qlearn.learn(obs, action, reward, new_obs, done)
        # if (action == 1):
        # print(obs, action, reward, new_obs)
        # env.render()
        # time.sleep(0.01)
        obs = new_obs

    cum_total_reward += total_reward
    total_reward_hist.append(cum_total_reward)

q = qlearn.q
policy = np.argmax(q, axis=1)

value = np.max(q, axis=1)
policy = policy.reshape(maze_shape)
value = value.reshape(maze_shape)
maze = read_maze(maze_file)
im = graph_value_policy(value, policy, maze)

# plt.figure()
# plt.plot(range(episodes), reward_hist)

timestr = time.strftime("%Y%m%d-%H%M%S")

q3 = q
q_start_hist3 = q_start_hist
total_reward_hist3 = total_reward_hist

np.save("data/{}-large-q-epsilon-p1.npy".format(timestr), q1)
np.save("data/{}-large-q_start_hist-epsilon-p1.npy".format(timestr), q_start_hist1)
np.save("data/{}-large-total_reward_hist-epsilon-p1.npy".format(timestr), total_reward_hist1)

np.save("data/{}-large-q-epsilon-p05.npy".format(timestr), q2)
np.save("data/{}-large-q_start_hist-epsilon-p05.npy".format(timestr), q_start_hist2)
np.save("data/{}-large-total_reward_hist-epsilon-p05.npy".format(timestr), total_reward_hist2)

np.save("data/{}-large-q-epsilon-p2.npy".format(timestr), q3)
np.save("data/{}-large-q_start_hist-epsilon-p2.npy".format(timestr), q_start_hist3)
np.save("data/{}-large-total_reward_hist-epsilon-p2.npy".format(timestr), total_reward_hist3)



















# p = 0.5

# value_iter = []
# policy_iter = []
# value_tot = []
# policy_tot = []

# value_time = []
# policy_time = []

# discount_factors = np.arange(0.8, 1.0, 0.02)

# Ps = np.arange(0.1, 1.0, 0.1)

# # discount_factor=0.9
# df = 0.9
# for df in discount_factors:


#     plt.subplots(12)
#     plt.subplot(121)
#     maze = read_maze(maze_file)
#     env = MazeEnv(maze_file=maze_file)
#     obs = env.reset()

#     prev_time = time()
#     policy, V, num_iter, total_access = value_iteration(env, discount_factor=df)
#     tot_time = time() - prev_time
#     value_time.append(tot_time)

#     value_iter.append(num_iter)
#     value_tot.append(total_access)

#     coord, action = np.where(policy == 1)
#     policy = action.reshape(maze_shape)
#     V = V.reshape(maze_shape)
#     im = graph_value_policy(V, policy, maze)

#     VV = V

#     # plt.show()

#     plt.subplot(122)

#     maze = read_maze(maze_file)
#     # env = WindyMazeEnv(maze_file=maze_file, wind_prob = p)
#     obs = env.reset()
#     prev_time = time()
#     policy, V, num_iter, total_access = policy_improvement(env, discount_factor=df)
#     tot_time = time() - prev_time
#     policy_time.append(tot_time)



#     policy_iter.append(num_iter)
#     policy_tot.append(total_access)


#     coord, action = np.where(policy == 1)
#     policy = action.reshape(maze_shape)
#     V = V.reshape(maze_shape)
#     VP = V
#     im = graph_value_policy(V, policy, maze)
#     # plt.colorbar()
#     print(np.sum(VV - VP))
#     # print(VV[5][5])
#     # print(VP[5][5])
#     # plt.show()

#     # plt.show()
# print("Iterations")
# print(value_iter)
# print(policy_iter)
# print("Count totals")
# print(value_tot)
# print(policy_tot)
# print("Time totals")
# print(value_time)
# print(policy_time)