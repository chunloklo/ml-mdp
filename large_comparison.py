from maze_generator import MazeEnv, read_maze
from value_iteration import value_iteration
from policy_iteration import policy_improvement
import numpy as np
from mdp_graph import graph_value_policy
import matplotlib.pyplot as plt
from time import time

maze_shape = (32, 32)
maze_file = 'maze/mazeLarge.png'

p = 0.5

value_iter = []
policy_iter = []
value_tot = []
policy_tot = []

value_time = []
policy_time = []

discount_factors = np.arange(0.8, 1.0, 0.02)

Ps = np.arange(0.1, 1.0, 0.1)




# p_val = []
# p_pol = []
# # discount_factor=0.9
# df = 0.9
# for p in Ps:


#     plt.subplots(12)
#     plt.subplot(121)
#     maze = read_maze(maze_file)
#     env = WindyMazeEnv(maze_file=maze_file, wind_prob = p)
#     obs = env.reset()

#     prev_time = time()
#     policy, V, num_iter, total_access = value_iteration(env, discount_factor=df)
#     tot_time = time() - prev_time
#     # value_time.append(tot_time)
#     p_val.append(num_iter)
#     # value_iter.append(num_iter)
#     # value_tot.append(total_access)

#     coord, action = np.where(policy == 1)
#     policy = action.reshape(maze_shape)
#     V = V.reshape(maze_shape)
#     im = graph_value_policy(V, policy, maze)

#     VV = V

#     plt.colorbar()


#     plt.subplot(122)

#     maze = read_maze(maze_file)
#     # env = WindyMazeEnv(maze_file=maze_file, wind_prob = p)
#     obs = env.reset()
#     prev_time = time()
#     policy, V, num_iter, total_access = policy_improvement(env, discount_factor=df)
#     tot_time = time() - prev_time
#     # policy_time.append(tot_time)

#     p_pol.append(num_iter)

#     # policy_iter.append(num_iter)
#     # policy_tot.append(total_access)


#     coord, action = np.where(policy == 1)
#     policy = action.reshape(maze_shape)
#     V = V.reshape(maze_shape)
#     VP = V
#     im = graph_value_policy(V, policy, maze)
#     plt.colorbar()
#     # print(np.sum(VV - VP))
#     # print(VV[5][5])
#     # print(VP[5][5])



# print(p_val)
# print(p_pol)





# discount_factor=0.9
df = 0.9
for df in discount_factors:


    plt.subplots(12)
    plt.subplot(121)
    maze = read_maze(maze_file)
    env = MazeEnv(maze_file=maze_file)
    obs = env.reset()

    prev_time = time()
    policy, V, num_iter, total_access = value_iteration(env, discount_factor=df)
    tot_time = time() - prev_time
    value_time.append(tot_time)

    value_iter.append(num_iter)
    value_tot.append(total_access)

    coord, action = np.where(policy == 1)
    policy = action.reshape(maze_shape)
    V = V.reshape(maze_shape)
    im = graph_value_policy(V, policy, maze)

    VV = V

    # plt.show()

    plt.subplot(122)

    maze = read_maze(maze_file)
    # env = WindyMazeEnv(maze_file=maze_file, wind_prob = p)
    obs = env.reset()
    prev_time = time()
    policy, V, num_iter, total_access = policy_improvement(env, discount_factor=df)
    tot_time = time() - prev_time
    policy_time.append(tot_time)



    policy_iter.append(num_iter)
    policy_tot.append(total_access)


    coord, action = np.where(policy == 1)
    policy = action.reshape(maze_shape)
    V = V.reshape(maze_shape)
    VP = V
    im = graph_value_policy(V, policy, maze)
    # plt.colorbar()
    print(np.sum(VV - VP))
    # print(VV[5][5])
    # print(VP[5][5])
    # plt.show()

    # plt.show()
print("Iterations")
print(value_iter)
print(policy_iter)
print("Count totals")
print(value_tot)
print(policy_tot)
print("Time totals")
print(value_time)
print(policy_time)