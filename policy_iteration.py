# Taken from Policy Evaluation Exercise!
import numpy as np
from mdp_graph import graph_value_policy
import matplotlib.pyplot as plt

def policy_eval(policy, env, discount_factor=1.0, theta=0.000001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.

    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment.
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with a random (all 0) value function
    V = np.zeros(env.nS)
    total_access = 0
    while True:
        delta = 0
        # For each state, perform a "full backup"
        total_access += 1
        for s in range(env.nS):
            v = 0
            # Look at the possible next actions
            for a, action_prob in enumerate(policy[s]):
                # For each action, look at the possible next states...
                for  prob, next_state, reward, done in env.P[s][a]:
                    # Calculate the expected value
                    # v += action_prob * prob * (reward + discount_factor * V[next_state])


                    if done:
                        v += action_prob * prob * reward
                    else:
                        v += action_prob * prob * (reward + discount_factor * V[next_state])
            # How much our value function changed (across any states)
            delta = max(delta, np.abs(v - V[s]))
            # if (np.abs(v - V[s]) >= 0.5):
            #     print("trace")
            #     print(s)
            #     print(V[s])
            #     print(v)
            #     print(delta)
            if s == 0:

                pass
                # print(V[s])
                # print(v)
                # print(delta)
            V[s] = v
        # Stop evaluating once our value function change is below a threshold
        if delta < theta:
            break
    return np.array(V), total_access

def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=1.0):
    """
    Policy Improvement Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.

    Args:
        env: The OpenAI envrionment.
        policy_eval_fn: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.

    Returns:
        A tuple (policy, V).
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.

    """
    hist = []
    def one_step_lookahead(state, V):
        """
        Helper function to calculate the value for all action in a given state.

        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.nS

        Returns:
            A vector of length env.nA containing the expected value of each action.
        """
        A = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[state][a]:

                if done:
                    A[a] += prob * (reward)
                else:
                    A[a] += prob * (reward + discount_factor * V[next_state])
        return A

    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA

    i = 0
    total_access = 0
    while True:
        i += 1

        # Evaluate the current policy
        V, tot_count = policy_eval_fn(policy, env, discount_factor)
        total_access += tot_count

        # Will be set to false if we make any changes to the policy
        policy_stable = True
        num_changed = 0
        # For each state...
        for s in range(env.nS):

            # The best action we would take under the currect policy
            chosen_a = np.argmax(policy[s])

            # Find the best action by one-step lookahead
            # Ties are resolved arbitarily
            action_values = one_step_lookahead(s, V)
            best_a = np.argmax(action_values)

            # Greedily update the policy
            if chosen_a != best_a:
                policy_stable = False
                num_changed += 1
                # print(s)
            policy[s] = np.eye(env.nA)[best_a]
        hist.append(num_changed)

        if (len(hist) >= 4):
            print(hist)
            if(hist[-1] == hist[-3] and hist[-2] == hist[-4]):
                policy_stable = True

        # print(i)
        # V = V.reshape((16, 16))
        # coord, action = np.where(policy == 1)
        # policy1 = action.reshape((16, 16))
        # im = graph_value_policy(V, policy1.reshape((16, 16)), env.desc.reshape((16, 16 )))
        # plt.show()

        # If the policy is stable we've found an optimal policy. Return it
        if policy_stable:
            print("Iterations needed: {}".format(i))
            return policy, V, i, total_access