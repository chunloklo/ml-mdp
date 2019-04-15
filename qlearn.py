import random
import numpy as np
class QLearn:
    def __init__(self, num_states, num_actions, epsilon=0.1, alpha=0.1, gamma=0.9, softmax=False):
        self.q = np.zeros((num_states, num_actions))

        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma
        self.num_actions = num_actions
        self.softmax = softmax

    def getQ(self, state, action):
        return self.q[state, action]
        # return self.q.get((state, action), 1.0)

    def learnQ(self, state, action, reward, value):
        '''
        Q-learning:
            Q(s, a) += alpha * (reward(s,a) + max(Q(s') - Q(s,a))
        '''
        oldv = self.q[state, action]
        if oldv is None:
            self.q[state, action] = reward
        else:
            self.q[state, action] = oldv + self.alpha * (value - oldv)

    def chooseAction(self, state):
        self.temperature = 0.1
        if self.softmax:
            q = [self.getQ(state, a) for a in range(self.num_actions)]
            if self.temperature > 0:
                # Compute action probabilities using temperature; when
                # temperature is high, we're treating values of very different
                # Q-values as more equally choosable
                action_probs_numes = []
                denom = 0
                for m in q:
                    import math
                    val = math.exp(m / self.temperature)
                    action_probs_numes.append(val)
                    denom += val
                action_probs = [x / denom for x in action_probs_numes]

                # Pick random move, in which moves with higher probability are
                # more likely to be chosen, but it is obviously not guaranteed
                rand_val = random.uniform(0, 1)
                prob_sum = 0
                for i, prob in enumerate(action_probs):
                    prob_sum += prob
                    if rand_val <= prob_sum:
                        picked_move = i
                        break
            # print(picked_move)
            return picked_move
            # maxQ = max(q)
            # count = q.count(maxQ)
            # # In case there're several state-action max values
            # # we select a random one among them
            # if count > 1:
            #     best = [i for i in range(self.num_actions) if q[i] == maxQ]
            #     i = random.choice(best)
            # else:
            #     i = q.index(maxQ)

            # action = i


        if random.random() < self.epsilon:
            action = np.random.randint(self.num_actions)
        else:
            q = [self.getQ(state, a) for a in range(self.num_actions)]
            maxQ = max(q)
            count = q.count(maxQ)
            # In case there're several state-action max values
            # we select a random one among them
            if count > 1:
                best = [i for i in range(self.num_actions) if q[i] == maxQ]
                i = random.choice(best)
            else:
                i = q.index(maxQ)

            action = i
        return action

    def learn(self, state1, action1, reward, state2, done):
        maxqnew = max([self.getQ(state2, a) for a in range(self.num_actions)])
        if not done:
            self.learnQ(state1, action1, reward, reward + self.gamma*maxqnew)
        else:
            self.learnQ(state1, action1, reward, reward)