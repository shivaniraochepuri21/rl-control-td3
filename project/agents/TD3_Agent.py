from collections import deque
from project.agents.TD3 import *
from project.agents.TD3 import TD3
# device = torch.device('cpu')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("__device__ : ", device)

class Agent():
    def __init__(self, state_dim, action_dim, max_action = 1, action_scale = [1.0], action_add=[0.0], batch_size = 64) -> None:
        self.TD = TD3(state_dim, action_dim, max_action, action_scale, action_add)
        self.steps_done = 0
        self.memory = deque(maxlen=100000)
        self.batch_size = batch_size

    def memorize(self, state, action, reward, next_state, not_done):
        self.memory.append([state, action, reward, next_state, not_done])
    
    def learn(self):
        if len(self.memory) < self.batch_size:  
            return 0, 0
        err_actor, err_critic = self.TD.train(self.memory, self.batch_size)
        return err_actor, err_critic

    def act(self, state):
        action = self.TD.select_action(state)
        return action
