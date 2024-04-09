import torch.nn as nn   
import torch
import numpy as np
import random
import os
import torch.nn.functional as F
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import time
import yaml
from itertools import count
import wandb
import gym
from model import Dueling_D3QN
from torchvision import transforms as T
from collections import deque
import numpy as np
import random
import torch.nn as nn   
import torch
import torch.nn.functional as F

########################################## Model ###############################################

class Dueling_D3QN(nn.Module):
    """
    Network structure (D3QN).
    The network will takein a state (a stack of images) and output the Q values for each action.
    """
    def __init__(self, action_dim=12, stack_size=4, H=84, W=84):
        super(Dueling_D3QN, self).__init__()
        self.H = H
        self.W = W

        self.action_dim = action_dim
        self.stack_size = stack_size

        # Convolutional layer
        self.conv1 = nn.Conv2d(stack_size, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)


        # Calculate the flatten size
        new_W = (((W - 8) // 4 + 1 - 4) // 2 + 1 - 3) // 1 + 1
        new_H = (((H - 8) // 4 + 1 - 4) // 2 + 1 - 3) // 1 + 1
        flatten_size = 64 * new_W * new_H

        self.Value1 = nn.Linear(flatten_size, 256)
        self.Adv1 = nn.Linear(flatten_size, 256)        
        self.Value2 = nn.Linear(256, 1) # State value
        self.Adv2 = nn.Linear(256, action_dim) # Advantage value

        torch.nn.init.normal_(self.conv1.weight, 0, 0.01)
        torch.nn.init.normal_(self.conv2.weight, 0, 0.01)
        torch.nn.init.normal_(self.conv3.weight, 0, 0.01)
        torch.nn.init.normal_(self.Value1.weight, 0, 0.01)    
        torch.nn.init.normal_(self.Adv1.weight, 0, 0.01)
        torch.nn.init.normal_(self.Value2.weight, 0, 0.01)
        torch.nn.init.zeros_(self.Adv2.weight)
        torch.nn.init.zeros_(self.Adv2.bias)
        # torch.nn.init.normal_(self.Adv2.weight, 0, 0.01)
        
        self.memory_usage_checker()

    def memory_usage_checker(self):
        """
        Check the memory usage of the model
        """
        total_params = sum(p.numel() for p in self.parameters())
        mem = total_params * 4 / 1024 / 1024
        if mem > 10.0:
            # Use red color if the memory usage is too high
            print(f'\033[91m{mem:.1f}MB memory usage is too high.\033[0m')
        else:
            print(f'{mem:.1f}MB memory usage.')

    def forward(self, x, sample_noise=True):
        # If state is not on the device, move it to the device
        if next(self.parameters()).device != x.device:
            x = x.to(next(self.parameters()).device)

        x = x.view(-1, self.stack_size, self.W, self.H)   

        # Image-wise convolutional layer
        x = self.conv1(x)
        x = F.elu(x)
        x = self.conv2(x)
        x = F.elu(x)
        x = self.conv3(x)
        x = F.elu(x)

        # Flatten
        x = nn.Flatten()(x)
        
        val = self.Value1(x)
        val = F.elu(val)    
        val = self.Value2(val)

        adv = self.Adv1(x)
        adv = F.elu(adv)        
        adv = self.Adv2(adv)

        adv_ave = torch.mean(adv, axis=1, keepdim=True)

        out = adv + val - adv_ave
        return out

    def select_action(self, states):
        # If the given states is not torch.Tensor, convert it

        if len(states.shape) == 4:
            states = states.squeeze(0)

        assert states.shape[0] == self.stack_size, "The stack size of the given states is not correct."

        # Get the Q values and select the action
        with torch.no_grad():
            Q = self.forward(states)
            action_index = torch.argmax(Q)

        # Return the action index
        return action_index.item()



############################################ PER ###############################################

class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    def __getstate__(self):
        # Return the object's state as a dictionary
        return {
            'tree': self.tree,
            'capacity': self.capacity,
            'data': self.data,
            'n_entries': self.n_entries
        }
    
    def __setstate__(self, state):
        # Set the object's state from the dictionary
        self.tree = state['tree']
        self.capacity = state['capacity']
        self.data = state['data']
        self.n_entries = state['n_entries']

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])

class PER:  # stored as ( s, a, r, s_ ) in SumTree
    
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def __getstate__(self):
        # Return the object's state as a dictionary
        return {
            'capacity': self.capacity,
            'tree': self.tree
        }
    
    def __setstate__(self, state):
        # Set the object's state from the dictionary
        self.capacity = state['capacity']
        self.tree = state['tree']

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def add(self, error, sample):
        p = self._get_priority(error.data.item())
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, error):
        for i, e in zip(idx, error):
            p = self._get_priority(e)
            self.tree.update(i, p)
        
    def size(self):
        return self.tree.n_entries
    

#######################################　Trainer #############################################

class Trainer():
    def __init__(self, folder="./result", ckpt_path=None, hyperpara_path=None, memory_path=None):
        # Create the folder if it doesn't exist
        if not os.path.exists(folder):
            os.makedirs(folder)
        # Create a subfolder named "run{idx}" with the largest idx in the folder
        idx = 0
        while os.path.exists(os.path.join(folder, "run{}".format(idx))):
            idx += 1
        self.folder = os.path.join(folder, "run{}".format(idx))
        os.makedirs(self.folder)
        # Create a subfolder named "models" in the folder
        os.makedirs(os.path.join(self.folder, "models"))

        # Device settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device: ", self.device)

        # Load the hyperparameters if exist
        if hyperpara_path is not None:
            # The file should be in yaml format
            assert hyperpara_path.endswith(".yaml"), "The file should be in yaml format."
            with open(hyperpara_path, "r") as f:
                self.config = yaml.safe_load(f)
            print("Hyperparameters loaded from {}".format(hyperpara_path))
        else:
            self.config = {
                "batch_size": 32,                           # Number of samples in a batch
                "gamma": 0.999,                             # Discount factor
                "resize_frame": True,                       # Resize the frame
                "W":84,                                     # Resize width
                "H":84,                                     # Resize height
                "frame_skip": 4,                            # Number of frames skipped
                "save_frequency": 10,                       # Save the model every save_frequency episodes
                "save_best": True,                          # Save the best model
                "eval_frequency": 10,                       # Evaluate the model every eval_frequency episodes
                "eval_rounds": 1,                           # Number of rounds for evaluation
                "logging_frequency": 1000,                  # Log the results every logging_frequency steps
                "visualize_eval": False,                    # Visualize the evaluation
                "epsilon": 0.1,                             # Initial epsilon value
                "epsilon_min": 0.0001,                      # Minimum epsilon value
                "epsilon_decay": 0.0000005,                 # Epsilon decay rate
                "learning_rate": 0.00005,                   # Learning rate
                "weight_decay": 0.0000,                     # Weight decay
                "action_dim": 5,                            # Number of actions
                "stack_size": 4,                            # Number of frames stacked
                "replay_memory_size": 100000,               # Replay memory size
                "save_memory": False,                       # Save the memory
                "target_network_update_strategy": "hard",   # "hard" or "soft"
                "ema_factor": 0.99999,                      # Exponential moving average (for soft update)
                "update_target_frequency": 10000,           # Update frequency of the target network (fpr hard update)
                "noopthres": 1000,                          # Number of no-op threshold
                "log_wandb": True                           # Log the results to wandb
            }
            print("Hyperparameters are not provided. Using default hyperparameters.")

        # Log the hyperparameters
        self.log_hyperparameters()

        # Set up wandb
        if self.config["log_wandb"]:
            wandb.init(project="mario-rl", config=self.config)
            wandb.run.name = f"run{idx}"
            wandb.run.save()
            print("wandb initialized successfully.")
    

        self.network = Dueling_D3QN(action_dim=self.config["action_dim"], stack_size=self.config["stack_size"], W=self.config["W"], H=self.config["H"]).to(self.device)
        self.target_network = Dueling_D3QN(action_dim=self.config["action_dim"], stack_size=self.config["stack_size"], W=self.config["W"], H=self.config["H"]).to(self.device)

        # Load the model if exist
        if ckpt_path is not None:
            self.network.load_state_dict(torch.load(ckpt_path))
            self.target_network.load_state_dict(torch.load(ckpt_path))
            print("Model loaded from {}".format(ckpt_path))
        else:
            print("Model initialized successfully.")

        # Copy the network parameters to the target network
        self.target_network.load_state_dict(self.network.state_dict())

        # Optimizer and loss function
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.config["learning_rate"], weight_decay=self.config["weight_decay"])
        self.loss_fn = torch.nn.SmoothL1Loss()

        # Memory
        if memory_path is not None:
            self.per = torch.load(memory_path)
            print("Memory loaded from {}".format(memory_path))
        else:
            self.per = PER(self.config["replay_memory_size"])
            print("Memory initialized successfully.")


        # Environment
        self.env = gym_super_mario_bros.make('SuperMarioBros-1-2-v0')
        self.env = JoypadSpace(self.env, COMPLEX_MOVEMENT)

        # Image process
        self.transforms1 = T.Compose( [T.ToTensor(), T.Grayscale()] )
        self.transforms2 = T.Compose( [T.Resize((self.config["W"], self.config["H"]), antialias=True), T.Normalize(0, 255)] )
        self.stack_state = deque(maxlen=4) # frame = frame / 255.0
        print("Initialized successfully.")

    def preprocess_frame(self, observation, reset=False):
        """
        Resize the frame with resize_rate by average pooling.
        """
        observation = self.transforms1(observation.astype('int64').copy())
        observation = self.transforms2(observation.float()).squeeze(0)

        if reset:
            self.stack_state.clear()
        
        while len(self.stack_state) < 4:
            self.stack_state.append(observation)
        self.stack_state.append(observation)
        
        observation = gym.wrappers.frame_stack.LazyFrames(list(self.stack_state))
        
        observation = observation[0].__array__() if isinstance(observation, tuple) else observation.__array__()
        
        observation = torch.tensor(observation).unsqueeze(0)
        
        return observation
    
    def log_hyperparameters(self):
        # Log the hyperparameters
        print("Hyperparameters:")
        for key, value in self.config.items():
            # If target_network_update_strategy is soft, don't print update_target_frequency, vice versa
            if key == "update_target_frequency" and self.config["target_network_update_strategy"] == "soft":
                continue
            elif key == "ema_factor" and self.config["target_network_update_strategy"] == "hard":
                continue
            print("{}: {}".format(key, value))

        # Log into a file with name "hyperparameters.yaml"
        with open(os.path.join(self.folder, "hyperparameters.yaml"), "w") as f:
            for key, value in self.config.items():
                if key == "update_target_frequency" and self.config["target_network_update_strategy"] == "soft":
                    continue
                elif key == "ema_factor" and self.config["target_network_update_strategy"] == "hard":
                    continue
                f.write("{}: {}\n".format(key, value))

    def save_hyperparameters(self, name):
        with open(os.path.join(self.folder, f"hyperparameters_{name}.yaml"), "w") as f:
            for key, value in self.config.items():
                f.write("{}: {}\n".format(key, value))
        print("Hyperparameters saved to {}".format(os.path.join(self.folder, "hyperparameters.yaml")))

    def save_model(self, name):
        path = os.path.join(self.folder, "models", f"model_{name}.pth")
        torch.save(self.network.state_dict(), path)
        print("Model saved to {}".format(path))

    def save_memory(self, name):
        path = os.path.join(self.folder, f"memory_{name}.pkl")
        torch.save(self.per, path)
        print("Memory saved to {}".format(path))

    def epsilon_greedy(self, state, epsilon):
        is_random = random.random() < epsilon
        if is_random:
            return random.randint(0, self.config["action_dim"] - 1)
        else:
            return self.network.select_action(state)
        
    def evaluate(self, episodes=1):
        env = gym_super_mario_bros.make('SuperMarioBros-v0')
        env = JoypadSpace(env, COMPLEX_MOVEMENT)
        total_reward = 0
        self.network.eval()
        print("Start evaluating...")
        for episode in range(episodes):
            begin = time.time()
            state = env.reset()
            state = self.preprocess_frame(state, True)
            done = False
            while not done and time.time() - begin < 120:
                action = self.network.select_action(state) # Greedy action
                reward = 0
                for _ in range(self.config["frame_skip"]):
                    next_state, _reward, done, info = env.step(action)
                    reward += _reward    
                    if done: break
                    if self.config["visualize_eval"]:
                        env.render()

                state = self.preprocess_frame(next_state)
                total_reward += reward
        print("Evaluation done. Average reward: {}".format(total_reward / episodes))
        self.network.train()
        env.close()
        return total_reward / episodes
    
    def train(self):
        # Initialize the state
        state = self.env.reset()
        state = self.preprocess_frame(state)
        done = False
        total_reward = 0
        episodic_reward = 0
        episode = 0
        best_reward = 0
        self.network.train()
        self.target_network.eval()
        noopcnt = 0        
        try:
            for step in count():
                ############ Interaction with the environment ############
                # Select the action
                action = self.epsilon_greedy(state, self.config["epsilon"])
                # print(action)
                if action == 0:
                    noopcnt += self.config["frame_skip"]
                else:
                    noopcnt = 0

                # Perform the action for frame_skip times
                reward = 0
                for _ in range(self.config["frame_skip"]):
                    next_state, single_reward, done, info = self.env.step(action)
                    reward += single_reward
                    if done:
                        break

                # Preprocess the next state
                next_state = self.preprocess_frame(next_state)

                # Compute td error for the prioritized experience replay
                with torch.no_grad():
                    # Compute q values
                    Q = self.network(state)[0, action]
                    # Compute td target
                    if done:
                        target_Q = reward
                    else:
                        target_Q = reward + self.config["gamma"] * self.target_network(next_state).max()

                # Compute the error
                td_error = torch.abs(Q - target_Q)

                # Store the experience
                self.per.add(td_error, (state, action, reward, next_state, done))
                
                # Update the state
                state = next_state

                ###################### Training ###########################
                if self.per.size() > self.config["batch_size"]:
                    # Sample the batch
                    batch, idxs, is_weight = self.per.sample(self.config["batch_size"])
                    states = torch.stack([s[0].clone().detach() for s in batch]).to(self.device)
                    actions = torch.tensor([s[1] for s in batch], dtype=torch.long).to(self.device)
                    rewards = torch.tensor([s[2] for s in batch], dtype=torch.float32).to(self.device)
                    next_states = torch.stack([s[3].clone().detach() for s in batch]).to(self.device)
                    dones = torch.tensor([s[4] for s in batch], dtype=torch.float32).to(self.device)

                    # Compute the Q values
                    Q = self.network(states)
                    Q = Q.gather(1, actions.view(-1, 1)).squeeze()

                    # Compute the target Q values
                    with torch.no_grad():
                        best_actions = self.network(next_states).max(dim=1).indices
                        future_Q = self.target_network(next_states).gather(1, best_actions.view(-1, 1)).squeeze()
                        target_Q = rewards + self.config["gamma"] * future_Q * (1 - dones)

                    # Compute the loss
                    loss = self.loss_fn(Q, target_Q) * torch.tensor(is_weight).to(self.device).mean()

                    # New td error
                    new_td_error = torch.abs(Q - target_Q).detach().cpu().numpy()

                    # Update the priority
                    self.per.update(idxs, new_td_error)

                    # Optimize
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # Update the target network
                    if step % self.config["update_target_frequency"] == 0:
                        if self.config["target_network_update_strategy"] == "hard":
                            self.target_network.load_state_dict(self.network.state_dict())
                        elif self.config["target_network_update_strategy"] == "soft":
                            for target_param, param in zip(self.target_network.parameters(), self.network.parameters()):
                                target_param.data.copy_(self.config["ema_factor"] * target_param.data + (1 - self.config["ema_factor"]) * param.data)  

                    # Logging loss and epsilon
                    if step % self.config["logging_frequency"] == 0:
                        print("Step: {}, Loss: {}, Epsilon: {}".format(step, loss.item(), self.config["epsilon"]))
                        if self.config["log_wandb"]:
                            wandb.log({"Loss": loss.item(), "Epsilon": self.config["epsilon"]}, step=step)

                ############## Post-processing and logging ##############
                total_reward += reward
                episodic_reward += reward

                if done or noopcnt > self.config["noopthres"]: # Episode ends
                    episode += 1 # Increase the episode number
                    self.config["epsilon"] = max(self.config["epsilon"] - self.config["epsilon_decay"], self.config["epsilon_min"])


                    # Evaluate the model and logging
                    print("=========== Episode {} Done ===========".format(episode))
                    if episode % self.config["eval_frequency"] == 0 and self.per.size() > self.config["batch_size"]:
                        avg_reward = self.evaluate(self.config["eval_rounds"])
                        if avg_reward > best_reward and self.config["save_best"]:
                            self.save_model(f"best_{avg_reward}")
                            best_reward = avg_reward
                        print("Step: {}, Episode: {}, Training reward: {}, Evaluate reward: {}".format(step, episode, episodic_reward, avg_reward, self.config["epsilon"]))
                        # Logging
                        if self.config["log_wandb"]:
                            wandb.log({"Training reward": episodic_reward, "Evaluate reward": avg_reward, "Epsilon": self.config["epsilon"]}, step=step)
                    else:
                        print("Step: {}, Episode: {}, Training reward: {}".format(step, episode, episodic_reward, self.config["epsilon"]))
                        if self.config["log_wandb"]:
                            wandb.log({"Training reward": episodic_reward, "Epsilon": self.config["epsilon"]}, step=step)

                    # Save the model
                    if episode % self.config["save_frequency"] == 0 and self.per.size() > self.config["batch_size"]:
                        self.save_model(f"episode_{episode}")

                    # Reset the environment
                    state = self.env.reset()
                    state = self.preprocess_frame(state, True)
                    done = False

                    # Reset the episodic reward
                    episodic_reward = 0

        except KeyboardInterrupt:
            print("Training interrupted. Saving the model and memory...")

            # Save the model and memory
            self.save_model("interrupted")
            if self.config["save_memory"]:
                self.save_memory("interupted")
            self.save_hyperparameters("interupted")
            print("Model and memory saved successfully.")

# Test the Trainer
trainer = Trainer(ckpt_path="./result/run41/models/model_best_5089.0.pth", hyperpara_path=None, memory_path=None)
trainer.train()