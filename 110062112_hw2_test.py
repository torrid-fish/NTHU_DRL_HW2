import torch.nn as nn   
import torch
import torch.nn.functional as F
from collections import deque
import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from torchvision import transforms as T

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
        torch.nn.init.normal_(self.Adv2.weight, 0, 0.01)
        
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

class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.model = Dueling_D3QN(
            action_dim=5,
            stack_size=4,
            H=84,
            W=84
        )
        path = "./110062112_hw2_data"
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

        # Get initial state
        env = gym_super_mario_bros.make('SuperMarioBros-v0')
        env = JoypadSpace(env, COMPLEX_MOVEMENT)
        self.init_state = env.reset()

        # Preprocess the frame
        self.transforms1 = T.Compose( [T.ToTensor(), T.Grayscale()] )
        self.transforms2 = T.Compose( [T.Resize((84, 84), antialias=True), T.Normalize(0, 255)] )
        self.stack_state = deque(maxlen=4) #     frame = frame / 255.0

        # Frame skip counter
        self.frame_skip = 0
        self.action = 0
        self.act_counter = 0

    def preprocess_frame(self, observation, reset=False):
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

    def act(self, observation):
        if self.act_counter == 2835:
            self.act_counter = 1
            reset = True
        else:
            self.act_counter += 1
            reset = False

        if reset or self.frame_skip % 4 == 0 or (self.act_counter > 1994 and self.act_counter < 2100):
            self.action = self.model.select_action(self.preprocess_frame(observation, reset))
            self.reset = False
            self.frame_skip = 1
        else:
            self.frame_skip += 1

        return self.action