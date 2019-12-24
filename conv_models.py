import torch
import torch.nn as nn
import torch.nn.functional as F

"""
the input x in both networks should be [o, g], where o is the observation and g is the goal.

"""


# TODO: change to convolutional network
# TODO: image observation concat with goal state

# define the actor network
class actor(nn.Module):
    def __init__(self, env_params):
        super(actor, self).__init__()
        self.max_action = env_params['action_max']
        self.conv1 = nn.Conv2d(env_params['input_channel'], 12, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(12, 32, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=4, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        self.g = nn.Linear(env_params['goal'], env_params['goal'])

        # self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'], 256)
        self.fc1 = nn.Linear(32*16*16+env_params['goal'], 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.action_out = nn.Linear(256, env_params['action'])

    def forward(self, x, g_coord):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        g = F.relu(self.g(g_coord))

        x = F.relu(self.fc1(torch.cat([x.view(-1, 32*16*16), g])))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))

        return actions


class critic(nn.Module):
    def __init__(self, env_params):
        super(critic, self).__init__()
        self.max_action = env_params['action_max']
        self.conv1 = nn.Conv2d(env_params['input_channel'], 12, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(12, 32, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=4, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        # self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'] + env_params['action'], 256)
        self.fc1 = nn.Linear(32*16*16, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)

        self.action_value = nn.Linear(self.max_action, 256 + env_params['goal'])

        self.q_out = nn.Linear(256 + env_params['goal'], 1)

    def forward(self, x, actions, g_coord):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # x = torch.cat([x, actions / self.max_action], dim=1)ß
        x = F.relu(self.fc1(x.view(-1, 32 * 16 * 16)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        action_value = F.relu(self.action_value(actions))
        q_value = F.relu(torch.add(torch.cat([x, g_coord]), action_value))
        q_value = self.q_out(q_value)

        return q_value
