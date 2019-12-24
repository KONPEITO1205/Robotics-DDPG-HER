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
    def __init__(self, env_params, params):
        super(actor, self).__init__()
        self.max_action = env_params['action_max']
        self.conv1 = nn.Conv2d(params.input_channel, 12, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(12, 32, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=2, stride=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.g = nn.Linear(3, 12)

        # self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'], 256)
        self.fc1 = nn.Linear(10816+12, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.action_out = nn.Linear(256, 4)

    def forward(self, x, g_coord):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        g = F.relu(self.g(g_coord))
        # print(x.view(-1, 10816).shape) # x.view(-1, 10816)
        # print(g.shape)
        x = F.relu(self.fc1(torch.cat([x.view(-1, 10816)[0], g[0]]).unsqueeze(0)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))

        return actions


class critic(nn.Module):
    def __init__(self, env_params, params):
        super(critic, self).__init__()
        self.conv1 = nn.Conv2d(params.input_channel, 12, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(12, 32, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=2, stride=1)
        self.bn4 = nn.BatchNorm2d(64)
        # self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'] + env_params['action'], 256)
        self.fc1 = nn.Linear(10816, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)

        self.action_value = nn.Linear(3, 256 +4)

        self.q_out = nn.Linear(256 + 4, 1)

    def forward(self, x, actions, g_coord):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        # x = torch.cat([x, actions / self.max_action], dim=1)ß
        x = F.relu(self.fc1(x.view(-1, 10816)[0]))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # print(x.shape)  # -> [256]
        # print(actions.shape)  # -> [256, 3]
        action_value = F.relu(self.action_value(actions))
        # print(action_value.shape)  # -> [256, 268]
        # print(g_coord.shape)       # -> [1, 4]
        # print(torch.cat([x, g_coord[0]]).shape)  # -> [260]
        q_value = F.relu(torch.add(torch.cat([x, g_coord[0]]), action_value))
        q_value = self.q_out(q_value)

        return q_value
