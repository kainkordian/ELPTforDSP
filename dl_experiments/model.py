import math
import torch
from torch.nn import functional as F
import torch.nn as nn


class MyBaseModel(nn.Module):

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def forward_eval_single(self, *args, **kwargs):
        raise NotImplementedError

    def __reset_parameters__(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class MyCNN(MyBaseModel):
    def __init__(self, *args, **kwargs):
        super(MyCNN, self).__init__()

        # define fields and their types
        self.input_dim: int
        self.output_dim: int
        self.dropout: float
        self.num_conv_kernels: int
        self.conv_kernel_size: int
        self.pool_kernel_size: int

        self.__dict__.update(kwargs)

        self.conv_padding = math.floor(self.conv_kernel_size // 2)
        self.pool_padding = math.floor(self.pool_kernel_size // 2)

        self.conv = nn.Conv1d(
            1, self.num_conv_kernels, self.conv_kernel_size, padding=self.conv_padding, stride=1)

        self.pool = nn.AvgPool1d(
            self.pool_kernel_size, padding=self.pool_padding, stride=1)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(self.num_conv_kernels * self.input_dim, self.input_dim)
        self.fc2 = nn.Linear(self.input_dim, self.output_dim)

        self.__reset_parameters__()

    def forward(self, x):

        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.pool(x)
        x = torch.tanh(x)

        x = self.flatten(x)

        x = F.elu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.fc2(x)

    # Just for compatibility #
    def forward_eval_single(self, x):
        return self.forward(x)

    def __repr__(self):
        return "MyCNN"

    def __str__(self):
        return "MyCNN"


class MyGRU(MyBaseModel):
    def __init__(self, *args, **kwargs):
        super(MyGRU, self).__init__()

        # define fields and their types
        self.input_dim: int
        self.hidden_dim: int
        self.output_dim: int
        self.dropout: float
        self.num_layers: int
        self.bidirectional: bool
        self.device: str

        self.__dict__.update(kwargs)

        # Used to memorize hidden state when doing online sequence predictions
        self.h_previous = None

        fc_input_dim = 2 * self.hidden_dim if self.bidirectional else self.hidden_dim

        self.gru = nn.GRU(
            self.input_dim,
            self.hidden_dim,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional,
            num_layers=self.num_layers)

        self.fc = nn.Linear(fc_input_dim, self.output_dim)

        self.__reset_parameters__()

    def forward(self, x):

        x = x.unsqueeze(0)
        num_directions = 2 if self.bidirectional else 1

        h_0 = self.__init_hidden__(num_directions, x)
        output_gru, _ = self.gru(x, h_0)
        output_gru = output_gru.squeeze(0)

        return self.fc(output_gru)

    def forward_eval_single(self, x_t, reset=False):

        x_t = x_t.unsqueeze(0)
        num_directions = 2 if self.bidirectional else 1

        # Hidden state in first seq of the GRU
        if reset or self.h_previous is None:
            h_0 = self.__init_hidden__(num_directions, x_t)
        else:
            h_0 = self.h_previous

        output_gru, h_n = self.gru(x_t, h_0)
        self.h_previous = h_n
        output_gru = output_gru.squeeze(0)

        return self.fc(output_gru)

    def __init_hidden__(self, num_directions, x):
        h0 = torch.zeros((num_directions * self.num_layers, x.size(0), self.hidden_dim), dtype=torch.float64,
                         device=self.device)
        return h0

    def __repr__(self):
        return "MyGRU"

    def __str__(self):
        return "MyGRU"


class MyCNNGRU(MyBaseModel):
    def __init__(self, *args, **kwargs):
        super(MyCNNGRU, self).__init__()

        # define fields and their types
        self.device: str

        self.__dict__.update(kwargs)

        self.h_conv1 = 32
        # self.h_conv2 = 32
        self.h_rnn = 16

        # Used to memorize hidden state when doing online sequence predictions
        self.h_previous = None

        self.conv1 = nn.Conv1d(1, self.h_conv1, kernel_size=10)
        self.rnn = nn.GRU(
            input_size=96,
            hidden_size=self.h_rnn,
            num_layers=1,
            batch_first=True)
        self.linear = nn.Linear(self.h_rnn, 1)

        self.__reset_parameters__()

    def forward(self, x):
        h_0 = self.__init_hidden__(1, x)
        x = x.reshape((x.shape[0], 5, 1, 20))
        batch_size, timesteps, C, L = x.size()
        c_in = x.view(batch_size * timesteps, C, L)

        c_out = F.relu(F.max_pool1d(self.conv1(c_in), 3))
        # c_out = F.relu(F.max_pool1d(self.conv2(c_out), 2))
        c_out = c_out.view(-1, 96)

        r_in = c_out.view(batch_size, timesteps, -1)
        r_out, h_n = self.rnn(r_in, h_0)
        r_out2 = self.linear(r_out[:, -1, :])
        return r_out2

    def forward_eval_single(self, x_t, reset=False):
        if reset or self.h_previous is None:
            h_0 = self.__init_hidden__(1, x_t)
        else:
            h_0 = self.h_previous

        x = x_t.reshape((x_t.shape[0], 5, 1, 20))
        batch_size, timesteps, C, L = x.size()
        c_in = x.view(batch_size * timesteps, C, L)

        c_out = F.relu(F.max_pool1d(self.conv1(c_in), 3))
        # c_out = F.relu(F.max_pool1d(self.conv2(c_out), 2))
        c_out = c_out.view(-1, 96)

        r_in = c_out.view(batch_size, timesteps, -1)
        r_out, h_n = self.rnn(r_in, h_0)
        self.h_previous = h_n

        r_out2 = self.linear(r_out[:, -1, :])
        return r_out2

    def __init_hidden__(self, num_directions, x):
        h0 = torch.zeros((num_directions * 1, x.size(0), self.h_rnn), dtype=torch.float64,
                         device=self.device)
        return h0

    def __repr__(self):
        return "MyCNNGRU"

    def __str__(self):
        return "MyCNNGRU"
