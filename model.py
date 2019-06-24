
import torch
import torch.nn as nn


class DurationModel(nn.Module):
    def __init__(self, D_in, H, D_out, layers):
        super(DurationModel, self).__init__()
        self.input_linear = nn.Linear(D_in, H)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(H, H) for _ in range(layers)])
        self.output_linear = nn.Linear(H, D_out)
        self.relu = nn.Tanh()

    def forward(self, x):
        h = self.relu(self.input_linear(x))
        for hl in self.hidden_layers:
            h = self.relu(hl(h))
        return self.output_linear(h)


class AcousticModel(nn.Module):
    def __init__(self, D_in, H, D_out, layers):
        super(AcousticModel, self).__init__()
        self.input_linear = nn.Linear(D_in, H)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(H, H) for _ in range(layers)])
        self.output_linear = nn.Linear(H, D_out)
        self.relu = nn.Tanh()

    def forward(self, x):
        h = self.relu(self.input_linear(x))
        for hl in self.hidden_layers:
            h = self.relu(hl(h))
        return self.output_linear(h)


class RNNet(nn.Module):
    def __init__(self, D_in, H, D_out, num_layers=1, bidirectional=False):
        super(RNNet, self).__init__()
        num_directions = 2 if bidirectional else 1
        self.gru = nn.GRU(D_in, H, num_layers, batch_first=True, bidirectional=bidirectional)
        self.output_linear = nn.Linear(num_directions * H, D_out)

    def forward(self, inputs, input_lengths=None):
        if input_lengths is not None:
            inputs = nn.utils.rnn.pack_padded_sequence(inputs, input_lengths, batch_first=True)
        outputs, _ = self.gru(inputs)
        if input_lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        outputs = self.output_linear(outputs)
        return outputs

