import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepResnet(nn.Module):
    def __init__(self, num_inputs_features, num_layers, num_hidden, dropout_rate, num_outputs):
        super().__init__()
        
        self.num_inputs_features = num_inputs_features
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.dropout_rate = dropout_rate
        self.num_outputs = num_outputs

        '''Define structures'''
        # Input Layer
        self.input_layer = nn.Linear(num_inputs_features, num_hidden)

        # Hidden Layers
        self.linears = nn.ModuleList(
            [nn.Linear(num_hidden, num_hidden) for i in range(num_layers)]
            )
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

        # Output
        self.classifier = nn.Linear(num_hidden, num_outputs)

    def forward(self, inputs):

        # 1st hidden layer by feeding input data to the neural net
        hidden = self.input_layer(inputs)

        # Compute Resnet hidden layers and return output layer
        for resid in self.linears:
            hidden = hidden + self.dropout(F.relu(resid(hidden)))

        out = self.classifier(hidden)

        return out