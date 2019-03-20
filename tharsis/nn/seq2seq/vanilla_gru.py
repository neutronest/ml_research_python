import torch
import torch.nn as nn
from torch.autograd import Variable
from tharsis.nn import embedding

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VanillaEncoder(nn.Module):
    def __init__(
        self,
        embedding_vectors,
        n_input,
        n_hidden,
        n_layers=1):
        super(VanillaEncoder, self).__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.embedding_layer, n_vocab, embedding_dim = embedding.create_embedding_layer(
            embedding_vectors,
            is_trainable=True
        )
        self.gru = nn.GRU(self.n_input, self.n_hidden, n_layers, batch_first=True)
    
    def forward(self, input_x, hidden_z):
        embedded_x = self.embedding_layer(input_x).view(1, 1, -1)
        output = embedded_x
        output, hidden = self.gru(embedded_x, hidden_z)
        return output, hidden
    
    def init_hidden(self, batch_size):
        return Variable(torch.zeros(self.n_layers, batch_size, self.n_hidden, device=device))

class VanillaDecoder(nn.Module):
    def __init__(
        self,
        embedding_vectors,
        n_hidden,
        n_output,
        n_layers=1,
        dropout_prob=0.1,
        max_length=50):

        super(VanillaDecoder, self).__init__()
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.dropout_prob = dropout_prob
        self.max_length = max_length

        # self.embedding_layer, n_vocab, embedding_dim = embedding.create_embedding_layer(
        #     embedding_vectors=embedding_vectors,
        #     is_trainable=True
        # )
        self.embedding_layer = nn.Embedding(n_output, n_hidden)
        self.gru = nn.GRU(n_hidden, self.n_hidden, n_layers, batch_first=True)
        self.fc = nn.Linear(self.n_hidden, self.n_output)
        self.softmax = nn.LogSoftmax(dim=1)
        return

    def forward(self, prev_output, hidden_z):
        output = self.embedding_layer(prev_output).view(1, 1, -1)
        output = nn.functional.relu(output)
        output, hidden = self.gru(output, hidden_z)
        output = self.softmax(self.fc(output[0]))
        return output, hidden

    def init_hidden(self, batch_size):
        Variable(torch.zeros(self.n_layers, batch_size, self.n_hidden, device=device))
