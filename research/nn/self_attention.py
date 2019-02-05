import numpy as np
import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        n_head
        ):
        super().__init__()
        self.n_head = n_head
        self.input_size  = input_size
        self.hidden_size = hidden_size
        
        self.w_queries = nn.Linear(input_size, n_head * hidden_size)
        self.w_keyes = nn.Linear(input_size, n_head * hidden_size)
        self.w_values = nn.Linear(input_size, n_head * hidden_size)

        nn.init.normal_(self.w_queries.weight, mean=0, std=np.sqrt(2.0 / (input_size+hidden_size)))
        nn.init.normal_(self.w_keyes.weight, mean=0, std=np.sqrt(2.0 / (input_size+hidden_size)))
        nn.init.normal_(self.w_values.weight, mean=0, std=np.sqrt(2.0 / (input_size+hidden_size)))        
        return
    
    def forward(self, input_x, mask=None):
        """
        """

        n_batch, len_of_sequence, _ = input_x.size()
        
        query_vector = self.w_queries(input_x)
        key_vector = self.w_keyes(input_x)
        value_vector = self.w_values(input_x)
        query_vector = query_vector.view(n_batch, len_of_sequence, self.n_head, self.hidden_size)
        key_vector = key_vector.view(n_batch, len_of_sequence, self.n_head, self.hidden_size)
        value_vector = value_vector.view(n_batch, len_of_sequence, self.n_head, self.hidden_size)
        
        return