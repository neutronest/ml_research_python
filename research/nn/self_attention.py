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
        self.w_values = nn.Linear(input_size, n_head, hidden_size)

        nn.init.normal_(self.w_queries, mean=0, std=np.sqrt(2.0 / 100))
        
        return