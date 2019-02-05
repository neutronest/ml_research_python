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

        self.temprature = np.power(self.hidden_size, 0.5)
        self.softmax_fn = nn.Softmax(dim=2)    
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
        

        # todo: understand
        # shape: n_batch * n_head, seq_length, hidden_size
        query_vector = query_vector.permute(2, 0, 1, 3).contiguous().view(-1, len_of_sequence, self.hidden_size)
        key_vector = key_vector.permute(2, 0, 1, 3).contiguous().view(-1, len_of_sequence, self.hidden_size)
        value_vector = value_vector.permute(2,0,1,3).contiguous().view(-1, len_of_sequence, self.hidden_size)
        """
        get score by q1*k1, q1*k2 .. q2*k1, q2*k2
        for each token in one seq, generate a seq_length vector to represent the score vector
        shape: n_batch*n_head, seq_length, seq_length
        """
        query_key_scores = torch.bmm(query_vector, key_vector.transpose(1, 2))
        query_key_scores = query_key_scores / self.temprature
        query_key_scores_softmax = self.softmax_fn(query_key_scores)
        
        affect_value_scores = torch.bmm(query_key_scores_softmax, value_vector)
        affect_value_result = affect_value_scores.contiguous().view(n_batch, self.n_head, len_of_sequence, self.hidden_size)
        return affect_value_scores