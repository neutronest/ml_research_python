import numpy as np
import torch
import torch.nn as nn


class NaiveMultiHeadSelfAttention(nn.Module):

    def __init__(
        self,
        n_head,
        n_query,
        n_key,
        n_value,
        n_hidden
        ):
        super().__init__()
        self.n_head = n_head
        self.n_query  = n_query
        self.n_key = n_key
        self.n_value = n_value
        self.hidden_size = n_hidden
        
        self.w_queries = nn.Linear(n_query, n_head * n_hidden)
        self.w_keyes = nn.Linear(n_key, n_head * n_hidden)
        self.w_values = nn.Linear(n_value, n_head * n_hidden)

        nn.init.normal_(self.w_queries.weight, mean=0, std=np.sqrt(2.0 / (n_query+n_hidden)))
        nn.init.normal_(self.w_keyes.weight, mean=0, std=np.sqrt(2.0 / (n_key+n_hidden)))
        nn.init.normal_(self.w_values.weight, mean=0, std=np.sqrt(2.0 / (n_value+n_hidden)))

        self.temprature = np.power(self.hidden_size, 0.5)
        self.softmax_fn = nn.Softmax(dim=2)    
        return
    
    def forward(self, input_query, input_key, input_value, mask=None):
        """
        """

        n_batch, len_of_sequence, _ = input_query.size()
        
        query_vector = self.w_queries(input_query)
        key_vector = self.w_keyes(input_key)
        value_vector = self.w_values(input_value)
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
        
        """
        apply the score to the value vector, 
        shap：n_batch*n_head, seq_length, hidden_size
        """
        affect_value_scores = torch.bmm(query_key_scores_softmax, value_vector)
        
        """
        swap the dimension to n_batch, seq_length, n_head*hidden_size
        """
        affect_value_result = affect_value_scores.view(n_batch,  self.n_head, len_of_sequence, self.hidden_size)
        affect_value_result = affect_value_result\
            .permute(0,2,1,3)\
            .contiguous()\
            .view(n_batch, len_of_sequence, self.n_head * self.hidden_size)
        return affect_value_result

class NaiveFeedForwardNeuralNetwork(nn.Module):
    def __init__(
        self,
        n_head,
        n_input,
        n_hidden
        ):
        super().__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_head = n_head
        self.feed_forward_fn = nn.Linear(n_input, n_head*n_hidden)
        return
    
    def forward(self, input_x):
        """
        input_x: torch.Tensor
                 shape: [n_batch, n_seq, n_head*n_hidden]
        """
        n_batch, n_seq, _ = input_x.size()
        input_x_transpose = input_x.contiguous().view(n_batch, n_seq, self.n_head, self.n_input)

        output = self.feed_forward_fn(input_x)
        return output.contiguous().view(n_batch, n_seq, self.n_head*self.n_hidden)


class PositionwiseFeedForwardNetwork(nn.Module):
    def __init__(self, n_input, n_hidden, dropout=0.1):
        super(PositionwiseFeedForwardNetwork, self).__init__()
        self.w_1 = nn.Linear(n_input, n_hidden)
        self.w_2 = nn.Linear(n_hidden, n_input)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.w_2(self.dropout(nn.functional.relu(self.w_1(x))))