import torch

from research.nn.naive_transformer import NaiveEncoder


n_batch = 2
n_head = 3
n_seq = 5
n_input = 7
n_hidden = 11

def test_naive_encoder():

    encoder = NaiveEncoder(
        n_head,
        n_input,
        n_hidden
    )
    input_x = torch.randn(n_batch, n_seq, n_input)
    output = encoder(input_x)
    print(output.size())