import torch

from research.nn.naive_transformer import NaiveEncoder, NaiveDecoder


n_batch = 2
n_head = 3
n_seq = 5
n_input = 7
n_hidden = 11

def test_naive_encoder():

    encoder = NaiveEncoder(
        n_head=n_head,
        n_query=n_input,
        n_key=n_input,
        n_value=n_input,
        n_hidden=n_hidden
    )
    input_x = torch.randn(n_batch, n_seq, n_input)
    output = encoder(input_x, input_x, input_x)
    print(output.size())

def test_naive_decoder():
    encoder = NaiveEncoder(
        n_head=n_head,
        n_query=n_input,
        n_key=n_input,
        n_value=n_input,
        n_hidden=n_hidden
    )
    decoder = NaiveDecoder(
        n_head=n_head,
        n_query=n_input,
        n_key=n_input,
        n_value=n_input,
        n_hidden=n_hidden
    )
    input_x = torch.randn(n_batch, n_seq, n_input)
    encoder_output = encoder(input_x, input_x, input_x)
    decoder_output = decoder(input_x, encoder_output)
    print(decoder_output.size())

    
