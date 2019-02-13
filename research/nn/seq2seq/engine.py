import random

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# class VanillaGRUEngine:
#     def __init__(
#         ):
#
#         self.query_input_tensor = query_input_tensor
#         self.content_input_tensor = content_input_tensor
#         self.target_tensor = target_tensor
#         self.query_encoder = query_encoder
#         self.content_encoder = content_encoder
#         self.answer_decoder = answer_decoder
#         self.query_encoder_optimizer = query_encoder_optimizer
#         self.content_encoder_optimizer = content_encoder_optimizer
#         self.answer_decoder_optimizer = answer_decoder_optimizer
#         self.criterion = criterion
#         self.max_length = max_length
#         return
#
#     def run(self):
#         for i in range(query_input_tensor.shape[0]):



def vanilla_gru_engine(
        word_map,
        query_input_tensor,
        content_input_tensor,
        target_answer_tensor,
        query_encoder,
        content_encoder,
        answer_decoder,
        query_encoder_optimizer,
        content_encoder_optimizer,
        answer_decoder_optimizer,
        criterion,
        max_length=50,
        batch_size=1,
        teacher_forcing_ratio=0.5):


    query_encoder_optimizer.zero_grad()
    content_encoder_optimizer.zero_grad()
    answer_decoder_optimizer.zero_grad()

    query_encoder_hidden = query_encoder.init_hidden(batch_size)
    content_encoder_hidden = content_encoder.init_hidden(batch_size)

    input_length = query_input_tensor.size(0)
    # query_encoder_outputs = torch.zeros(max_length, query_encoder.n_hidden)
    # content_encoder_outputs = torch.zeros(max_length, content_encoder.n_hidden)
    loss = 0

    for i in range(input_length):
        query_encoder_output, query_encoder_hidden = query_encoder(query_input_tensor[i], query_encoder_hidden)
        content_encoder_output, content_encoder_hidden = content_encoder(content_input_tensor[i], content_encoder_hidden)
    encoder_hidden = torch.cat((query_encoder_hidden, content_encoder_hidden), 2)
    decorder_hidden = encoder_hidden
    decorder_input = torch.LongTensor([[word_map.get("[start]")]], device=device)
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    # use_teacher_forcing = True
    if use_teacher_forcing:
        for j in range(target_answer_tensor.size(0)):
            decoder_output, decorder_hidden = answer_decoder(decorder_input, decorder_hidden)

            topv, topi = decoder_output.topk(1)
            decoder_output_label = topi.squeeze().detach()

            # num_classes = decoder_output.shape[1]
            # true_output = (target_answer_tensor[j] == torch.arange(num_classes).reshape(1, num_classes))
            loss += criterion(decoder_output, target_answer_tensor[j].reshape(1))
            decorder_input = target_answer_tensor[j]
    else:
        for j in range(target_answer_tensor.size(0)):
            decoder_output, decorder_hidden = answer_decoder(decorder_input, decorder_hidden)
            # ???
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()
            loss += criterion(decoder_output, target_answer_tensor[j].reshape(1))
            if word_map.get(decoder_input.item()) == "[end]":
                break
    loss.backward()
    query_encoder_optimizer.step()
    content_encoder_optimizer.step()
    answer_decoder_optimizer.step()
    return loss.item() / target_answer_tensor.size(0)
