import gensim
from gensim.test.utils import common_texts
import torch
import torch.nn as nn
from torch import optim

from tharsis.nn.seq2seq import vanilla_gru
from tharsis.nn import embedding
from tharsis.nn.seq2seq import engine

def generate_test_emebedding():
    embedding_model = gensim.models.Word2Vec(common_texts, size=100, window=5, min_count=1, workers=4)
    embedding_vectors = embedding_model.wv.vectors
    start_indicator = "[start]"
    end_indicator = "[end]"
    word_map = embedding.generate_word_map_from_word2vec_model(embedding_model)
    embedding_vectors, word_map = embedding.add_special_word_to_embedding_vectors(
        embedding_vectors,
        word_map,
        start_indicator
    )
    embedding_vectors, word_map = embedding.add_special_word_to_embedding_vectors(
        embedding_vectors,
        word_map,
        end_indicator
    )
    return embedding_vectors, word_map


def test_vanilla_encoder():

    embedding_vectors, word_map = generate_test_emebedding()
    n_vocab, n_embedding = embedding_vectors.shape
    vanilla_gru_machine = vanilla_gru.VanillaEncoder(
        embedding_vectors,
        n_input=n_embedding,
        n_hidden=64,
        n_layers=2
    )

    sentence = "[start] human interface computer survey user system [end]"
    sentence_word_ids = [word_map[word] for word in sentence.split(" ")]
    hidden_z = vanilla_gru_machine.init_hidden(1)
    output = None
    
    for word_id in sentence_word_ids:
        output, hidden_z = vanilla_gru_machine(torch.LongTensor([word_id]), hidden_z)
    assert (output.shape == (1, 1, 64))
    assert (hidden_z.shape == (2, 1, 64))

def test_vanilla_encoder_decoder():

    embedding_vectors, word_map = generate_test_emebedding()
    n_vocab, n_embedding = embedding_vectors.shape
    vanilla_encoder = vanilla_gru.VanillaEncoder(
        embedding_vectors,
        n_input=n_embedding,
        n_hidden=64,
        n_layers=2
    )
    vanilla_decoder = vanilla_gru.VanillaDecoder(
        embedding_vectors,
        n_layers=2,
        n_hidden=64,
        n_output=n_vocab,
    )

    sentence = "[start] human interface computer survey user system [end]"
    sentence_word_ids = [word_map[word] for word in sentence.split(" ")]
    hidden_z = vanilla_encoder.init_hidden(1)
    for word_id in sentence_word_ids:
        output, hidden_z = vanilla_encoder(torch.LongTensor([word_id]), hidden_z)
    decode_start = torch.LongTensor([word_map["[start]"]])
    output = decode_start
    hidden = hidden_z
    for i in range(10):
        output, hidden = vanilla_decoder(decode_start, hidden_z)
        print(output)
    print(output, hidden)
    assert (output.shape == torch.Size([1, n_vocab]))


def test_engine():
    embedding_vectors, word_map = generate_test_emebedding()
    n_vocab, n_embedding = embedding_vectors.shape
    vanilla_query_encoder = vanilla_gru.VanillaEncoder(
        embedding_vectors,
        n_input=n_embedding,
        n_hidden=64,
        n_layers=2
    )
    vanilla_content_encoder = vanilla_gru.VanillaEncoder(
        embedding_vectors,
        n_input=n_embedding,
        n_hidden=64,
        n_layers=2
    )
    vanilla_answer_decoder = vanilla_gru.VanillaDecoder(
        embedding_vectors,
        n_layers=2,
        n_hidden=128,
        n_output=n_vocab,
    )
    learning_rate = 0.01
    query_encoder_optimizer = optim.SGD(vanilla_query_encoder.parameters(), lr=learning_rate)
    content_encoder_optimizer = optim.SGD(vanilla_content_encoder.parameters(), lr=learning_rate)
    answer_encoder_optimizer = optim.SGD(vanilla_answer_decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    sentence = "[start] human interface computer survey user system [end]"
    sentence_word_ids = [word_map[word] for word in sentence.split(" ")]

    query_input_tensor = torch.LongTensor(sentence_word_ids)
    content_input_tensor = torch.LongTensor(sentence_word_ids)
    target_answer_tensor = torch.LongTensor(sentence_word_ids)

    loss = engine.vanilla_gru_engine(
        word_map=word_map,
        query_input_tensor=query_input_tensor,
        content_input_tensor=content_input_tensor,
        target_answer_tensor=target_answer_tensor,
        query_encoder=vanilla_query_encoder,
        content_encoder=vanilla_content_encoder,
        answer_decoder=vanilla_answer_decoder,
        query_encoder_optimizer=query_encoder_optimizer,
        content_encoder_optimizer=content_encoder_optimizer,
        answer_decoder_optimizer=answer_encoder_optimizer,
        criterion=criterion)




