import gensim
from gensim.test.utils import common_texts
import numpy as np
from torch import optim
import torch
import torch.nn as nn


from research.nn import embedding
from research.squad import data
from research.nn.seq2seq import vanilla_gru
from research.nn.seq2seq.engine import vanilla_gru_engine

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():

    # word2vec_model = gensim.models.Word2Vec(common_texts, size=100, window=5, min_count=1, workers=4)
    # embedding_vectors = word2vec_model.wv.vectors
    # word_map = embedding.generate_word_map_from_word2vec_model(word2vec_model)

    # word2vec_data_path = "./data/word_embedding_model/GoogleNews-vectors-negative300.bin.gz"
    # word2vec_model =  embedding.load_word2vec(word2vec_data_path)
    # embedding_vectors = word2vec_model.wv.vectors
    #word_map = embedding.generate_word_map_from_word2vec_model(word2vec_model)
    squad_data_list = data.generate_squad_data_list("./data/squadv2.0/train-v2.0.json")
    # test for small data
    flatten_data_list = data.generate_paragraph_question_answer_data(squad_data_list)
    # flatten_data_list = flatten_data_list[:10]
    print("date size: {}".format(len(flatten_data_list)))
    print("generate squad token map")
    squad_token_map = data.generate_tokens_map_by_flatten_pqa_data(flatten_data_list)
    print("Done")
    

    word_map = {"guard": 0}
    n_dimension = 100
    embedding_vectors = np.array(np.random.rand(1, n_dimension))
    print("Squad words' length: {}".format(len(list(squad_token_map.keys()))))
    for token_id, token in enumerate(list(squad_token_map.keys())):
        if token_id % 100 == 0:
            print(token_id)
        if word_map.get(token) is None:
            current_vocab_length = len(word_map.values())
            embedding_vectors, word_map = embedding.add_special_word_to_embedding_vectors(
                embedding_vectors,
                word_map,
                token
            )
            word_map[token] = current_vocab_length

    embedding_vectors, word_map = embedding.add_special_word_to_embedding_vectors(
        embedding_vectors,
        word_map,
        "[start]"
    )
    embedding_vectors, word_map = embedding.add_special_word_to_embedding_vectors(
        embedding_vectors,
        word_map,
        "[end]"
    )
    # weights = torch.FloatTensor(embedding_vectors)
    # embedding_result = nn.Embedding.from_pretrained(weights)

    n_vocab, n_embedding = embedding_vectors.shape
    vanilla_query_encoder = vanilla_gru.VanillaEncoder(
        embedding_vectors,
        n_input=n_embedding,
        n_hidden=16,
        n_layers=1
    )
    vanilla_content_encoder = vanilla_gru.VanillaEncoder(
        embedding_vectors,
        n_input=n_embedding,
        n_hidden=16,
        n_layers=1
    )
    vanilla_answer_decoder = vanilla_gru.VanillaDecoder(
        embedding_vectors,
        n_layers=1,
        n_hidden=32,
        n_output=n_vocab,
    )

    n_epoch = 1000
    print_every = 1000
    learning_rate = 0.02
    query_encoder_optimizer = optim.SGD(vanilla_query_encoder.parameters(), lr=learning_rate)
    content_encoder_optimizer = optim.SGD(vanilla_content_encoder.parameters(), lr=learning_rate)
    answer_encoder_optimizer = optim.SGD(vanilla_answer_decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    train_iter = 0
    print_loss_total = 0.0
    for epoch in range(n_epoch):
        for i in range(len(flatten_data_list)):
            content, question, answer = flatten_data_list[i]

            #sentence_word_ids = [word_map[word] for word in sentence.split(" ")]
            content_word_ids = [word_map[word] for word in content.split(" ")]
            query_word_ids = [word_map[word] for word in question.split(" ")]
            answer_word_ids = [word_map[word] for word in answer.split(" ")]
            query_input_tensor = torch.LongTensor(query_word_ids, device=device)
            content_input_tensor = torch.LongTensor(content_word_ids, device=device)
            target_answer_tensor = torch.LongTensor(answer_word_ids, device=device)

            loss = vanilla_gru_engine(
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
            print_loss_total += loss

            if train_iter % print_every  == 0:
                print("Epoch: {}, data_i: {}, loss: {}".format(epoch, i, print_loss_total * 1.0 / print_every))
                print_loss_total = 0.0
            train_iter += 1


if __name__ == "__main__":
    main()