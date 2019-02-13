import gensim
from gensim.test.utils import common_texts

from research.nn import embedding

def test_add_special_word_to_embedding_vectors():
    embedding_model = gensim.models.Word2Vec(common_texts, size=100, window=5, min_count=1, workers=4)
    embedding_vector = embedding_model.wv.vectors
    assert (embedding_vector.shape == (12, 100))

    speical_indicator = "[start]"
    word_map = embedding.generate_word_map_from_word2vec_model(embedding_model)
    embedding_vector, word_map = embedding.add_special_word_to_embedding_vectors(
        embedding_vector,
        word_map,
        speical_indicator
    )
    assert (embedding_vector.shape == (13, 100))
    assert (word_map.get(speical_indicator) == 12)
    return