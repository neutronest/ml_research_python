import tensorflow as tf
import numpy as np

from research.nn.self_attention import SelfAttention

INPUT_SIZE = 64
HIDDEN_SIZE = 16
SEQUENCE_SIZE = 3
BATCH_SIZE = 10

def test_get_query_key_value_vectors():
    session_conf = tf.ConfigProto()
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        
        input_data = []
        for sentence_id in range(BATCH_SIZE):
            sentence_embedding = []
            for word_id in range(SEQUENCE_SIZE):
                word_embedding = np.random.normal(0, 1, INPUT_SIZE)
                sentence_embedding.append(word_embedding)
            input_data.append(sentence_embedding)
        input_data = np.array(input_data)
        
        self_attention_machine = SelfAttention(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            max_sequence_length=SEQUENCE_SIZE
        )
        sess.run(tf.global_variables_initializer())

        feed_dict = {
            self_attention_machine.input_x: input_data
        }
        
        result = sess.run(
            [self_attention_machine.qkv_list],
            feed_dict=feed_dict
        )
        qkv_tuple_list_result = result[0]
        query_list_result = qkv_tuple_list_result[0]
        key_list_result = qkv_tuple_list_result[1]
        value_list_result = qkv_tuple_list_result[2]
        assert(query_list_result.shape[0] == BATCH_SIZE)
        assert(query_list_result.shape[1] == SEQUENCE_SIZE)
        assert(query_list_result.shape[2] == 1)
        assert(query_list_result.shape[3] == HIDDEN_SIZE)
    return