import tensorflow as tf
import numpy as np
class SelfAttention:
    def __init__(
        self,
        input_size,
        hidden_size,
        max_sequence_length
        ):

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.max_sequence_length = max_sequence_length
        self.input_x = tf.placeholder(tf.float32, [None, max_sequence_length, input_size], name="input_x")    

        self.Q_W = tf.get_variable(
            "Q_W",
            shape=[input_size, hidden_size],
            initializer=tf.contrib.layers.xavier_initializer()
        )
        self.K_W = tf.get_variable(
            "K_W",
            shape=[input_size, hidden_size],
            initializer=tf.contrib.layers.xavier_initializer()
        )
        self.V_W = tf.get_variable(
            "V_W",
            shape=[input_size, hidden_size],
            initializer=tf.contrib.layers.xavier_initializer()
        )
        
        qkv_batch_list = tf.map_fn(
            lambda x: tf.map_fn(
                self.get_query_key_value_vectors, 
                x, 
                dtype=(
                    tf.float32,
                    tf.float32,
                    tf.float32
                )),
            self.input_x,
            dtype=(
                    tf.float32,
                    tf.float32,
                    tf.float32
                ))
        # qkv_batch_list = tf.map_fn(
        #     lambda sequence_embedding: tf.map_fn(
        #         self.apply,
        #         sequence_embedding,
        #         dtype=(
        #             tf.float32
        #         )
        #     ),
        #     self.input_x
        # )

        self.qkv_list = qkv_batch_list
        return

    def apply(self, input_embedding):
        qkv_list = tf.map_fn(
            self.get_query_key_value_vectors,
            input_embedding,
            dtype=(
                tf.float32,
                tf.float32,
                tf.float32))
        query_key_softmax_scores = self.get_query_key_score(qkv_list)
        return query_key_softmax_scores
        
    

    def get_query_key_score(self, query_key_value_list):
        query_vector_list = [qkv[0] for qkv in query_key_value_list]
        key_vector_list = [qkv[1] for qkv in query_key_value_list]
        value_vector_list = [qkv[2] for qkv in query_key_value_list]
        
        for embedding_id, query_vector in enumerate(query_vector_list):
            # boardcast
            query_key_scores = query_vector * key_vector_list
            query_key_scores /= tf.sqrt(self.hidden_size)
            query_key_softmax_scores = tf.nn.softmax(query_key_scores, axis=1)
        return query_key_softmax_scores

    def get_query_key_value_vectors(self, input_embedding):
        input_embedding_expand_dim = tf.expand_dims(input_embedding, 0)
        q = tf.matmul(input_embedding_expand_dim, self.Q_W)
        k = tf.matmul(input_embedding_expand_dim, self.K_W)
        v = tf.matmul(input_embedding_expand_dim, self.V_W)
        return q,k,v


    def _embedding_query_iteration(self):
        return
        

if __name__ == "__main__":


    INPUT_SIZE = 64
    HIDDEN_SIZE = 16
    SEQUENCE_SIZE = 3

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto()
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            
            input_data = []
            for sentence_id in range(10):
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
            print(result)
