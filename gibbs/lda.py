import numpy as np
class BasicLDA:
    def __init__(self, alpha, gamma, n_topics, n_documents, n_vocab):
        self.alpha = alpha
        self.gamma = gamma
        self.n_topics = n_topics
        self.n_documents = n_documents
        self.n_vocab = n_vocab

        self.Z = np.zeros(shape=[self.n_documents, self.n_vocab])
        for document_idx in range(self.n_documents):
            for vocab_idx in range(self.n_vocab):
                self.Z[document_idx, vocab_idx] = np.random.randint(self.n_topics)
        
        
        # document-topic distribution
        document_topic_distribution = np.zeros([self.n_documents, self.n_topics])
        for document_idx in range(self.n_documents):
            document_topic_distribution[document_idx] = np.random.dirichlet(self.alpha*np.ones(self.n_topics))
        self.document_topic_distribution = document_topic_distribution

        self.topic_word_distribution = np.zeros([self.n_topics, self.n_vocab])
        for topic_idx in range(self.n_topics):
            self.topic_word_distribution[topic_idx] = np.random.dirichlet(gamma*np.ones(self.n_vocab))
        return
    
    def train(self, document_token_matrix, iterate_times=1000):
        # gibbs
        n_documents = document_token_matrix.shape[0]
        n_words = document_token_matrix.shape[1]
        for t in range(iterate_times):
            for document_idx in range(self.n_documents):
                for vocab_idx in range(self.n_vocab):
                    p_iv = np.exp(
                        np.log(self.document_topic_distribution[document_idx]) + 
                        np.log(self.topic_word_distribution[:, document_token_matrix[document_idx, vocab_idx]]))
                    p_iv /= np.sum(p_iv)

                    self.Z[document_idx, vocab_idx] = np.random.multinomial(1, p_iv).argmax()
            
            for document_idx in range(self.n_documents):
                m = np.zeros(self.n_topics)
                for topic_idx in range(self.n_topics):
                    m[topic_idx] = np.sum(self.Z[document_idx] == topic_idx)
                self.document_topic_distribution[document_idx, :] = np.random.dirichlet(self.alpha + m)

            for topic_idx in range(self.n_topics):
                n = np.zeros(self.n_vocab)
                for vocab_idx in range(self.n_vocab):
                    for document_idx in range(self.n_documents):
                        for vocab_idx_2 in range(self.n_vocab):
                            n[vocab_idx] += (document_token_matrix[document_idx, vocab_idx_2] == vocab_idx) and (self.Z[document_idx, vocab_idx_2] == topic_idx)
                self.topic_word_distribution[topic_idx, :] = np.random.dirichlet(self.gamma + n)

        # for document_idx in range(n_documents):
        #     for word_idx in range(n_words):

        return
    
    def inference(self, document_tokens):
        return

if __name__ == "__main__":
    
    W = np.array([0, 1, 2, 3, 4])
    # D := document words
    X = np.array([
        [0, 0, 1, 2, 2],
        [0, 0, 1, 1, 1],
        [0, 1, 2, 2, 2],
        [4, 4, 4, 4, 4],
        [3, 3, 4, 4, 4],
        [3, 4, 4, 4, 4]
    ])
    N_D = X.shape[0]
    N_W = W.shape[0]
    N_K = 2
    lda_model = BasicLDA(1, 1, N_K, N_D, N_W)
    lda_model.train(X)
    print(lda_model.document_topic_distribution)


        
