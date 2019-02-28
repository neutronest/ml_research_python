import torch

from transformer import preprocess
from transformer.transformer import Constants as Constants
from research.squad import data

def prepare_data_instances(data_path):
    squad_data_list = data.generate_squad_data_list(data_path)
    flatten_data_list = data.generate_paragraph_question_answer_data(squad_data_list)

    word_instances = []
    pqa_instances = []
    for (paragraph_text, question_text, answer_text) in flatten_data_list:
        p_instance = [Constants.BOS_WORD] + paragraph_text.split(" ") + [Constants.EOS_WORD]
        q_instance = [Constants.BOS_WORD] + question_text.split(" ") + [Constants.EOS_WORD]
        a_instance = [Constants.BOS_WORD] + answer_text.split(" ") + [Constants.EOS_WORD]
        # concate the question and paragraph together
        word_instances.append(q_instance + p_instance)
        word_instances.append(a_instance)
        pqa_instances.append((q_instance + p_instance, a_instance))
    return word_instances, pqa_instances

def prepare_idx_instances(pqa_instances, word2idx):
    pqa_idx_instances = []
    for qp_instance, a_instance in pqa_instances:
        qp_idx_instance = preprocess.convert_instance_to_idx_seq(qp_instance, word2idx)
        a_idx_instance = preprocess.convert_instance_to_idx_seq(a_instance, word2idx)
        pqa_idx_instances.append((qp_idx_instance, a_idx_instance))
    return pqa_idx_instances

def main():
    train_word_instances, train_pqa_instances = prepare_data_instances("./data/squadv2.0/train-v2.0.json")
    dev_word_instances, dev_pqa_instances = prepare_data_instances("./data/squadv2.0/dev-v2.0.json")

    word2idx = preprocess.build_vocab_idx(train_word_instances+dev_word_instances, 5)

    train_pqa_idx_instances = prepare_idx_instances(train_pqa_instances, word2idx)
    dev_pda_idx_instances = prepare_idx_instances(dev_pqa_instances, word2idx)

    data = {
        "train_instances": train_pqa_instances,
        "dev_instances": dev_pqa_instances,
        "train_idx": train_pqa_idx_instances,
        "dev_idx": dev_pda_idx_instances,
        "word2idx": word2idx
    }
    torch.save(data, "./output/squad_idx.data")

    

    




if __name__ == "__main__":
    main()    