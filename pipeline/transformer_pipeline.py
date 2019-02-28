import torch
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from transformer.dataset import TranslationDataset, paired_collate_fn
from transformer.transformer.Models import Transformer
from transformer.transformer.Optim import ScheduledOptim
from transformer.train import train as train_fn

def main():

    parser = argparse.ArgumentParser()

    # parser.add_argument('-data', required=True)

    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-batch_size', type=int, default=64)

    #parser.add_argument('-d_word_vec', type=int, default=512)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=2048)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-n_warmup_steps', type=int, default=4000)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')

    parser.add_argument('-log', default=None)
    parser.add_argument('-save_model', default=None)
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-label_smoothing', action='store_true')
    parser.add_argument('-num_workers', type=int, default=2)

    opt = parser.parse_args()
    opt.cuda = True
    opt.d_word_vec = opt.d_model
    opt.max_token_seq_len = 256

    squad_data_path = "./output/squad_idx.data"
    data = torch.load(squad_data_path)
    print("load data done")

    train_qp_idx, train_a_idx = data["train_idx"]
    dev_qp_idx, dev_a_idx = data["dev_idx"]
    train_src_idx = train_qp_idx
    train_tgt_idx = train_a_idx
    dev_src_idx = dev_qp_idx
    dev_tgt_idx = dev_a_idx
    word2idx = data["word2idx"]
    # import pdb
    # pdb.set_trace()

    src_vocab_size = len(word2idx.keys())
    tgt_vocab_size = len(word2idx.keys())
    # max_token_seq_len = 256

    # proj_share_weight = True
    # embs_share_weight = True

    # d_k = 64
    # d_v = 64
    # d_model = 512
    # d_word_vec = d_model
    # d_inner_hid = 2048
    # n_layers = 1
    # n_head = 8
    # dropout = 0.1

    is_use_cuda = False
    device = torch.device('cuda' if is_use_cuda else 'cpu')
    transformer = Transformer(
        src_vocab_size,
        tgt_vocab_size,
        opt.max_token_seq_len,
        tgt_emb_prj_weight_sharing=opt.proj_share_weight,
        emb_src_tgt_weight_sharing=opt.embs_share_weight,
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout).to(device)
    print("prepare model done")
    
    training_data = torch.utils.data.DataLoader(
        TranslationDataset(
            src_word2idx=word2idx,
            tgt_word2idx=word2idx,
            src_insts=train_src_idx,
            tgt_insts=train_tgt_idx),
        num_workers=opt.num_workers,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn,
        shuffle=True)

    validation_data = torch.utils.data.DataLoader(
        TranslationDataset(
            src_word2idx=word2idx,
            tgt_word2idx=word2idx,
            src_insts=dev_src_idx,
            tgt_insts=dev_tgt_idx),
        num_workers=opt.num_workers,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn)
    print("prepare torch dataloader done")

    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda x: x.requires_grad, transformer.parameters()),
            betas=(0.9, 0.98), eps=1e-09),
        opt.d_model, opt.n_warmup_steps)
    print("begin to train")
    train_fn(transformer, training_data, validation_data, optimizer, device ,opt)


    
    


if __name__ == "__main__":
    main()