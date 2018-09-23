import argparse

import torch
import torch.nn as nn

from data_loader import DataLoader
import data_loader
from language_model import LanguageModel as LM
import trainer


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('-model', required=True)
    p.add_argument('-train', required=True)
    p.add_argument('-valid', required=True)
    p.add_argument('-gpu_id', type=int, default=-1)

    p.add_argument('-batch_size', type=int, default=64)
    p.add_argument('-n_epochs', type=int, default=20)
    p.add_argument('-print_every', type=int, default=50)
    p.add_argument('-early_stop', type=int, default=3)
    p.add_argument('-iter_ratio_in_epoch', type=float, default=1.)
    p.add_argument('-lr_decay_rate', type=float, default=.5)
    
    p.add_argument('-dropout', type=float, default=.3)
    p.add_argument('-word_vec_dim', type=int, default=256)
    p.add_argument('-hidden_size', type=int, default=256)
    p.add_argument('-max_length', type=int, default=80)

    p.add_argument('-n_layers', type=int, default=4)
    p.add_argument('-max_grad_norm', type=float, default=5.)
    p.add_argument('-lr', type=float, default=1.)
    p.add_argument('-min_lr', type=float, default=.000001)

    p.add_argument('-gen', type=int, default=32)
    
    config = p.parse_args()

    return config


def to_text(indice, vocab):
    lines = []

    for i in range(len(indice)):
        line = []
        for j in range(len(indice[i])):
            index = indice[i][j]

            if index == data_loader.EOS:
                # line += ['<EOS>']
                break
            else:
                line += [vocab.itos[index]]

        line = ' '.join(line)
        lines += [line]

    return lines


if __name__ == '__main__':
    config = define_argparser()

    loader = DataLoader(config.train, 
                        config.valid, 
                        batch_size=config.batch_size, 
                        device=config.gpu_id,
                        max_length=config.max_length
                        )
    model = LM(len(loader.text.vocab), 
               word_vec_dim=config.word_vec_dim, 
               hidden_size=config.hidden_size, 
               n_layers=config.n_layers, 
               dropout_p=config.dropout, 
               max_length=config.max_length
               )

    # Let criterion cannot count PAD as right prediction, because PAD is easy to predict.
    loss_weight = torch.ones(len(loader.text.vocab))
    loss_weight[data_loader.PAD] = 0
    criterion = nn.NLLLoss(weight=loss_weight, size_average=False)

    print(model)
    print(criterion)

    if config.gpu_id >= 0:
        model.cuda(config.gpu_id)
        criterion.cuda(config.gpu_id)

    if config.n_epochs > 0:
        trainer.train_epoch(model, 
                            criterion, 
                            loader.train_iter, 
                            loader.valid_iter, 
                            config
                            )

    if config.gen > 0:
        total_gen = 0
        while total_gen < config.gen:
            current_gen = min(config.batch_size, config.gen - total_gen)
            _, indice = model.search(batch_size=current_gen)
            total_gen += current_gen

            lines = to_text(indice, loader.text.vocab)
            print('\n'.join(lines))
