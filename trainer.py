import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils as torch_utils

import utils


def get_loss(y, y_hat, criterion, do_backward=True):
    batch_size = y.size(0)

    loss = criterion(y_hat.contiguous().view(-1, y_hat.size(-1)), 
                     y.contiguous().view(-1)
                     )
    if do_backward:
        # since size_average parameter is off, we need to devide it by batch size before back-prop.
        loss.div(batch_size).backward()

    return loss


def train_epoch(model, criterion, train_iter, valid_iter, config):
    current_lr = config.lr

    lowest_valid_loss = np.inf
    no_improve_cnt = 0

    for epoch in range(1, config.n_epochs + 1):
        # optimizer = optim.Adam(model.parameters(), lr = current_lr)
        optimizer = optim.SGD(model.parameters(),
                              lr=current_lr
                              )
        print("current learning rate: %f" % current_lr)
        print(optimizer)

        sample_cnt = 0
        total_loss, total_word_count, total_parameter_norm, total_grad_norm = 0, 0, 0, 0
        start_time = time.time()
        train_loss = np.inf

        for batch_index, batch in enumerate(train_iter):
            optimizer.zero_grad()

            current_batch_word_cnt = torch.sum(batch.text[1])
            # Most important lines in this method.
            # Since model takes BOS + sentence as an input and sentence + EOS as an output,
            # x(input) excludes last index, and y(index) excludes first index.
            x = batch.text[0][:, :-1]
            y = batch.text[0][:, 1:]
            # feed-forward
            y_hat = model(x)

            # calcuate loss and gradients with back-propagation
            loss = get_loss(y, y_hat, criterion)
            
            # simple math to show stats
            total_loss += float(loss)
            total_word_count += int(current_batch_word_cnt)
            total_parameter_norm += float(utils.get_parameter_norm(model.parameters()))
            total_grad_norm += float(utils.get_grad_norm(model.parameters()))

            if (batch_index + 1) % config.print_every == 0:
                avg_loss = total_loss / total_word_count
                avg_parameter_norm = total_parameter_norm / config.print_every
                avg_grad_norm = total_grad_norm / config.print_every
                elapsed_time = time.time() - start_time

                print("epoch: %d batch: %d/%d\t|param|: %.2f\t|g_param|: %.2f\tloss: %.4f\tPPL: %.2f\t%5d words/s %3d secs" % (epoch, 
                                                                                                                               batch_index + 1, 
                                                                                                                               int((len(train_iter.dataset.examples) // config.batch_size)  * config.iter_ratio_in_epoch), 
                                                                                                                               avg_parameter_norm, 
                                                                                                                               avg_grad_norm, 
                                                                                                                               avg_loss,
                                                                                                                               np.exp(avg_loss),
                                                                                                                               total_word_count // elapsed_time,
                                                                                                                               elapsed_time
                                                                                                                               ))

                total_loss, total_word_count, total_parameter_norm, total_grad_norm = 0, 0, 0, 0
                start_time = time.time()

                train_loss = avg_loss

            # Another important line in this method.
            # In orther to avoid gradient exploding, we apply gradient clipping.
            torch_utils.clip_grad_norm_(model.parameters(), 
                                        config.max_grad_norm
                                        )
            # Take a step of gradient descent.
            optimizer.step()

            sample_cnt += batch.text[0].size(0)
            if sample_cnt >= len(train_iter.dataset.examples) * config.iter_ratio_in_epoch:
                break

        sample_cnt = 0
        total_loss, total_word_count = 0, 0

        model.eval()
        for batch_index, batch in enumerate(valid_iter):
            current_batch_word_cnt = torch.sum(batch.text[1])
            x = batch.text[0][:, :-1]
            y = batch.text[0][:, 1:]
            y_hat = model(x)

            loss = get_loss(y, y_hat, criterion, do_backward=False)

            total_loss += float(loss)
            total_word_count += int(current_batch_word_cnt)

            sample_cnt += batch.text[0].size(0)
            if sample_cnt >= len(valid_iter.dataset.examples):
                break

        avg_loss = total_loss / total_word_count
        print("valid loss: %.4f\tPPL: %.2f" % (avg_loss, np.exp(avg_loss)))

        if lowest_valid_loss > avg_loss:
            lowest_valid_loss = avg_loss
            no_improve_cnt = 0
        else:
            # decrease learing rate if there is no improvement.
            current_lr = max(config.min_lr, current_lr * config.lr_decay_rate)
            no_improve_cnt += 1

        model.train()

        model_fn = config.model.split(".")
        model_fn = model_fn[:-1] + ["%02d" % epoch, 
                                    "%.2f-%.2f" % (train_loss, np.exp(train_loss)), 
                                    "%.2f-%.2f" % (avg_loss, np.exp(avg_loss))
                                    ] + [model_fn[-1]]

        # PyTorch provides efficient method for save and load model, which uses python pickle.
        torch.save({"model": model.state_dict(),
                    "config": config,
                    "epoch": epoch + 1,
                    "current_lr": current_lr
                    }, ".".join(model_fn))

        if config.early_stop > 0 and no_improve_cnt >= config.early_stop:
            break
