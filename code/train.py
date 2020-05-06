import argparse
import random
import sys
import numpy as np
import time
import torch
from torch import optim
from gadget import *
from models import *
from data import *
from utils import *
import math

def _parse_args():
    parser = argparse.ArgumentParser(description='main.py')

    parser.add_argument('dataset', help='specified dataset')
    # General system running and configuration options

    # Some common arguments for your convenience
    parser.add_argument("--gpu", type=str, default="0", help="gpu id")
    parser.add_argument('--seed', type=int, default=0, help='RNG seed (default = 0)')
    parser.add_argument('--epochs', type=int, default=100, help='num epochs to train for')
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--clip_grad', type=float, default=10.0)

    # regarding model saving
    parser.add_argument('--model_id', type=str, default=None, help='model identifier')
    parser.add_argument('--saving_from', type=int, default=50, help='saving from - epoch')
    parser.add_argument('--saving_interval', type=int, default=10, help='saving iterval')

    # 65 is all you need for GeoQuery
    parser.add_argument('--decoder_len_limit', type=int, default=170, help='output length limit of the decoder')
    parser.add_argument('--input_dim', type=int, default=100, help='input vector dimensionality')
    parser.add_argument('--output_dim', type=int, default=100, help='output vector dimensionality')
    parser.add_argument('--hidden_size', type=int, default=200, help='hidden state dimensionality')

    # Hyperparameters for the encoder -- feel free to play around with these!
    parser.add_argument('--no_bidirectional', dest='bidirectional', default=True, action='store_false', help='bidirectional LSTM')
    parser.add_argument('--reverse_input', dest='reverse_input', default=False, action='store_true')
    parser.add_argument('--emb_dropout', type=float, default=0.2, help='input dropout rate')
    parser.add_argument('--rnn_dropout', type=float, default=0.2, help='dropout rate internal to encoder RNN')

    args = parser.parse_args()
    return args


def train_decode_with_output_of_encoder(enc_out_each_word, enc_context_mask,
                            enc_final_states, output_indexer, gt_out, gt_out_lens,
                            model_output_emb, model_dec, decoder_len_limit, p_forcing):
    batch_size = enc_context_mask.size(0)
    context_inf_mask = get_inf_mask(enc_context_mask)
    input_words = torch.from_numpy(np.asarray([output_indexer.index_of(SOS_SYMBOL) for _ in range(batch_size)]))
    input_words = input_words.to(config.device)
    input_words = input_words.unsqueeze(1)
    dec_hidden_states = enc_final_states

    gt_out_mask = sent_lens_to_mask(gt_out_lens, gt_out.size(1))
    output_max_len = torch.max(gt_out_lens).item()

    using_teacher_forcing = np.random.uniform() < p_forcing
    loss = 0

    if using_teacher_forcing:
        for i in range(output_max_len):
            input_embeded_words = model_output_emb.forward(input_words)
            input_embeded_words = input_embeded_words.reshape((1, batch_size, -1))
            voc_scores, dec_hidden_states = model_dec(input_embeded_words, dec_hidden_states, enc_out_each_word, context_inf_mask)
            input_words = gt_out[:, i].view((-1, 1))

            loss += masked_cross_entropy(voc_scores, gt_out[:, i], gt_out_mask[:, i])

    else:
        for i in range(output_max_len):
            input_embeded_words = model_output_emb.forward(input_words)
            input_embeded_words = input_embeded_words.reshape((1, batch_size, -1))
            voc_scores, dec_hidden_states = model_dec(input_embeded_words, dec_hidden_states, enc_out_each_word, context_inf_mask)
            output_words = voc_scores.argmax(dim=1, keepdim=True)
            input_words = output_words.detach()
            loss += masked_cross_entropy(voc_scores, gt_out[:, i], gt_out_mask[:, i])

    num_entry = gt_out_lens.sum().float().item()
    loss = loss / num_entry
    return loss, num_entry

def model_perplexity(test_loader,
                    model_input_emb, model_enc, model_output_emb, model_dec,
                    input_indexer, output_indexer, args):
    device = config.device
    model_input_emb.eval()
    model_enc.eval()
    model_output_emb.eval()
    model_dec.eval()

    test_iter = iter(test_loader)
    epoch_loss = 0.0
    epoch_num_entry = 0.0

    with torch.no_grad():
        for _, batch_data in enumerate(test_iter):
            batch_in, batch_in_lens, batch_out, batch_out_lens = batch_data
            batch_in, batch_in_lens, batch_out, batch_out_lens = \
                batch_in.to(device), batch_in_lens.to(device), batch_out.to(device), batch_out_lens.to(device)

            enc_out_each_word, enc_context_mask, enc_final_states = \
                encode_input_for_decoder(batch_in, batch_in_lens, model_input_emb, model_enc)

            loss, num_entry = \
                train_decode_with_output_of_encoder(enc_out_each_word, enc_context_mask, enc_final_states, output_indexer,
                batch_out, batch_out_lens, model_output_emb, model_dec, args.decoder_len_limit, 1)
            epoch_loss += (loss.item() * num_entry)
            epoch_num_entry += num_entry
        perperlexity = epoch_loss / epoch_num_entry
    return perperlexity

def train_model_encdec_ml(train_data, test_data, input_indexer, output_indexer, args):
    device = config.device
    # Sort in descending order by x_indexed, essential for pack_padded_sequence
    train_data.sort(key=lambda ex: len(ex.x_indexed), reverse=True)
    test_data.sort(key=lambda ex: len(ex.x_indexed), reverse=True)

    # Create indexed input
    train_input_max_len = np.max(np.asarray([len(ex.x_indexed) for ex in train_data]))
    test_input_max_len = np.max(np.asarray([len(ex.x_indexed) for ex in test_data]))
    input_max_len = max(train_input_max_len, test_input_max_len)
    # input_max_len = 100

    all_train_input_data = make_padded_input_tensor(train_data, input_indexer, input_max_len, args.reverse_input)
    all_test_input_data = make_padded_input_tensor(test_data, input_indexer, input_max_len, args.reverse_input)

    train_output_max_len = np.max(np.asarray([len(ex.y_indexed) for ex in train_data]))
    test_output_max_len = np.max(np.asarray([len(ex.y_indexed) for ex in test_data]))
    output_max_len = max(train_output_max_len, test_output_max_len)
    all_train_output_data = make_padded_output_tensor(train_data, output_indexer, output_max_len)
    all_test_output_data = make_padded_output_tensor(test_data, output_indexer,  np.max(np.asarray([len(ex.y_indexed) for ex in test_data])) )
    all_test_output_data = np.maximum(all_test_output_data, 0)

    print("Train length: %i" % input_max_len)
    print("Train output length: %i" % np.max(np.asarray([len(ex.y_indexed) for ex in train_data])))
    print("Train matrix: %s; shape = %s" % (all_train_input_data, all_train_input_data.shape))

    # Create model
    model_input_emb = EmbeddingLayer(args.input_dim, len(input_indexer), args.emb_dropout)
    model_enc = RNNEncoder(args.input_dim, args.hidden_size, args.rnn_dropout, args.bidirectional)
    model_output_emb = EmbeddingLayer(args.output_dim, len(output_indexer), args.emb_dropout)
    model_dec = AttnRNNDecoder(args.input_dim, args.hidden_size, 2 * args.hidden_size if args.bidirectional else args.hidden_size,len(output_indexer), args.rnn_dropout)

    model_input_emb.to(device)
    model_enc.to(device)
    model_output_emb.to(device)
    model_dec.to(device)

    # Loop over epochs, loop over examples, given some indexed words, call encode_input_for_decoder, then call your
    # decoder, accumulate losses, update parameters

    # optimizer = None
    train_loader = BatchDataLoader(train_data, all_train_input_data, all_train_output_data, batch_size=args.batch_size, shuffle=True)
    test_loader = BatchDataLoader(test_data, all_test_input_data, all_test_output_data, batch_size=args.batch_size, shuffle=False)

    train_iter = iter(train_loader)

    optimizer = optim.Adam([
        {'params': model_input_emb.parameters()},
        {'params': model_enc.parameters()},
        {'params': model_output_emb.parameters()},
        {'params': model_dec.parameters()}], lr=0.001)

    get_teaching_forcing_ratio = lambda x: 1.0
    clip = args.clip_grad

    best_dev_perplexity = np.inf
    for epoch in range(1, args.epochs + 1):

        model_input_emb.train()
        model_enc.train()
        model_output_emb.train()
        model_dec.train()

        print('epoch {}'.format(epoch))
        epoch_loss = 0.0
        epoch_num_entry = 0.0
        for batch_idx, batch_data in enumerate(train_iter):

            optimizer.zero_grad()

            batch_in, batch_in_lens, batch_out, batch_out_lens = batch_data
            batch_in, batch_in_lens, batch_out, batch_out_lens = \
                batch_in.to(device), batch_in_lens.to(device), batch_out.to(device), batch_out_lens.to(device)

            enc_out_each_word, enc_context_mask, enc_final_states = \
                encode_input_for_decoder(batch_in, batch_in_lens, model_input_emb, model_enc)

            tf_ratio = get_teaching_forcing_ratio(epoch)
            loss, num_entry = \
                train_decode_with_output_of_encoder(enc_out_each_word, enc_context_mask, enc_final_states, output_indexer,
                batch_out, batch_out_lens, model_output_emb, model_dec, args.decoder_len_limit, tf_ratio)

            loss.backward()
            epoch_loss += (loss.item() * num_entry)
            epoch_num_entry += num_entry
            # print('epoch loss', epoch_loss, 'epoch entry', epoch_num_entry)
            _ = torch.nn.utils.clip_grad_norm_(model_input_emb.parameters(), clip)
            _ = torch.nn.utils.clip_grad_norm_(model_enc.parameters(), clip)
            _ = torch.nn.utils.clip_grad_norm_(model_output_emb.parameters(), clip)
            _ = torch.nn.utils.clip_grad_norm_(model_dec.parameters(), clip)
            optimizer.step()

        print('epoch {} tf: {} train loss: {}'.format(epoch, tf_ratio, epoch_loss / epoch_num_entry))

        if (epoch < args.saving_from) or (args.model_id is None):
            continue

        # start saving
        dev_perplexity = model_perplexity(test_loader, model_input_emb, model_enc, model_output_emb, model_dec, input_indexer, output_indexer, args)
        print('epoch {} tf: {} dev loss: {}'.format(epoch, tf_ratio, dev_perplexity))

        if dev_perplexity < best_dev_perplexity:
            parameters = {'input_emb': model_input_emb.state_dict(), 'enc': model_enc.state_dict(),
                'output_emb': model_output_emb.state_dict(), 'dec': model_dec.state_dict()}
            best_dev_perplexity = dev_perplexity
            torch.save(parameters, get_model_file(args.dataset, args.model_id + "-best"))

        if (epoch - args.saving_from) % args.saving_interval == 0:
            parameters = {'input_emb': model_input_emb.state_dict(), 'enc': model_enc.state_dict(),
                'output_emb': model_output_emb.state_dict(), 'dec': model_dec.state_dict()}
            torch.save(parameters, get_model_file(args.dataset, args.model_id + "-" + str(epoch)))

if __name__ == '__main__':
    args = _parse_args()
    print(args)
    # global device
    set_global_device(args.gpu)

    print("Pytroch using device ", config.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    # Load the training and test data
    train, dev, input_indexer, output_indexer = load_datasets(args.dataset)
    train_data_indexed, dev_data_indexed = index_datasets(train, dev, input_indexer, output_indexer, args.decoder_len_limit)

    train_model_encdec_ml(train_data_indexed, dev_data_indexed, input_indexer, output_indexer, args)
