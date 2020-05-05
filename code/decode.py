import argparse
import random
import numpy as np
import time
import torch
from torch import optim
# from lf_evaluator import *
from models import *
from data import *
from utils import *
import math
from os.path import join
from gadget import *
import os
import shutil

def _parse_args():
    parser = argparse.ArgumentParser(description='main.py')

    parser.add_argument('dataset', help='specified dataset')
    parser.add_argument('model_id', help='specified model id')
    
    parser.add_argument('--split', type=str, default='test', help='test split')
    parser.add_argument('--do_eval', dest='do_eval', default=False, action='store_true', help='only output')
    # parser.add_argument('--outfile', dest='outfile', default='beam_output.txt', help='output file of beam')
    # Some common arguments for your convenience

    parser.add_argument('--gpu', type=str, default=None, help='gpu id')
    parser.add_argument('--seed', type=int, default=0, help='RNG seed (default = 0)')
    parser.add_argument('--beam_size', type=int, default=20, help='beam size')

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

def make_input_tensor(exs, reverse_input):
    x = np.array(exs.x_indexed)
    len_x = len(exs.x_indexed)
    if reverse_input:
        x = np.array(x[::-1])
    # add batch dim
    x = x[np.newaxis, :]
    len_x = np.array([len_x])
    x = torch.from_numpy(x).long()
    len_x = torch.from_numpy(len_x)
    return x, len_x

def decode(model_path, test_data, input_indexer, output_indexer, args):
    device = config.device
    if 'cpu' in str(device):
        checkpoint = torch.load(model_path, map_location=device)
    else:
        checkpoint = torch.load(model_path)
    
    #  Create model
    model_input_emb = EmbeddingLayer(args.input_dim, len(input_indexer), args.emb_dropout)
    model_enc = RNNEncoder(args.input_dim, args.hidden_size, args.rnn_dropout, args.bidirectional)
    model_output_emb = EmbeddingLayer(args.output_dim, len(output_indexer), args.emb_dropout)
    model_dec = AttnRNNDecoder(args.input_dim, args.hidden_size, 2 * args.hidden_size if args.bidirectional else args.hidden_size,len(output_indexer), args.rnn_dropout)

    # load dict
    model_input_emb.load_state_dict(checkpoint['input_emb'])
    model_enc.load_state_dict(checkpoint['enc'])
    model_output_emb.load_state_dict(checkpoint['output_emb'])
    model_dec.load_state_dict(checkpoint['dec'])

    # map device
    model_input_emb.to(device)
    model_enc.to(device)
    model_output_emb.to(device)
    model_dec.to(device)

    # switch to eval
    model_input_emb.eval()
    model_enc.eval()
    model_output_emb.eval()
    model_dec.eval()

    pred_derivations = []
    with torch.no_grad():
        for i, ex in enumerate(test_data):
            if i % 50 == 0:
                print("Done", i)
            x, len_x = make_input_tensor(ex, args.reverse_input)
            x, len_x = x.to(device), len_x.to(device)

            enc_out_each_word, enc_context_mask, enc_final_states = \
                    encode_input_for_decoder(x, len_x, model_input_emb, model_enc)
            
            pred_derivations.append(beam_decoder(enc_out_each_word, enc_context_mask, enc_final_states,
                output_indexer, model_output_emb, model_dec, args.decoder_len_limit, args.beam_size))


    output_derivations(test_data, pred_derivations, args)

def beam_decoder(enc_out_each_word, enc_context_mask, enc_final_states, output_indexer,
                    model_output_emb, model_dec, decoder_len_limit, beam_size):
    ders, scores = batched_beam_sampling(enc_out_each_word, enc_context_mask, enc_final_states, output_indexer,
                    model_output_emb, model_dec, decoder_len_limit, beam_size)
    pred_tokens = [[output_indexer.get_object(t) for t in y] for y in ders]
    return pred_tokens

def output_derivations(test_data, pred_derivations, args):
    outfile = get_decode_file(args.dataset, args.split, args.model_id)
    with open(outfile, "w") as out:
        for i, pred_ders in enumerate(pred_derivations):
            out.write(" ".join(["".join(x[1]) for x in enumerate(pred_ders)]) + "\n")

if __name__ == '__main__':
    args = _parse_args()
    print(args)
    # global device
    set_global_device(args.gpu)
    
    print("Pytroch using device ", config.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    # Load the training and test data
    test, input_indexer, output_indexer = load_test_dataset(args.dataset, args.split)
    test_data_indexed = index_data(test, input_indexer, output_indexer, args.decoder_len_limit)
    # test_data_indexed = tricky_filter_data(test_data_indexed)

    model_path = get_model_file(args.dataset, args.model_id)
    decode(model_path, test_data_indexed, input_indexer, output_indexer, args)