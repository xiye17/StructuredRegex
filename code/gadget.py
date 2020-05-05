
import torch
import sys
from torch import optim
# from lf_evaluator import *
from models import *
from data import *
from utils import *

class config():
    device = None

def set_global_device(gpu):
    if gpu is not None:
        config.device = torch.device(('cuda:' + gpu) if torch.cuda.is_available() else 'cpu')
    else:
        config.device = 'cpu'
    
# Analogous to make_padded_input_tensor, but without the option to reverse input
def make_padded_output_tensor(exs, output_indexer, max_len):
    return np.array([[ex.y_indexed[i] if i < len(ex.y_indexed) else output_indexer.index_of(PAD_SYMBOL) for i in range(0, max_len)] for ex in exs])

# Takes the given Examples and their input indexer and turns them into a numpy array by padding them out to max_len.
# Optionally reverses them.
def make_padded_input_tensor(exs, input_indexer, max_len, reverse_input):
    if reverse_input:
        return np.array(
            [[ex.x_indexed[len(ex.x_indexed) - 1 - i] if i < len(ex.x_indexed) else input_indexer.index_of(PAD_SYMBOL)
              for i in range(0, max_len)]
             for ex in exs])
    else:
        return np.array([[ex.x_indexed[i] if i < len(ex.x_indexed) else input_indexer.index_of(PAD_SYMBOL)
                          for i in range(0, max_len)]
                         for ex in exs])

def masked_cross_entropy(voc_scores, gt, mask):
    corss_entropy = -torch.log(torch.gather(voc_scores, 1, gt.view(-1, 1)))
    loss = corss_entropy.squeeze(1).masked_select(mask).sum()
    return loss

def encode_input_for_decoder(x_tensor, inp_lens_tensor, model_input_emb, model_enc):
    input_emb = model_input_emb.forward(x_tensor)
    (enc_output_each_word, enc_context_mask, enc_final_states) = model_enc.forward(input_emb, inp_lens_tensor)
    # print(enc_output_each_word.size(), enc_context_mask.size())
    enc_final_states_reshaped = (enc_final_states[0].unsqueeze(0), enc_final_states[1].unsqueeze(0))
    return (enc_output_each_word, enc_context_mask, enc_final_states_reshaped)

def batched_beam_sampling(enc_out_each_word, enc_context_mask, enc_final_states, output_indexer,
                    model_output_emb, model_dec, decoder_len_limit, beam_size):
    device = config.device
    EOS = output_indexer.get_index(EOS_SYMBOL)
    context_inf_mask = get_inf_mask(enc_context_mask)

    completed = []
    cur_beam = [([], .0, .0)]
    # 0 toks, 1 score
    input_words = torch.LongTensor([[output_indexer.index_of(SOS_SYMBOL)]]).to(device)
    input_states = enc_final_states
    for _ in range(decoder_len_limit):

        # input_words = torch.LongTensor([[x[1] for x in cur_beam]]).to(device)
        input_embeded_words = model_output_emb.forward(input_words)

        batch_voc_scores, batch_next_states = model_dec(input_embeded_words, input_states, enc_out_each_word, context_inf_mask)
        batch_voc_scores = torch.log(batch_voc_scores)
        batch_voc_scores_cpu = batch_voc_scores.tolist()

        next_beam = []
        action_pool = []
        for b_id, voc_scores in enumerate(batch_voc_scores_cpu):
            base_score = cur_beam[b_id][1]
            for voc_id, score_cpu in enumerate(voc_scores):
                # next_beam.append()
                action_pool.append((b_id, voc_id, base_score + score_cpu, True))
        
        for b_id, (_, score, _) in enumerate(completed):
            action_pool.append((b_id, 0, score, False ))
        
        action_pool.sort(key=lambda x: x[2], reverse=True)
        kept_b_id = []
        next_input_words = []
        next_completed = []
        for b_id, voc_id, new_score, is_gen in action_pool[:beam_size]:
            if is_gen:
                if voc_id == EOS:
                    next_completed.append((cur_beam[b_id][0], new_score, cur_beam[b_id][2] + batch_voc_scores[b_id][voc_id]))
                else:
                    next_beam.append((cur_beam[b_id][0] + [voc_id], new_score, cur_beam[b_id][2] + batch_voc_scores[b_id][voc_id]))
                    next_input_words.append(voc_id)
                    kept_b_id.append(b_id)
            else:
                next_completed.append(completed[b_id])
        completed = next_completed
        if not next_beam:
            break
        kept_b_id = torch.LongTensor(kept_b_id).to(device)
        input_words = torch.LongTensor([next_input_words]).to(device)
        cur_beam = next_beam
        input_states = batch_next_states[0].index_select(1, kept_b_id), batch_next_states[1].index_select(1, kept_b_id)

    completed.sort(key=lambda x: x[1], reverse=True)
    ders = [x[0] for x in completed]
    sum_probs = [x[2] for x in completed]
    
    # print('---------------')
    # pred_tokens = [[output_indexer.get_object(t) for t in y] for y in ders]
    # [print(''.join(p)) for p in pred_tokens]
    return ders, sum_probs
