from data import *
from os.path import join
import numpy as np
import argparse
from tqdm import tqdm
import sys
import subprocess
import os

def _parse_args():
    parser = argparse.ArgumentParser(description='main.py')

    parser.add_argument('dataset', help='specified dataset')
    parser.add_argument('decodes_file', help='specified decodes file')
    parser.add_argument('--split', type=str, default='test', help='test split')
    parser.add_argument('--decoder_len_limit', type=int, default=170, help='output length limit of the decoder')

    args = parser.parse_args()
    return args

def read_derivations(decode_file):
    with open(decode_file) as f:
        lines = f.readlines()
        lines = [x.rstrip() for x in lines]
        lines = [x.split(" ") for x in lines]

    return lines

def inverse_regex_with_map(r, maps):
    for m in maps:
        src = m[0]
        if len(m[1]) == 1:
            dst = "<{}>".format(m[1])
        else:
            dst = "const(<{}>)".format(m[1])
        r = r.replace(src, dst)
    return r

def external_evaluation(gt_spec, preds, exs, flag_force=False):
    pred_line = " ".join(preds)
    exs_line = " ".join(["{},{}".format(x[0], x[1]) for x in exs])
    flag_str = "true" if flag_force else "false"

    flag_use_file = len(pred_line) > 200
    if flag_use_file:
        filename = join("./external/", "tmp.in")
        with open(filename, "w") as f:
            f.write(pred_line + "\n")
            f.write(exs_line + "\n")
            f.write(gt_spec)
        out = subprocess.check_output(
            ['java', '-cp', './external/datagen.jar:./external/lib/*', '-ea', 'datagen.Main', 'evaluate_single_file',
                filename, flag_str], stderr=subprocess.DEVNULL)
        os.remove(filename)
    else:
        out = subprocess.check_output(
            ['java', '-cp', './external/datagen.jar:./external/lib/*', '-ea', 'datagen.Main', 'evaluate_single',
                pred_line, exs_line, gt_spec, flag_str], stderr=subprocess.DEVNULL)

    out = out.decode("utf-8")
    out = out.rstrip()
    vals = out.split(" ")
    return vals[0], vals[1:]

def filtering_test(gt, preds, m, exs, flag_force=False):
    gt = gt.replace(" ", "")
    gt = inverse_regex_with_map(gt, m)
    preds = [inverse_regex_with_map(x, m) for x in preds]
    global_res, pred_res = external_evaluation(gt, preds, exs, flag_force)
    if global_res in ["exact", "equiv"]:
        return True, global_res, pred_res
    else:
        return False, global_res, pred_res

if __name__ == "__main__":
    args = _parse_args()
    print(args)
    test, input_indexer, output_indexer = load_test_dataset(args.dataset, args.split)

    const_maps = load_const_maps(args.dataset, args.split)
    exs_lists = load_exs(args.dataset, args.split)

    decode_file = get_decode_file(args.dataset, args.split, args.decodes_file)
    pred_derivations = read_derivations(decode_file)

    cnt = 0
    results = []
    for (_,gt), p, m, exs in tqdm(zip(test, pred_derivations, const_maps, exs_lists), desc='eval', file=sys.stdout, total=len(test)):
        # print(gt)
        match_result = filtering_test(gt, p, m, exs, flag_force=True)
        # print(match_result[0])
        results.append(match_result)
    
    # Top 0 DFA Acc
    num_top0_correct = sum([x[2][0] in ["exact", "equiv"] for x in results])
    print('Top-1 Derivation DFA ACC', num_top0_correct * 1. / len(results))  
    # Filter Acc
    num_correct_after_filtering = sum([x[0] for x in results])
    print('ACC with Filtering', num_correct_after_filtering * 1. / len(results))
