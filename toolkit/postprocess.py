from regex_io import *
from os.path import join
from functools import reduce
from base import *
from template import *
import random
import re
import nltk
from nltk.translate import IBMModel1
from nltk.translate import Alignment, AlignedSent
from nltk.translate.ibm3 import IBMModel3
from nltk.translate.ibm2 import IBMModel2
from prepare_regex_data import gen_bad_examples
import numpy as np
import spacy
import pickle

# postprocess responses
def filter_responces(resps):
    resps = [x for x in resps if len(x[4]["description"].split()) > 4]
    return resps

def mannually_filter_responces(resps):
    # filter by worker id
    bad_workers = ["A3HM4LUQL1ASJ1", "A2C33799VENPIP", "A2V0BI2FTB17HN"]
    print("Before M", len(resps))
    resps = [x for x in resps if x[4]["worker_id"] not in bad_workers]
    print("After M", len(resps))
    return resps

def build_reg_libs():
    regexes_b1 = read_tsv_file(join("./results", "batch1_record.txt"), delimiter=" ")
    regexes_b2 = read_tsv_file(join("./results", "batch2_record.txt"), delimiter=" ")
    regexes_b3 = read_tsv_file(join("./results", "batch3_record.txt"), delimiter=" ")
    regexes = regexes_b1 + regexes_b2 + regexes_b3
    regexes = dict([(x[0],build_func_from_str(x[1])) for x in regexes])
    return regexes

def extract_const_comps(node):
    if isinstance(node, SingleToken):
        return [("tok", node.tok)]
    elif isinstance(node, StringToken):
        return [("str", node.tok)]
    else:
        return reduce(lambda x,y: x + y, [extract_const_comps(x) for x in node.children], [])

def extract_ints(node):
    toks = node.ground_truth()
    toks = tokenize(toks)
    toks = [x for x in toks if x.isdigit()]
    return toks

# "a" or "." or "," -> unique mode
# digit -> unique 
def map_tokens_with_consts(tokens, consts, ints):
    maps = []
    tokens_bak = tokens[:]
    quoated_tokens = []
    symbol_tokens = []
    for const in consts:
        if const[0] == "tok":
            if const[1] == "a" or const[1] == "." or const[1] == ",":
                continue
            if const[1].isdigit():
                # overlapping
                if const[1] in ints:
                    continue

        target = const[1]
        quoated_tokens = []
        for i, tok in enumerate(tokens):
            if tok == target:
                quoated_tokens.append('"{}"'.format(target))
            else:
                quoated_tokens.append(tok)
        tokens = quoated_tokens
    if not quoated_tokens:
        quoated_tokens = tokens[:]

    c_id = 0
    for const in consts:
        target = const[1]
        symbol = "const{}".format(c_id)
        symbol_tokens = []
        flag_mapped = False
        for i, tok in enumerate(quoated_tokens):
            if tok == '"{}"'.format(target):
                symbol_tokens.append(symbol)
                flag_mapped = True
            else:
                symbol_tokens.append(tok)
        if flag_mapped:
            maps.append((symbol, target))
            c_id += 1
        quoated_tokens = symbol_tokens
    
    if not symbol_tokens:
        symbol_tokens = quoated_tokens[:]
    # leftout free consts
    free_consts = []
    for tok in symbol_tokens:
        if tok[0] == '"' and tok[-1] == '"' and len(tok) > 2:
            if tok[1:-1] not in free_consts:
                free_consts.append(tok[1:-1])
    map_before = maps[:]
    symbol_descp = " ".join(symbol_tokens)
    for fcons in free_consts:
        symbol = "const{}".format(c_id)
        symbol_descp = symbol_descp.replace('"{}"'.format(fcons), symbol)
        maps.append((symbol, fcons))
        c_id += 1
    symbol_tokens = symbol_descp.split(" ")
    return symbol_tokens, maps
    
def remove_duplicate_const(consts):
    y = []
    for c in consts:
        if c not in y:
            y.append(c)
    
    return y

def special_processing(descp):
    new_tokens = []
    tokens = descp.split(" ")
    a_to_b_re = re.compile("\d+-\d+")
    a_or_more_re = re.compile("\d+\+")
    for tok in tokens:
        if a_to_b_re.match(tok):
            nums = tok.split("-")
            new_tokens.append(nums[0])
            new_tokens.append("-")
            new_tokens.append(nums[1])
            continue
        if a_or_more_re.match(tok):
            nums = tok[:-1]
            new_tokens.append(nums)
            new_tokens.append("+")
            continue
        new_tokens.append(tok)
    
    # clear . again
    new_descp = " ".join(new_tokens)
    new_descp = new_descp.replace(".", " . ")
    new_tokens = new_descp.split(" ")
    new_tokens = [x for x in new_tokens if len(x)]

    return " ".join(new_tokens)

def replace_regex_with_map(r, maps):
    for m in maps:
        if len(m[1]) == 1:
            src = "<{}>".format(m[1])
        else:
            src = "const(<{}>)".format(m[1])
        dst = m[0]
        r = r.replace(src, dst)
    return r

_CONTRACTIONS2 = [
        r"(?i)\b(can)(?#X)(not)\b",
        r"(?i)\b(gon)(?#X)(na)\b",
        r"(?i)\b(got)(?#X)(ta)\b",
        r"(?i)\b(wan)(?#X)(na)b",
        r"(?i)\b(there)(?#X)('s)b",
        r"(?i)\b(it)(?#X)('s)b",
    ]
_CONTRACTIONS3 = [r"(?i)\b(\w+)(?#X)(n't)\b"]
CONTRACTIONS2 = list(map(re.compile, _CONTRACTIONS2))
CONTRACTIONS3 = list(map(re.compile, _CONTRACTIONS3))

def remove_contractions(x):
    for regexp in CONTRACTIONS2:
        x = regexp.sub(r'\1 \2', x)
    x = x.replace("can't", "can not")
    for regexp in CONTRACTIONS3:
        x = regexp.sub(r'\1 not', x)
    # x = x.replace("can't", "can n‘t")
    # for regexp in CONTRACTIONS3:
    #     x = regexp.sub(r'\1 \2', x)
    return x


# like 3 a's or 3 "a"s
def clear_prural(tokens):
    new_toks = []
    for t in tokens:
        if t.endswith("'s") and t != "'s":
            new_toks.append(t[:-2])
        elif t.endswith('"s'):
            new_toks.append(t[:-1])
        else:
            new_toks.append(t)
    return new_toks

# fix unpaired token
def fix_unpaired_quotas(tokens):
    s = " ".join(tokens)
    double_quot_regex = re.compile("\"([^'\s]+)\"")
    s = re.sub(double_quot_regex, r' "\1" ', s)

    new_toks = []
    for t in s.split(" "):
        if "'" in t or '"' in t:
            if t == "'s" or t == "n't":
                new_toks.append(t)
                continue
            if t[0] == '"' and t[-1] == '"':
                new_toks.append(t)
                continue

            t = t.replace("'", " ")
            t = t.replace('"', " ")
            new_toks.extend(t.split(" "))
        else:
            new_toks.append(t)
    new_toks = [x for x in new_toks if len(x)]
    return new_toks

def spacy_processing(descp):
    doc = spacy_tokenizer(descp)
    return " ".join([x.text for x in doc])

def record_to_metadata(r, reg_libs):
    # print(r["problem_id"])
    descp = r["description"]
    id = r["problem_id"]
    regex = reg_libs[id]    
    consts = extract_const_comps(regex)
    consts = remove_duplicate_const(consts)
    ints = extract_ints(regex)
    # deal with , .
    descp_clear_punc = descp + " "
    descp_clear_punc = descp_clear_punc.replace(". ", " . ")
    descp_clear_punc = descp_clear_punc.replace(", ", " , ")
    descp_clear_punc = descp_clear_punc.replace("(", " ( ")
    descp_clear_punc = descp_clear_punc.replace(")", " ) ")
    descp_clear_punc = descp_clear_punc.rstrip()

    descp_clear_punc = remove_contractions(descp_clear_punc)
    # fix minor issue    
    descp_clear_punc = descp_clear_punc.replace("’", "'")
    single_quot_regex = re.compile("'([^'\s]+)'")
    descp_clear_single = re.sub(single_quot_regex, r'"\1"', descp_clear_punc)

    # clear prural
    descp_tokens = descp_clear_single.split(" ")
    descp_tokens = [x for x in descp_tokens if len(x)]
    descp_tokens = clear_prural(descp_tokens)

    # fix unpaired quota
    descp_tokens = fix_unpaired_quotas(descp_tokens)

    # map const, if not quoated, quote them
    symbol_tokens, token_const_maps = map_tokens_with_consts(descp_tokens, consts, ints)

    symbol_descp = " ".join(symbol_tokens)
    symbol_descp = symbol_descp.lower()

    num_pairs = [(' one ', ' 1 '), (' two ', ' 2 '), (' three ', ' 3 '), (' four ', ' 4 '),
        (' five ', ' 5 '), (' six ', ' 6 '), (' seven ', ' 7 '), (' eight ', ' 8 '), (' nine ', ' 9 '), (' ten ', ' 10 ')]
    for pair in num_pairs:
        symbol_descp = symbol_descp.replace(pair[0], pair[1])
    symbol_descp = special_processing(symbol_descp)
    symbol_descp = spacy_processing(symbol_descp)
    # dealwith groudtruth
    ground_truth = regex.ground_truth()
    ground_truth = replace_regex_with_map(ground_truth, token_const_maps)
    ground_truth = " ".join(tokenize(ground_truth))
    
    pos_exs = r["pos_examples"]
    neg_exs = r["neg_examples"]
    exs = []
    for pex in pos_exs.split("\n"):
        exs.append("+,"+pex)
    for nex in neg_exs.split("\n"):
        exs.append("-,"+nex)  
    exs = " ".join(exs)
    rec = r
    return symbol_descp, ground_truth, token_const_maps, exs, rec, regex

def write_dataset(records, split):
    print(len(records))
    print("Split", split)
    print(len(records))
    descps = [x[0] for x in records]
    gts = [x[1] for x in records]
    maps = [x[2] for x in records]
    exs = [x[3] for x in records]
    rec = [x[4] for x in records]
    regex = [x[5] for x in records]

    prefix = "./ARealBase"
    with open(join(prefix, "src-{}.txt".format(split)), "w") as f:
        descps_lines = "\n".join(descps)
        f.write(descps_lines)
    
    with open(join(prefix, "targ-{}.txt".format(split)), "w") as f:
        gts_lines = "\n".join(gts)
        f.write(gts_lines)
    
    with open(join(prefix, "map-{}.txt".format(split)), "w") as f:
        maps_lines = [ " ".join([str(len(m))] + ["{},{}".format(pair[0], pair[1]) for pair in m]) for m in maps]
        maps_lines = "\n".join(maps_lines)
        f.write(maps_lines)

    with open(join(prefix, "exs-{}.txt".format(split)), "w") as f:
        exs_lines = "\n".join(exs)
        f.write(exs_lines)
    
    with open(join(prefix, "rec-{}.pkl".format(split)), "wb") as f:
        rec_lines = [{"id":r["problem_id"], "worker_id": r["worker_id"]} for r in rec]
        pickle.dump(rec_lines, f)

def make_random_example():
    reg_libs = build_reg_libs()
    prefix = "./dataset"
    with open(join(prefix, "id-val.txt")) as f:
        lines = f.readlines()
        lines = [x.rstrip() for x in lines]
    id_lines = lines
    id_set = list(set(id_lines))

    bad_exs = []
    for id in id_set:
        print(id)
        r = reg_libs[id]
        spec = r.specification()
        exs = gen_bad_examples(spec)
        bad_exs.append(exs)
    id_exs_dict = dict(zip(id_set, bad_exs))
    
    with open(join(prefix, "bad_exs-val.txt"), "w") as f:
        exs_lines = [id_exs_dict[x] for x in id_lines]
        exs_lines = [" ".join(["{},{}".format(x[1],x[0]) for x in exs]) for exs in exs_lines]
        f.write("\n".join(exs_lines))

def make_dr_data():
    reg_libs = build_reg_libs()
    records_b1 = read_result("results/batch1_res.csv")
    records_b2 = read_result("results/batch2_res.csv")
    records_b3 = read_result("results/batch3_res.csv")
    records = records_b1 + records_b2 + records_b3
    records = group_by_filed(records, "problem_id")
    print(len(records))
    records = [records[k] for k in records]
    random.shuffle(records)
    split_point = 900
    train_records = records[:split_point]
    test_records = records[split_point:]
    
    train_records = list(reduce(lambda  x,y: x + y, map(lambda z: [record_to_metadata(r, reg_libs) for r in z], train_records)))
    train_records = filter_responces(train_records)
    train_records = mannually_filter_responces(train_records)
    test_records = list(reduce(lambda  x,y: x + y, map(lambda z: [record_to_metadata(r, reg_libs) for r in z], test_records)))
    test_records = filter_responces(test_records)
    test_records = mannually_filter_responces(test_records)
    write_dataset(train_records, "train")
    write_dataset(test_records, "val")

def post_process_data():
    reg_libs = build_reg_libs()
    records_b1 = read_result("results/batch1_res.csv")
    records_b2 = read_result("results/batch2_res.csv")
    records_b3 = read_result("results/batch3_res.csv")
    records = records_b1 + records_b2 + records_b3
    records = group_by_filed(records, "problem_id")
    print(len(records))
    records = [records[k] for k in records]
    train_records = records
    
    train_records = list(reduce(lambda  x,y: x + y, map(lambda z: [record_to_metadata(r, reg_libs) for r in z], train_records)))
    train_records = mannually_filter_responces(train_records)
    train_records = filter_responces(train_records)
    write_dataset(train_records, "col")

def read_file_lines(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [x.rstrip() for x in lines]
    return lines

def write_file_lines(filename, lines):
    with open(filename, "w") as f:
        f.write("\n".join(lines))
    return lines

def load_dataset(split):
    prefix = "./FinalSplit0"
    descps = read_file_lines(join(prefix, "src-{}.txt".format(split)))
    gts = read_file_lines(join(prefix, "targ-{}.txt".format(split)))
    exs = read_file_lines(join(prefix, "exs-{}.txt".format(split)))
    maps = read_file_lines(join(prefix, "map-{}.txt".format(split)))
    
    with open(join(prefix, "rec-{}.pkl".format(split)), "rb") as f:
        records = pickle.load(f)
    return list(zip(descps, gts, maps, exs, records))

def num_of_three_types(rec):
    return [len([y for y in rec if t in y[4]["id"]]) for t in ["uns", "cat", "sep"]]

def mannually_decide_testex(records_by_worker, full_records):
    random.shuffle(records_by_worker)
    min_allowed = 350
    found = []
    for r in records_by_worker:
        print(len(r), end=" ")
        found.extend(r)
        if len(found) > min_allowed:
            print()
            break
    num_type = [len([y for y in found if t in y["id"]]) for t in ["uns", "cat", "sep"]]
    if any([x >= 130 or x <= 110 for x in num_type]):
        return False
    print([len([y for y in found if t in y["id"]]) for t in ["uns", "cat", "sep"]])
    workers = list(set([x["worker_id"] for x in found]))
    ids = list(set([x["id"] for x in found]))
    print(workers)
    print(len(workers), len(found))
    print(len(ids), len(found), 3 * len(ids) - len(found))
    left_out = [x for x in full_records if x["id"] in ids and x["worker_id"] not in workers]
    print([len([y for y in left_out if t in y["id"]]) for t in ["uns", "cat", "sep"]], len(left_out))
    return True
    

TO_VOID_IN_TESTE =  ['A2RMJNF6IPI42F', 'A2ECRNQ3X5LEXD', 'A2PFLDMSADON5K']
# split1
# TEST_EX_WORKER = ['AE861G0AY5RGT', 'A6PRQVQM8YZ4W', 'A2AKGKD22DWZHI', 'A2F5AZQ55LXHKT', 'A20OJ1Q95TMP8B']

# split2
TEST_EX_WORKER = ['A1U5D0C8S15TIK', 'A2F5AZQ55LXHKT', 'A2AKGKD22DWZHI', 'A6PRQVQM8YZ4W', 'A38Z99XF4NDNH0', 'A3RMDIRX16L60E', 'A3DCO9GJ4XDVE2', 'A2DDPSXH2X96RF', 'A3UVLUYTHE86UA']

#split3
# TEST_EX_WORKER = ['A1E0WK5W1BFPWR', 'A2F5AZQ55LXHKT', 'A2NYUS12FHF2Y', 'AE861G0AY5RGT', 'A28QUR0QYD2WI7', 'A1MKCTVKE7J0ZP', 'A2AKGKD22DWZHI', 'A1U5D0C8S15TIK', 'A3UVLUYTHE86UA', 'A38Z99XF4NDNH0']

#split4
# TEST_EX_WORKER = ['A1IATW3PMVL6J3', 'A28QUR0QYD2WI7', 'A2UFGZT4QUY5ON', 'A3DCO9GJ4XDVE2', 'A2BDIIXOFUX18', 'A2RLSRUHS830A7', 'A38Z99XF4NDNH0', 'A2F5AZQ55LXHKT']

# split5
# TEST_EX_WORKER = ['AE861G0AY5RGT', 'A3UVLUYTHE86UA', 'A2UFGZT4QUY5ON', 'A3K0GYICW2CXM3', 'A1IATW3PMVL6J3', 'A3DCO9GJ4XDVE2']

# split6
# TEST_EX_WORKER = ['A3RMDIRX16L60E', 'A12LP4V8NTTEUE', 'A28QUR0QYD2WI7', 'A1MKCTVKE7J0ZP', 'A38Z99XF4NDNH0', 'A279JAYOXWD7PO', 'A002160837SWJFPIAI7L7', 'A1U5D0C8S15TIK', 'A2NYUS12FHF2Y', 'AE861G0AY5RGT']

def dump_dataset(records, split):
    descps = [x[0] for x in records]
    gts = [x[1] for x in records]
    maps = [x[2] for x in records]
    exs = [x[3] for x in records]
    rec = [x[4] for x in records]
    prefix = "./FinalSplit"
    write_file_lines(join(prefix, "src-{}.txt".format(split)), descps)
    write_file_lines(join(prefix, "targ-{}.txt".format(split)), gts)
    write_file_lines(join(prefix, "exs-{}.txt".format(split)), exs)
    write_file_lines(join(prefix, "map-{}.txt".format(split)), maps)
    
    with open(join(prefix, "rec-{}.pkl".format(split)), "wb") as f:
        pickle.dump(rec, f)


def decide_ex_workers():
    records = load_dataset("col")

    records = [r[4] for r in records]
    # print(records)
    records_by_worker = group_by_filed(records, "worker_id")
    records_by_worker = [records_by_worker[k] for k in records_by_worker if k not in TO_VOID_IN_TESTE]

    
    records_by_worker = [x for x in records_by_worker if (len(x) < 110 and len(x) > 20)]
    while(not  mannually_decide_testex(records_by_worker, records)):
        pass
    # mannually_decide_testex(records_by_worker)
    
def make_testex_split():
    records = load_dataset("col")

    records = [x for x in records if x[4]["worker_id"] in TEST_EX_WORKER]
    dump_dataset(records, "teste")
    
def make_train_dev_testi_split():
    records = load_dataset("col")

    ex_records = [x for x in records if x[4]["worker_id"] in TEST_EX_WORKER]
    ex_ids = list(set([x[4]["id"] for x in ex_records]))
    
    records = [x for x in records if x[4]["worker_id"] not in TEST_EX_WORKER]
    print(len(records))
    must_in_test_records = [x for x in records if x[4]["id"] in ex_ids]
    print(len(must_in_test_records))

    rest_records = [x for x in records if x[4]["id"] not in ex_ids]
    print(len(rest_records))
    rest_ids = list(set([x[4]["id"] for x in rest_records]))
    # random.seed(235)
    random.shuffle(rest_ids)
    num_each_type = 40
    rest_ids_by_types = [[x for x in rest_ids if t in x] for t in ["uns", "cat", "sep"]]
    rem_ids = sum([x[:num_each_type] for x in rest_ids_by_types], [])
    print(len(rem_ids))

    rem_records = [x for x in rest_records if x[4]["id"] in rem_ids]
    train_records = [x for x in rest_records if x[4]["id"] not in rem_ids]

    dev_records = rem_records
    testi_records = must_in_test_records
    print(num_of_three_types(train_records))
    print(num_of_three_types(dev_records))
    print(num_of_three_types(testi_records))

    dump_dataset(train_records, "train")
    dump_dataset(dev_records, "val")
    dump_dataset(testi_records, "testi")
    simple_stats(dev_records)

def simple_stats(records):


    def calc_ast_depth(x):
        return 1 + max([0] + [calc_ast_depth(c) for c in x.children])

    def calc_ast_size(x):
        return 1 + sum([calc_ast_size(c) for c in x.children])

    descps = [x[0] for x in records]
    gts = [x[1] for x in records]
    maps = [x[2] for x in records]
    
    descp_lens = np.array([len(x.split(" ")) for x in descps])
    print(descp_lens.mean())
    print(np.quantile(descp_lens, [0.0, 0.25, 0.5, 0.75, 1.0]))
    
    gt_toks = [x.replace(" ", "") for x in gts]
    targ_toks = [tokenize(x) for x in gt_toks]
    
    targ_asts = [build_dataset_ast_from_toks(x, 0)[0] for x in targ_toks]
    ast_sizes = np.array([calc_ast_size(x) for x in targ_asts])
    ast_depths = np.array([calc_ast_depth(x) for x in targ_asts])

    print("Avg Size", ast_sizes.mean())
    print("Qutiles Size", np.quantile(ast_sizes, [0,0.25,0.5,0.75,1.0])) 
    # view_special_ones(targ_lines, ast_sizes)
    print("Avg Depth", ast_depths.mean())
    print("Qutiles Depth", np.quantile(ast_depths, [0,0.25,0.5,0.75,1.0])) 

# def make_train_dev_testi_split():
#     records = load_dataset("col")

#     ex_records = [x for x in records if x[4]["worker_id"] in TEST_EX_WORKER]
#     ex_ids = list(set([x[4]["id"] for x in ex_records]))
    
#     records = [x for x in records if x[4]["worker_id"] not in TEST_EX_WORKER]
#     print(len(records))
#     must_in_test_records = [x for x in records if x[4]["id"] in ex_ids]
#     print(len(must_in_test_records))

#     rest_records = [x for x in records if x[4]["id"] not in ex_ids]
#     print(len(rest_records))
#     rest_ids = list(set([x[4]["id"] for x in rest_records]))
#     random.seed(666)
#     random.shuffle(rest_ids)
#     target_num = 710 - len(must_in_test_records)
    
#     rem_ids = []
#     cnt = 0
#     for id in rest_ids:
#         cnt += sum([x[4]["id"] == id for x in rest_records])
#         rem_ids.append(id)
#         if cnt > target_num:
#             break
    
#     print(len(rem_ids), cnt)

#     rem_records = [x for x in rest_records if x[4]["id"] in rem_ids]
#     train_records = [x for x in rest_records if x[4]["id"] not in rem_ids]
#     all_test_records = rem_records + must_in_test_records
#     half_num = len(all_test_records) // 2
#     print(len(train_records), len(all_test_records), half_num)
#     random.shuffle(all_test_records)

#     dev_records = all_test_records[:half_num]
#     testi_records = all_test_records[half_num:]
#     dump_dataset(train_records, "train")
#     dump_dataset(dev_records, "dev")
#     dump_dataset(testi_records, "testi")
    
def verify_partion():
    train = load_dataset("train")
    dev = load_dataset("dev")
    testi = load_dataset("testi")
    teste = load_dataset("teste")

    train_descps = [x[0] for x in train]
    print(sum([x[0] in train_descps for x in dev]))
    print(sum([x[0] in train_descps for x in testi]))
    print(sum([x[0] in train_descps for x in teste]))

    testi_descps = [x[0] for x in testi]
    dev_descps = [x[0] for x in dev]
    print(sum([x[0] in dev_descps for x in teste]))
    print(sum([x[0] in testi_descps for x in teste]))
    # print(sum([x in train_descps for x in dev]))

    train_regexes = [x[1] for x in train]
    print(sum([x[1] in train_regexes for x in dev]))
    print(sum([x[1] in train_regexes for x in testi]))
    print(sum([x[1] in train_regexes for x in teste]))
    
    testi_regexes = [x[1] for x in testi]
    dev_regexes = [x[1] for x in dev]
    print(sum([x[1] in dev_regexes for x in teste]))
    print(sum([x[1] in testi_regexes for x in teste]))

def stats_partion():
    train = load_dataset("train")
    dev = load_dataset("val")
    testi = load_dataset("testi")
    teste = load_dataset("teste")

    def targ_length(x):
        targs = [y[1] for y in x]
        return max([len(y.split()) for y in targs])
    print(targ_length(train))
    print(targ_length(dev))
    print(targ_length(testi))
    print(targ_length(teste))
    simple_stats(dev)

def mannually_assesment():
    train = load_dataset("train")
    dev = load_dataset("val")

    gts = [x[0] for x in train] + [x[0] for x in dev]
    targs = [x[1] for x in train] + [x[1] for x in dev]
    recs =  [x[4] for x in train] + [x[4] for x in dev]

    types = ["uns", "cat", "sep"]
    infos = list(zip(gts, targs, recs))
    random.shuffle(infos)
    def sample_n_things(t):
        t_samples = []
        id_sets = set()
        for gt, targ, r in infos:
            id = r["id"]   
            if len(id_sets) == 50:
                break
            if t not in id:
                continue
            if id in id_sets:
                continue
            id_sets.add(id)
            t_samples.append((gt, targ))

        return t_samples
    
    samples = [sample_n_things(t) for t in types]
    samples = sum(samples, [])
    sample_length = np.array([len(x[0].split()) for x in samples])
    print(sample_length.mean())
    with open("random_sample.txt", "w") as f:
        for gt, targ in samples:
            f.write('"{}","{}"\n'.format(gt, targ))


            


def get_spacy_tokenizer():
    disable = ["vectors", "textcat", "tagger", "parser", "ner"]
    spacy_model = spacy.load("en_core_web_sm", disable=disable)
    return spacy_model

# spacy.load("en_core_web_sm")
# spacy_tokenizer = get_spacy_tokenizer()
if __name__ == "__main__":  
    # random.seed(2333)
    # post_process_data()
    # decide_ex_workers()
    # make_testex_split()
    # make_train_dev_testi_split()
    # verify_partion()
    stats_partion()
    # mannually_assesment()