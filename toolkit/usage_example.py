from doctest import Example
from random import random


import random
import sys
import random
from collections import Counter
import subprocess
import re
from base import *
from template import *
from constraints import ComposedByCons
from filters import *
from os.path import join
from regex_io import read_tsv_file, build_func_from_str
from prepare_regex_data import gen_hit_pos_exs, gen_hit_neg_exs, gen_random_examples

# RegexClass: refer to Function class in base.py 
def sample_regexes_usage():
    random.seed(123)

    # sample a single regex wth the types described in the paper
    print(UnstructuredField.generate(5).specification())
    print(ConcatenationField.generate(6).specification())
    print(SeperatedField.generate().specification())

    # this will yield some crapy regex so we need to do rejection
    num_per_type = 10

    uns_regexes =[UnstructuredField.generate(5) for _ in range(num_per_type)]
    cat_regexes = [ConcatenationField.generate(6) for _ in range(num_per_type)]
    sep_regexes = [SeperatedField.generate() for _ in range(num_per_type)]
    regexes = uns_regexes + cat_regexes + sep_regexes

    # do filtering
    regexes = filter_regexes(regexes)
   
    print(len(regexes))
    print(regexes[0].specification())

    # save to file
    with open(join("usage_example_raw_regexes.txt"), "w") as f:
        [f.write("{}\n".format(r.to_string())) for r in uns_regexes]


# x: Function class
def sample_distinguish_examples(x):
    pos_examples = gen_hit_pos_exs(x)
    neg_examples = gen_hit_neg_exs(x)
    pos_examples = [(x,'+') for x in pos_examples]
    neg_examples = [(x,'-') for x in neg_examples]
    return pos_examples + neg_examples

# Distingushing of examples as described in the paper
def sample_distinguishing_exmples_usage():
    random.seed(123)

    # read the stored file, and get the first data point as example
    regexes = read_tsv_file("usage_example_raw_regexes.txt")
    regexes = [build_func_from_str(x[0]) for x in regexes]
    regex = regexes[0]
    print(regex.to_string())
    print(regex.specification())

    examples = sample_distinguish_examples(regex)
    print(examples)

def sample_random_examples(x):
    return gen_random_examples(x.specification())

# random examples
def sample_random_examples_usage():
    random.seed(123)

    # read the stored file, and get the first data point as example
    regexes = read_tsv_file("usage_example_raw_regexes.txt")
    regexes = [build_func_from_str(x[0]) for x in regexes]
    regex = regexes[0]
    print(regex.to_string())
    print(regex.specification())

    examples = sample_random_examples(regex)
    print(examples)

if __name__ == '__main__':
    sample_regexes_usage()
    sample_distinguishing_exmples_usage()
    sample_random_examples_usage()