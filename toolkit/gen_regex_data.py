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

def get_posted_regexes():
    posted_batches = [1, 2]
    posted_regexes = []
    for b in posted_batches:
        fname = join("./HitArxiv", "batch" + str(b), "batch-{}-record.txt".format(b))
        regexes = [x[1] for x in read_tsv_file(fname, delimiter=" ")]
        regexes = [build_func_from_str(x) for x in regexes]
        posted_regexes.extend(regexes)
    return posted_regexes

def gen_pilot_template():
    # uns_regexes =[UnstructuredField.generate(5) for _ in range(300)]
    # cat_regexes = [ConcatenationField.generate(6) for _ in range(350)]
    sep_regexes = [SeperatedField.generate() for _ in range(350)]
    # print("uns", len(uns_regexes))
    # print("cat", len(cat_regexes))
    print("sep", len(sep_regexes))

    # do filtering
    # uns_regexes = filter_regexes(uns_regexes)
    # cat_regexes = filter_regexes(cat_regexes)
    sep_regexes = filter_regexes(sep_regexes)

    # print("uns", len(uns_regexes))
    # print("cat", len(cat_regexes))
    print("sep", len(sep_regexes))

    posted_regexes = get_posted_regexes()
    posted_forms = [x.logical_form() for x in posted_regexes]
    # print(len(uns_regexes), len(cat_regexes), len(sep_regexes))
    # uns_regexes = [x for x in uns_regexes if x.logical_form() not in posted_forms] 
    # cat_regexes = [x for x in cat_regexes if x.logical_form() not in posted_forms] 
    sep_regexes = [x for x in sep_regexes if x.logical_form() not in posted_forms] 
    # print(len(uns_regexes), len(cat_regexes), len(sep_regexes))

    prefix = "regexes-raw"
    # random.shuffle(uns_regexes)
    # with open(join(prefix, "batch3_uns.txt"), "w") as f:
    #     [f.write("{}\n".format(r.to_string())) for r in uns_regexes]
    # with open(join(prefix, "uns-plot.txt"), "w") as f:
        # [f.write("{}\n".format(tok(r.logical_form()))) for r in uns_regexes]
    
    # random.shuffle(cat_regexes)
    # with open(join(prefix, "batch3_cat.txt"), "w") as f:
    #     [f.write("{}\n".format(r.to_string())) for r in cat_regexes]
    # with open(join(prefix, "cat-plot.txt"), "w") as f:
    #     [f.write("id {}\n".format(tok(r.logical_form()))) for r in cat_regexes]
    
    random.shuffle(sep_regexes)
    with open(join(prefix, "batch3_sep.txt"), "w") as f:
        [f.write("{}\n".format(r.to_string())) for r in sep_regexes]
    # with open(join(prefix, "sep-plot.txt"), "w") as f:
    #     [f.write("{}\n".format(tok(r.logical_form()))) for r in sep_regexes]

def main():
    gen_pilot_template()

if __name__ == "__main__":
    random.seed(34341)
    main()
