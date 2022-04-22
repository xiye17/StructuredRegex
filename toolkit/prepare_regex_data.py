import os
from base import tok, is_valid
from os.path import join
from regex_io import build_func_from_str, read_tsv_file
from template import InfSeperatedField
import subprocess
import random
# REGEX_TYPES = ["uns", "cat", "sep"]
REGEX_TYPES = ["uns", "cat", "sep"]
RAW_DIR = "regexes-raw"
NUM_TRYIED = 100
NUM_KEEP = 6
NUM_EXS = 10

def prepare_pos_examples():
    # regexes_by_type = [read_regex_file(join(RAW_DIR, t + ".txt")) for t in REGEX_TYPES]
    for t in REGEX_TYPES:
        regexes = read_tsv_file(join(RAW_DIR, t + ".txt"))
        # with open(join(pr))
        examples = []
        for i, x in enumerate(regexes):
            print(t, i)
            examples.append(gen_examples(x[1]))
        # examples = [gen_examples(x) for x in regexes]
        examples_lines = ["\t".join(x) for x in examples]
        with open(join(RAW_DIR, t + "-pos.txt"), "w") as f:
            f.write("\n".join(examples_lines)) 

def gen_random_examples(regex, num_keep=NUM_KEEP, num_gen=NUM_TRYIED):
    out = subprocess.check_output(
        ['java', '-cp', './external/jars/datagen.jar:./external/lib/*', '-ea', 'datagen.Main', 'example',
            str(num_gen), regex])
    out = out.decode("utf-8")
    lines = out.split("\n")
    lines = lines[1:]
    lines = [x for x in lines if len(x)]
    fields = [(x[1:-3],x[-1]) for x in lines]
    pos_exs = [x for x in fields if x[1] == "+"]
    neg_exs = [x for x in fields if x[1] == "-"]
    random.shuffle(pos_exs)
    random.shuffle(neg_exs)
    pos_exs = pos_exs[:num_keep]
    neg_exs = neg_exs[:num_keep]
    exs = pos_exs + neg_exs
    return exs

def gen_pos_examples(regex, num_gen=NUM_TRYIED, is_spec=False):
    # try:
    if not is_spec:
        regex = regex.specification()
    print("Gen", regex)
    out = subprocess.check_output(
        ['java', '-cp', './external/jars/datagen.jar:./external/lib/*', '-ea', 'datagen.Main', 'example',
            str(num_gen), regex])
    out = out.decode("utf-8")
    return parse_examples(out)

def match_spec_example(regex, example):
    # try:
    print("Match", regex, example)
    out = subprocess.check_output(
        ['java', '-cp', './external/jars/datagen.jar:./external/lib/*', '-ea', 'datagen.Main', 'evaluate',
            regex, example])
    out = out.decode("utf-8")
    out = out.rstrip()
    return out

def gen_neg_examples(regex, num_keep=NUM_KEEP):
    # for a large number, random sample a path, gen a positive example
    exs = []
    reg_candidates = regex.negative_candidates()
    for neg_regex in reg_candidates:
        if not is_valid(neg_regex):
            continue
        over_exs = gen_pos_examples(neg_regex, num_gen=50)
        random.shuffle(over_exs)
        exs.extend(over_exs[:num_keep])
    return exs

def parse_examples(exs_out):
    lines = exs_out.split("\n")
    lines = lines[1:]
    lines = [x for x in lines if len(x)]
    fields = [(x[1:-3],x[-1]) for x in lines]
    fields = [x[0] for x in fields if x[1] == "+"]
    return fields

def gen_examples_file(filename, regex):
    out = subprocess.check_output(
        ['java', '-cp', './external/jars/datagen.jar:./external/lib/*', '-ea', 'datagen.Main', 'example',
            str(NUM_TRYIED), regex])
    exs_out = out.decode("utf-8")
    
    lines = exs_out.split("\n")
    lines = lines[1:]
    lines = [x for x in lines if len(x)]
    random.shuffle(lines)
    lines = lines[:20]
    lines = ["// examples"] + lines + ["", "// gt", regex]

    with open(filename, "w") as f:
        f.write("\n".join(lines))

def make_examples_file(filename, regex, pos_exs, neg_exs):
    pos_lines = pos_exs
    random.shuffle(pos_lines)
    pos_lines = pos_lines[:NUM_EXS]
    pos_lines = ['"{}",+'.format(x) for x in pos_lines]

    neg_lines = neg_exs
    random.shuffle(neg_lines)
    neg_lines = neg_lines[:NUM_EXS]

    spec = regex.specification()
    gt = regex.ground_truth()
    match_results = [match_spec_example(spec, x) for x in neg_lines]
    neg_lines = [x[0] for x in zip(neg_lines, match_results) if x[1] == "false"]
    print("before:", len(neg_exs), "after:", len(neg_lines))
    neg_lines = ['"{}",-'.format(x) for x in neg_lines]

    lines = ["// examples"] + pos_lines + neg_lines + ["", "// gt", gt]
    with open(filename, "w") as f:
        f.write("\n".join(lines))

# def prepare_data():
#     regexes_by_type = [read_tsv_file(join(RAW_DIR, t + ".txt")) for t in REGEX_TYPES]
#     regexes_by_type = [[build_func_from_str(x[0]) for x in regexes] for regexes in regexes_by_type]

#     pos_examples_by_type = [read_tsv_file(join(RAW_DIR, t + "-pos.txt")) for t in REGEX_TYPES]
#     neg_examples_by_type = [read_tsv_file(join(RAW_DIR, t + "-neg.txt")) for t in REGEX_TYPES]
#     # tgt_num_by_type = [15, 15, 30]
#     tgt_num_by_type = [50, 50, 50]
#     data_regexes = []
#     for regexes, pos_exs, neg_exs, num_tgt in zip(regexes_by_type, pos_examples_by_type, neg_examples_by_type, tgt_num_by_type):
#         reg_ex_pairs = list(zip(regexes, pos_exs, neg_exs))
#         print("Len before:",len(reg_ex_pairs))
#         reg_ex_pairs = [x for x in reg_ex_pairs if len(x[1]) >= NUM_KEEP]
#         print("Len after:",len(reg_ex_pairs))
#         data_regexes.extend(reg_ex_pairs[:num_tgt])

#     with open("pilot.txt", "w") as f:
#         for p in data_regexes:
#             f.write(tok(p[0].logical_form()) + "\n")
    
#     with open("pilot-spec.txt", "w") as f:
#         for p in data_regexes:
#             f.write(p[0].specification() + "\n")

#     with open("pilot.csv", "w") as f:
#         lines = []
#         lines.append("image_url,str_examples")
#         for i, p in enumerate(data_regexes):
#             img_url = '"http://taur.cs.utexas.edu/hidden/p/{}.png"'.format(i)
#             random.shuffle(p[1])
#             exs_str = '"<ul>{}</ul>"'.format("".join(["<li>{}</li>".format(x) for x in p[1][:NUM_KEEP]]))
#             lines.append("{},{}".format(img_url, exs_str))
#         f.write("\n".join(lines))

def prepare_data():
    regexes_by_type = [read_tsv_file(join(RAW_DIR, t + ".txt")) for t in REGEX_TYPES]
    regexes_by_type = [[build_func_from_str(x[0]) for x in regexes] for regexes in regexes_by_type]

    pos_examples_by_type = [read_tsv_file(join(RAW_DIR, t + "-pos.txt")) for t in REGEX_TYPES]
    # neg_examples_by_type = [read_tsv_file(join(RAW_DIR, t + "-neg.txt")) for t in REGEX_TYPES]
    # tgt_num_by_type = [15, 15, 30]
    tgt_num_by_type = [50, 50, 50]
    data_regexes = []
    for regexes, pos_exs, num_tgt in zip(regexes_by_type, pos_examples_by_type, tgt_num_by_type):
        reg_ex_pairs = list(zip(regexes, pos_exs))
        print("Len before:",len(reg_ex_pairs))
        reg_ex_pairs = [x for x in reg_ex_pairs if len(x[1]) >= NUM_KEEP]
        print("Len after:",len(reg_ex_pairs))
        data_regexes.extend(reg_ex_pairs[:num_tgt])

    with open("pilot.txt", "w") as f:
        for p in data_regexes:
            f.write(tok(p[0].logical_form()) + "\n")
    
    with open("pilot.csv", "w") as f:
        lines = []
        lines.append("image_url,str_examples")
        for i, p in enumerate(data_regexes):
            img_url = '"http://taur.cs.utexas.edu/hidden/p/{}.png"'.format(i)
            random.shuffle(p[1])
            exs_str = '"<ul>{}</ul>"'.format("".join(["<li>{}</li>".format(x) for x in p[1][:NUM_KEEP]]))
            lines.append("{},{}".format(img_url, exs_str))
        f.write("\n".join(lines))

def io_test():
    for t in REGEX_TYPES:
        regexes = read_tsv_file(join(RAW_DIR, t + ".txt"))
        
        examples = [build_func_from_str(x[0]).to_string() for x in regexes]
        with open(join(RAW_DIR, t + "-re.txt"), "w") as f:
            [f.write(r + "\n") for r in examples]

def prepare_examples():
    for t in REGEX_TYPES:
        regexes = read_tsv_file(join(RAW_DIR, t + ".txt"))
        regexes = [build_func_from_str(x[0]) for x in regexes]

        # neg_examples = [gen_neg_examples(x) for x in regexes]
        # neg_examples_lines = ["\t".join(x) for x in neg_examples]
        # with open(join(RAW_DIR, t + "-neg.txt"), "w") as f:
        #     f.write("\n".join(neg_examples_lines)) 

        pos_examples = [gen_pos_examples(x) for x in regexes]
        pos_examples_lines = ["\t".join(x) for x in pos_examples]
        with open(join(RAW_DIR, t + "-pos.txt"), "w") as f:
            f.write("\n".join(pos_examples_lines)) 

def read_example_file(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [x.rstrip() for x in lines]
    lines = lines[1:-2]
    lines = [x for x in lines if len(x)]
    fields = [(x[1:-3],x[-1]) for x in lines]
    pos_exs = [x[0] for x in fields if x[1] == "+"]
    neg_exs = [x[0] for x in fields if x[1] == "-"]
    return pos_exs, neg_exs

def compare_neg_examples():

    root = HtmlElement("html")
    for id in range(60):
        id = str(id)
        new_fname = join("./benchmark", id)
        old_fname = join("./benchmark-old", id)
        _, new_neg_exs = read_example_file(new_fname)
        _, old_neg_exs = read_example_file(old_fname)
        
        div = HtmlElement("div")
        img_url = '"http://taur.cs.utexas.edu/hidden/p/{}.png"'.format(id)
        root.add_children(HtmlElement("p", [ImgElement(img_url)]))
        root.add_children(HtmlElement("p", [PlanText("ID:" + id)]))

    with open("compare_neg.html", "w") as f:
        f.write(root.html())
    
def gen_hit_pos_exs(regex):
    hit_pos_exs = []
    if isinstance(regex, InfSeperatedField):
        spec_regexes = regex.required_special_examples()
        for spec_r in spec_regexes:
            spec_pos = gen_pos_examples(spec_r, is_spec=True)
            if spec_pos:
                hit_pos_exs.append(random.choice(spec_pos))
            else:
                print(spec_r, "Not Good")
    pos_exs = gen_pos_examples(regex)
    random.shuffle(pos_exs)
    hit_pos_exs.extend(pos_exs[:(NUM_KEEP - len(hit_pos_exs))])
    return hit_pos_exs

def gen_hit_neg_exs(regex):
    hit_neg_exs = gen_neg_examples(regex)
    random.shuffle(hit_neg_exs)
    hit_neg_exs = hit_neg_exs[:(2*NUM_KEEP)]
    spec = regex.specification()
    match_results = [match_spec_example(spec, x) for x in hit_neg_exs]
    hit_neg_exs = [x[0] for x in zip(hit_neg_exs, match_results) if x[1] == "false"]
    return hit_neg_exs[:NUM_KEEP]

def prepare_hit_fields(name, regex):
    # id, img_url, pos_exs, neg_exs
    id = name
    img_url = "http://taur.cs.utexas.edu/hidden/p/{}.png".format(id)
    pos_exs = gen_hit_pos_exs(regex)
    pos_exs = '<ul>{}</ul>'.format("".join(["<li>{}</li>".format(x) for x in pos_exs]))
    neg_exs = gen_hit_neg_exs(regex)
    neg_exs = '<ul>{}</ul>'.format("".join(["<li>{}</li>".format(x) for x in neg_exs]))
    return (id, img_url, pos_exs, neg_exs)

def prepare_hits():
    batch_id = 3
    # prepare batch.txt
    # list of id regex
    named_regexes = []
    tgt_num_by_type = [150, 150, 150]
    for num_tgt, t in zip(tgt_num_by_type, REGEX_TYPES):
        regexes = read_tsv_file(join(RAW_DIR, "batch{}_{}.txt".format(batch_id, t)))
        regexes = [build_func_from_str(x[0]) for x in regexes]
        regexes = regexes[:num_tgt]
        regexes = [("b-{}_t-{}_id-{}".format(batch_id, t, i), x) for i, x in enumerate(regexes)]
        named_regexes.extend(regexes)

    with open("batch-{}-record.txt".format(batch_id), "w") as f:
        for name, r in named_regexes:
            f.write("{} {} {}\n".format(name, r.to_string(), tok(r.logical_form())))

    with open("batch-{}.txt".format(batch_id), "w") as f:
        for name, r in named_regexes:
            f.write("{} {}\n".format(name, tok(r.logical_form())))
    
    # prepare batch.csv
    # id, img_url, pos_exs, neg_exs
    csv_fields = []
    for name, r in named_regexes:
        print(name)
        fields = prepare_hit_fields(name, r)
        csv_fields.append(fields)
    csv_lines = []
    csv_lines.append("id,img_url,pos_exs,neg_exs\n")
    for id, img_url, pos_exs, neg_exs in csv_fields:
        csv_lines.append('"{}","{}","{}","{}"\n'.format(id,img_url,pos_exs,neg_exs))
    with open("batch-{}.csv".format(batch_id), "w") as f:
        f.writelines(csv_lines)
    
    # preview
    root = HtmlElement("html")
    for id, img_url, pos_exs, neg_exs in csv_fields:
        div = HtmlElement("div")
        img_url = '"http://0.0.0.0:8000/preview/{}.png"'.format(id)
        root.add_children(HtmlElement("p", [ImgElement(img_url)]))
        root.add_children(HtmlElement("p", [PlanText("ID:" + id)]))
        root.add_children(HtmlElement("p", [PlanText("POS:")]))
        root.add_children(HtmlElement("p", [PlanText(pos_exs)]))
        root.add_children(HtmlElement("p", [PlanText("NEG:")]))
        root.add_children(HtmlElement("p", [PlanText(neg_exs)]))

    with open("preview_dataset.html", "w") as f:
        f.write(root.html())
    

def main():
    # prepare_examples()
    # prepare_data()
    # compare_neg_examples()
    prepare_hits()

if __name__ == "__main__":
    random.seed(123)
    main()