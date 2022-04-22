from base import *
from template import *
from constraints import *
import csv
import sys

def str_to_class(str):
    return getattr(sys.modules[__name__], str)

def tokenize(x):
    y = []
    while len(x) > 0:
        head = x[0]
        if head in ["?", "(", ")", "{", "}", ","]:
            y.append(head)
            x = x[1:]
        elif head == "<":
            end = x.index(">") + 1
            y.append(x[:end])
            x = x[end:]
        else:
            leftover = [(i in ["?", "(", ")", "{", "}", "<", ">", ","]) for i in x]
            end = leftover.index(True)
            y.append(x[:end])
            x = x[end:]
    return y

class ASTNode:
    def __init__(self, node_class, children=[], params=[]):
        self.node_class = node_class
        self.children = children
        self.params = params
    
    def logical_form(self):
        if len(self.children) + len(self.params) > 0:
            return self.node_class + "(" + ",".join([x.logical_form() for x in self.children] + [str(x) for x in self.params]) + ")"
        else:
            return self.node_class

    def tokenized_logical_form(self):
        if len(self.children) + len(self.params) > 0:
            toks = [self.node_class] + ["("]
            toks.extend(self.children[0].tokenized_logical_form())
            for c in self.children[1:]:
                toks.append(",")
                toks.extend(c.tokenized_logical_form())
            for p in [str(x) for x in self.params]:
                toks.append(",")
                toks.append(p)
            toks.append(")")
            return toks
        else:
            return [self.node_class]

def build_func_from_str(s):
    toks = tokenize(s)
    ast, _ = build_ast_from_toks(toks, 0)
    func = build_func_from_ast(ast)
    return func
    

def build_func_from_ast(ast):
    node_class = ast.node_class
    if node_class == "SeperatedField":
        fields = [build_func_from_ast(x) for x in ast.children[0].children]
        delimeter = build_func_from_ast(ast.children[1].children[0])
        return SeperatedField(delimeter, fields)
    elif node_class == "InfSeperatedField":
        field = build_func_from_ast(ast.children[0].children[0])
        delimeter = build_func_from_ast(ast.children[1].children[0])
        return InfSeperatedField(delimeter, field)
    elif node_class == "SingleToken":
        cc_type = str_to_class(ast.children[0].node_class[1:-1])
        tok = ast.children[1].node_class[1:-1]
        return SingleToken(cc_type, tok)
    elif node_class == "StringToken":
        cc_type = str_to_class(ast.children[0].node_class[1:-1])
        tok = ast.children[1].node_class[1:-1]
        return StringToken(cc_type, tok)
    elif node_class == "RepeatMod":
        child = build_func_from_ast(ast.children[0])
        return RepeatMod(child, ast.params[0])
    elif node_class == "RepeatRangeMod":
        child = build_func_from_ast(ast.children[0])
        return RepeatRangeMod(child, ast.params[0], ast.params[1])
    elif node_class == "RepeatAtLeastMod":
        child = build_func_from_ast(ast.children[0])
        return RepeatAtLeastMod(child, ast.params[0])


    children = [build_func_from_ast(x) for x in ast.children]

    cls_type = str_to_class(node_class)
    return cls_type(*children)

def build_ast_from_toks(toks, cur):
    node_class = None
    children = []
    params = []

    while cur < len(toks):
        head = toks[cur]
        if head.startswith("<") and head.endswith(">"):
            return ASTNode(head), cur + 1
        elif head == ")":
            return ASTNode(node_class, children, params), cur + 1
        elif head == "(" or head == ",":
            next_tok = toks[cur + 1]
            if next_tok.isdigit():
                params.append(int(next_tok))
                cur = cur + 2
            elif head == "(" and next_tok == ")":
                return ASTNode(node_class), cur + 2
            else:
                ret_vals = build_ast_from_toks(toks, cur + 1)
                children.append(ret_vals[0])
                cur = ret_vals[1]
        else:
            node_class = head
            cur = cur + 1
    print(cur, node_class, children, params)

def build_dataset_ast_from_toks(toks, cur):
    node_class = None
    children = []
    params = []

    while cur < len(toks):
        head = toks[cur]
        if head.startswith("<") and head.endswith(">"):
            return ASTNode(head), cur + 1
        elif head.startswith("const") and head[5:].isdigit():
            return ASTNode(head), cur + 1
        elif head == ")":
            return ASTNode(node_class, children, params), cur + 1
        elif head == "(" or head == ",":
            next_tok = toks[cur + 1]
            if next_tok.isdigit():
                params.append(int(next_tok))
                cur = cur + 2
            elif head == "(" and next_tok == ")":
                return ASTNode(node_class), cur + 2
            else:
                ret_vals = build_dataset_ast_from_toks(toks, cur + 1)
                children.append(ret_vals[0])
                cur = ret_vals[1]
        else:
            node_class = head
            cur = cur + 1
    print(cur, node_class, children, params)

def read_tsv_file(filename, delimiter="\t"):
    with open(filename) as f:
        lines = f.readlines()
        lines = [x.rstrip() for x in lines]
        lines = [x.split(delimiter) for x in lines]
    return lines

def row_to_record(row, header):
    record = {}
    record["hit_id"] = row[header.index("HITId")]
    record["worker_id"] = row[header.index("WorkerId")]
    record["work_time"] = row[header.index("WorkTimeInSeconds")]
    pos_exs = row[header.index("Input.pos_exs")]
    neg_exs = row[header.index("Input.neg_exs")]
    # <ul><li>x</li><li>x</li></ul>
    pos_exs = pos_exs[8:-10].split("</li><li>")
    neg_exs = neg_exs[8:-10].split("</li><li>")
    record["pos_examples"] = "\n".join(pos_exs)
    record["neg_examples"] = "\n".join(neg_exs)
    record["imgurl"] = row[header.index("Input.img_url")]
    record["problem_id"] = row[header.index("Input.id")]
    record["description"] = row[header.index("Answer.description")]
    record["pos_exs"] = row[header.index("Answer.pos_example")]
    if len(row) < len(header):  
        row.append("")
        row.append("")
    return record

def read_result(filename):
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        header = next(csv_reader)
        print(header)
        # exit()
        return [row_to_record(x, header) for x in csv_reader]

def group_by_filed(records, key):
    key_set = list(set([x[key] for x in records]))
    key_set.sort()

    grouped_records = dict(zip(key_set, [[x for x in records if x[key] == y] for y in key_set]))
    return grouped_records
