from base import *
from template import *
from functools import reduce
import subprocess
from prepare_regex_data import gen_pos_examples


def filter_regexes(regexes):
    regexes = [(x[1],print("Is Valiid", x[0],len(regexes)))[0] for x in enumerate(regexes) if is_valid(x[1])]
    regexes = [(x[1],print("Is Good", x[0],len(regexes)))[0] for x in enumerate(regexes) if is_good(x[1])]
    regexes = [(x[1],print("Is Diverse", x[0],len(regexes)))[0] for x in enumerate(regexes) if is_diverse(x[1])]
    return regexes

def is_diverse(node):
    pos_exs = gen_pos_examples(node, 100)
    num_pos = len(set(pos_exs))
    return num_pos > 10

def is_good(node):
    if not all([is_good(x) for x in node.children]):
        return False
    
    # if isinstance(node, RepeatMod):
    #     child = node.children[0]
    #     times = node.params[0]
    #     if isinstance(child, SingleToken) and times == 3:
    #         return False

    if isinstance(node, OrComp):
        child_0 = node.children[0]
        child_1 = node.children[1]
        if child_0.logical_form() == child_1.logical_form():
            return False

    if isinstance(node, ConcatenationField) or isinstance(node, ConcatComp):
        if not check_cat_type(node):
            return False
    
    if isinstance(node, AndComp):
        if not check_and_type(node):
            return False

    if isinstance(node, UnstructuredField):
        if not check_uns_type(node):
            return False

    if isinstance(node, ComposedByCons):
        if not check_or_type(node):
            return False

    return True

def extract_terminal(node):
    if isinstance(node, Token):
        return (node.logical_form(),)
    elif isinstance(node, NotCCCons):
        return ("not" + extract_terminal(node.children[0])[0],)
    else:
        return reduce(lambda  x,y: x + y, [extract_terminal(x) for x in node.children])

def check_cat_type(node):
    if all([isinstance(c, OptionalCons) for c in node.children]):
        return False

    flat_children = []
    for c in node.children:
        if isinstance(c, OptionalCons):
            if isinstance(c.children[0], ConcatComp):
                flat_children.extend(c.children[0].children)
            else:
                flat_children.append(c.children[0])
        else:
            flat_children.append(c)
    terminals = [extract_terminal(x) for x in flat_children]
    for i in range(len(terminals) - 1):
        if terminals[i] == terminals[i + 1]:
            return False
    return True

def check_and_type(node):
    children_logical_forms = [x.logical_form() for x in node.children]
    if len(set(children_logical_forms)) < len(children_logical_forms):
        return False
    return True

def check_or_type(node):
    children_logical_forms = [x.logical_form() for x in node.children]
    if len(set(children_logical_forms)) < len(children_logical_forms):
        return False
    return True

def check_uns_type(node):
    if not check_and_type(node):
        return False

    complexity = 0
    # complexity check
    for child in node.children:
        if isinstance(child, AndComp):
            complexity += 2
        elif isinstance(child, StartwithCons) or isinstance(child, EndwithCons) or isinstance(child, ContainCons):
            if "Or(" in child.logical_form():
                complexity += 1
        else:
            complexity += 1
    
    max_complexity = 3 if isinstance(node, SimpleUnstructuredField) else 6
    if complexity > max_complexity:
        print("Complexity ({}|{}) Filter".format(complexity, max_complexity))
        print(node.description())
        return False

    # check compabality
    
    
    cons = []  
    not_contain_cons = []  
    composed_by_cons = None
    for child in node.children:
        if isinstance(child, NotCons):
            construct = child.children[0]
        else:
            construct = child
        if isinstance(construct, ComposedByCons):
            cons.append(child)
            composed_by_cons = child
        if isinstance(construct, ContainCons):
            cons.append(child)
            if isinstance(child, NotCons):
                not_contain_cons.append(construct)
        if isinstance(construct, StartwithCons):
            cons.append(child)
        if isinstance(construct, EndwithCons):
            cons.append(child)

    if (composed_by_cons is not None) and not_contain_cons:
        for nc_cons in not_contain_cons:
            banned_tok = nc_cons.children[0].logical_form()
            for tok in composed_by_cons.children:
                if banned_tok == tok.logical_form():
                    print("Semantic Filter")
                    print(node.description())
                    return False

    if len(cons) >= 2:
        origin_spec = AndComp.and_type_specification(cons)
        for i in range(len(cons)):
            reduced_spec = AndComp.and_type_specification(cons[:i] + cons[i + 1:])
            if check_equiv(origin_spec, reduced_spec) == "true":
                print("Redundancy Filter")
                print(node.description())
                return False
    
    return True


def check_equiv(spec0, spec1):
    # try:
    out = subprocess.check_output(
        ['java', '-cp', './external/jars/datagen.jar:./external/lib/*', '-ea', 'datagen.Main', 'equiv',
            spec0, spec1], stderr=subprocess.DEVNULL)
    out = out.decode("utf-8")
    out = out.rstrip()
    return out

    
