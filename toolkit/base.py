import random

LOG_FLAG = False
def ctrl_logger(*args):
    if LOG_FLAG:
        print(args)

def random_decision(p):
    r = random.random()
    return r < p

def weighted_random_decision(choices, p):
    return random.choices(choices, weights=p)[0]

class Function():
    def __init__(self, *args):
        self.parent = None
        self.children = []
        self.params = []
        self.lineage = []
        for value in args:
            if issubclass(value.__class__, Function):
                self.children.append(value)
                value.parent = self
            else:
                self.params.append(value)

    def logical_form(self):
        raise Exception("Not implemented for {}".format(
            self.__class__.__name__))

    def description(self):
        raise Exception("Not implemented for {}".format(
            self.__class__.__name__))

    def specification(self):
        raise Exception("Not implemented for {}".format(
            self.__class__.__name__))
    # specification
    def ground_truth(self):
        return self.specification()

    # string of the DSL used for generating regexes. This is NOT specification. It's used for reconstruct the FUNCTION class
    def to_string(self):
        return "{}({})".format(self.__class__.__name__, ",".join([x.to_string() for x in self.children] + [str(x) for x in self.params]))

    def get_all_functions_flat_list(self):
        cur_list = []
        for child in self.children:
            if child:
                cur_list = cur_list + [child]
                cid = child.get_all_functions_flat_list()
                if cid:
                    cur_list = cur_list + cid
        return cur_list

    def set_root(self, root):
        self.root = root
    
    def sample_negative(self):
        raise Exception("Not implemented for {}".format(
            self.__class__.__name__))
    
    def negative_candidates(self):
        raise Exception("Not implemented for {}".format(
            self.__class__.__name__))

class NoneToken(Function):
    def logical_form(self):
        return "NONE"

    def description(self):
        return "NONE"

class Token(Function):
    cnt = -1
    
    def specification(self):
        return self.logical_form()
    
    def sample_negative(self):
        return NotCCCons(self)

class NumToken(Token):
    def logical_form(self):
        return "<num>"

    def description(self):
        return "a number"

    @classmethod
    def random_tok(cls):
        return str(random.randint(0, 9))

    @classmethod
    def nice_tok(cls):
        cls.cnt += 1
        # return ["0", "1", "2", "3"][cls.cnt % 4]
        return random.choice(["0", "1", "2", "3"])

    @classmethod
    def nice_string(cls):
        cls.cnt += 1
        # return ["00", "012", "999", "99"][cls.cnt % 4]
        return random.choice(["00", "012", "01", "999", "99"])  

# Tokens
class CapitalToken(Token):
    def logical_form(self):
        return "<cap>"

    def description(self):
        return "a capital letter"

    @classmethod
    def random_tok(cls):
        return chr(random.randint(1, 26) + 64)

    @classmethod
    def nice_tok(cls):
        cls.cnt += 1
        # return ["A", "B", "C", "D"][cls.cnt % 4]
        return random.choice(["A", "B", "C", "D"])

    @classmethod
    def nice_string(cls):
        cls.cnt += 1
        # return ["AA", "ABC", "XX", "AA"][cls.cnt % 4]
        return random.choice(["AA", "ABC", "XX", "AA", "XYZ"])

class LowerToken(Token):
    def logical_form(self):
        return "<low>"

    def description(self):
        return "a lower-case letter"

    @classmethod
    def random_tok(cls):
        return chr(random.randint(1, 26) + 96)

    @classmethod
    def nice_tok(cls):
        cls.cnt += 1
        # return ["a", "b", "c", "d"][cls.cnt % 4]
        return random.choice(["a", "b", "c", "d"])

    
    @classmethod
    def nice_string(cls):
        cls.cnt += 1
        # return ["aa", "abc", "xx", "aa"][cls.cnt % 4]
        return random.choice(["aa", "abc", "xx", "aa", "xyz"])


class LetterToken(Token):
    def logical_form(self):
        return "<let>"

    def description(self):
        return "a letter"

    @classmethod
    def random_tok(cls):
        x = random.random()
        if x > 0.5:
            return LowerToken.random_tok()
        else:
            return CapitalToken.random_tok()

    @classmethod
    def nice_tok(cls):
        cls.cnt += 1
        # return ["a", "A", "b", "B"][cls.cnt % 4]
        return random.choice(["a", "A", "b", "B"])

    @classmethod
    def nice_string(cls):
        cls.cnt += 1
        # return ["aA", "Aa", "aAa", "AaA"][cls.cnt % 4]
        return random.choice(["aA", "Aa", "aB", "aAa", "AaA"])

# allowed space, dash, comma, colon, dot, plus, underscore,
class SpecialToken(Token):
    spec_toks_ = [ "-", ",", ";", ".", "_", "+", ":", "!", "@", "#", "$", "%", "&", "^", "*", "="]
    nice_toks_ = ["-", ",", ";", "."]
    div_toks_ = ["-", ",", ";", "."]
    spec_toks = [ "-", ",", ";", ".", "_", "+", ":", "!", "@", "#", "$", "%", "&", "^", "*", "="]
    nice_toks = ["-", ",", ";", "."]
    def logical_form(self):
        return "<spec>"

    def description(self):
        return "a special character"

    @classmethod
    def random_tok(cls):
        return random.choice(cls.spec_toks)

    @classmethod
    def nice_tok(cls):
        cls.cnt += 1
        # return cls.nice_toks[cls.cnt % len(cls.nice_toks)]    
        return random.choice(cls.nice_toks)

    @classmethod
    def gen_div_tok(cls):
        tok = random.choice(cls.div_toks_)
        return SingleToken(SpecialToken, tok)

    @classmethod
    def nice_string(cls):
        cls.cnt += 1
        # return 2*cls.nice_toks[cls.cnt % len(cls.nice_toks)]    
        return 2 * random.choice(cls.nice_toks)

    @classmethod
    def screen_tok(cls, tok):
        cls.spec_toks = [x for x in cls.spec_toks_ if x != tok]
        cls.nice_toks = [x for x in cls.nice_toks_ if x != tok]

    @classmethod
    def restore(cls):
        cls.spec_toks = cls.spec_toks_[:]
        cls.nice_toks = cls.nice_toks_[:]

class CharacterToken(Token):
    def logical_form(self):
        return "<any>"

    def description(self):
        return "a character"


# allowed space, dash, comma, colon, dot, plus, underscore,
# input a list of CC or SingleToken

class SingleToken(Token):
    def __init__(self, cc_type, tok):
        super().__init__()
        self.parent = None
        self.cc_type = cc_type
        self.tok = tok

    @classmethod
    def generate(cls, choices, is_random=True):
        base = random.choice(choices)
        if isinstance(base, SingleToken):
            cc_type = base.cc_type
            tok = base.tok
            return cls(cc_type, tok)
        else:
            cc_type = base
            if is_random:
                tok = cc_type.random_tok()
            else:
                # tok = cc_type.nice_tok()
                tok = cc_type.random_tok()
            return cls(cc_type, tok)

    def logical_form(self):
        return '<{}>'.format(self.tok)

    def description(self):
        return '"{}"'.format(self.tok)
    
    def to_string(self):
        return "SingleToken(<{}>,<{}>)".format(self.cc_type.__name__, self.tok)
    
    def sample_negative(self):
        return SingleToken.generate([self.cc_type])

class StringToken(Token):
    def __init__(self, cc_type, tok):
        super().__init__()
        self.parent = None
        self.cc_type = cc_type
        self.tok = tok

    @classmethod
    def generate(cls, choices, is_random=True, length=-1):
        cc_s = [x for x in choices if not isinstance(x, SingleToken)]
        if cc_s:
            base = random.choice(cc_s)
            if length > 0:
                tok = "".join([base.random_tok() for _ in range(length)])
            else:
                if is_random:
                    tok = "".join([base.random_tok() for _ in range(random.randint(2,3))])
                else:
                    # tok = base.nice_string()
                    tok = "".join([base.random_tok() for _ in range(random.randint(2,3))])
            return cls(base, tok)
        else:
            return NoneToken()

    def logical_form(self):
        return '<{}>'.format(self.tok)

    def description(self):
        return '"the string {}"'.format(self.tok)
    
    def specification(self):
        return "const(<{}>)".format(self.tok)

    def ground_truth(self):
        return 'const(<{}>))'.format(self.tok)

    def to_string(self):
        return "StringToken(<{}>,<{}>)".format(self.cc_type.__name__, self.tok)
    
    def sample_negative(self):
        return StringToken.generate([self.cc_type])

class SingleOrSingleToken(Token):
    pass

class StringOrStringToken(Token):
    pass

class Composition(Function):
    pass

class OrComp(Composition):
    def logical_form(self):
        return "Or({},{})".format(self.children[0].logical_form(), self.children[1].logical_form())

    def description(self):
        return "{} or {}".format(self.children[0].description(),self.children[1].description())

    def specification(self):
        return "or({},{})".format(self.children[0].specification(), self.children[1].specification())

    def sample_negative(self):
        return random.choice(self.children).sample_negative()

class ConcatComp(Composition):
    def logical_form(self):
        if len(self.children) == 1:
            return self.children[0].logical_form()
        else:
            return "Concat({})".format(",".join([x.logical_form() for x in self.children]))

    def description(self):
        if len(self.children) == 1:
            return self.children[0].logical_form()
        else:
            return " followed by ".join([x.logical_form() for x in self.children])
    
    def specification(self):
        return self.concat_type_specification(self.children)

    def sample_negative(self):
        i = random.choice(range(len(self.children)))
        new_children = self.children[:i] + [self.children[i].sample_negative()] + self.children[i+1:]
        if len(new_children) == 0:
            return NoneToken()
        if len(new_children) == 1:
            return new_children[0]
        return ConcatComp(*new_children)
    
    @staticmethod
    def concat_type_specification(children):
        if len(children) == 1:
            return "{}".format(children[0].specification())
        else:
            r = [x.specification() for x in children]
            r.reverse()
            y = r[0]
            for c in r[1:]:
                y = 'concat({},{})'.format(c,y)
            return y

class AndComp(Composition):
    def logical_form(self):
        return "And({},{})".format(self.children[0].logical_form(), self.children[1].logical_form())

    def description(self):
        return "{} and {}".format(self.children[0].description(),self.children[1].description())

    def specification(self):
        return "and({},{})".format(self.children[0].specification(), self.children[1].specification())

    def sample_negative(self):
        i = random.choice(range(len(self.children)))
        new_children = self.children[:i] + [self.children[i].sample_negative()] + self.children[i+1:]
        return AndComp(*new_children)
    
    @staticmethod
    def and_type_specification(children):
        if len(children) == 1:
            return "{}".format(children[0].specification())
        else:
            r = [x.specification() for x in children]
            r.reverse()
            y = r[0]
            for c in r[1:]:
                y = 'and({},{})'.format(c,y)
            return y

class Constraint(Function):
    def sample_negative(self):
        return NotCons(self)

class NotCons(Constraint):
    def logical_form(self):
        return "Not({})".format(self.children[0].logical_form())

    def description(self):
        return "Not {} ".format(self.children[0].description())
    
    def specification(self):
        return "not({})".format(self.children[0].specification())
    
    def sample_negative(self):
        return self.children[0]

class NotCCCons(Constraint):
    def logical_form(self):
        return "NotCC({})".format(self.children[0].logical_form())

    def description(self):
        return "NotCC {} ".format(self.children[0].description())
    
    def specification(self):
        return "notcc({})".format(self.children[0].specification())

    def sample_negative(self):
        return self.children[0]

class CCToken(Token):
    pass

class ConstSet2Token(Token):
    pass

class SimpleConcat2Token(Token):
    pass

class SimpleOr2Token(Token):
    pass

class Modifier(Function):
    pass

def soft_remove(l, x):
    if x in l:
        l.remove(x)

def is_valid(x):
    if not isinstance(x, NoneToken):
        return all([is_valid(y) for y in x.children])
    else:
        return False

def tok(x):
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
    return "|".join(y)
