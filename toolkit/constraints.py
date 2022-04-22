from base import *

CC = [NumToken, LetterToken, CapitalToken, LowerToken, SpecialToken]
CC_NO_SPEC = [NumToken, LetterToken, CapitalToken, LowerToken]
SINGLE_TOKEN_CHOICE = [NumToken, LetterToken, CapitalToken, LowerToken, SpecialToken]

class ComposedByCons(Constraint):
    def logical_form(self):
        return "Star(Or({}))".format(",".join([x.logical_form() for x in self.children]))

    def description(self):
        return "Composed by {}".format(", " .join([x.logical_form() for x in self.children]))
    
    def specification(self):
        if len(self.children) == 1:
            return "repeatatleast({},1)".format(",".join([x.specification() for x in self.children]))
        else:
            r = [x.specification() for x in self.children]
            r.reverse()
            y = r[0]
            for c in r[1:]:
                y = 'or({},{})'.format(c,y)
            return "repeatatleast({},1)".format(y)

    @classmethod
    def generate(cls, max_element=4):
        cls.config_children_attr()
        things = []
        allowed = []
        num_elements = random.randint(1, max_element)

        element_choices = CC[:]
        single_choices = CC[:]
        if num_elements == 1:
            picked_type = random.choice(CC_NO_SPEC)
            picked = picked_type()
            things.append(picked)
            allowed.append(picked_type)
        else:
            for _ in range(num_elements):
                if random_decision(0.15 if (len(element_choices) > 0) else 1.0):
                    picked_type = SingleToken
                else:
                    picked_type = random.choice(element_choices)
                if picked_type == SingleToken:
                    picked = SingleToken.generate(single_choices, False)
                    allowed.append(picked)
                else:
                    picked = picked_type()
                    allowed.append(picked_type)
                things.append(picked)
                cls.resolve_avaliable_choices(
                    element_choices, single_choices, picked_type, picked)
                if len(element_choices) == 0 and len(single_choices) == 0:
                    break
        if things:
            ComposedByCons.sort_tokens(things)
            return cls(*things), allowed
        return NoneToken(), None

    @staticmethod
    def sort_tokens(tokens):
        tokens.sort(key=lambda x: 1 if isinstance(x, SingleToken) else 0)

    @classmethod
    def resolve_avaliable_choices(self, element_choices, single_choices, picked_type, picked):
        # deal with element choice
        # deal with single choice
        if picked_type == SingleToken:
            soft_remove(element_choices, picked.cc_type)
            # soft_remove(element_choices, SingleToken)
            # soft_remove(element_choices, SingleOrSingleToken)
            if picked.cc_type == LetterToken:
                soft_remove(element_choices, LowerToken)
                soft_remove(element_choices, CapitalToken)
                soft_remove(single_choices, LowerToken)
                soft_remove(single_choices, CapitalToken)
            if picked.cc_type == CapitalToken or picked.cc_type == LowerToken:
                soft_remove(element_choices, LetterToken)
                soft_remove(single_choices, LetterToken)
            return

        soft_remove(element_choices, picked_type)
        soft_remove(single_choices, picked_type)

        if picked_type == LetterToken:
            soft_remove(single_choices, LowerToken)
            soft_remove(single_choices, CapitalToken)
            soft_remove(element_choices, LowerToken)
            soft_remove(element_choices, CapitalToken)
        if picked_type == CapitalToken or picked_type == LowerToken:
            soft_remove(element_choices, LetterToken)
            soft_remove(single_choices, LetterToken)

    @classmethod
    def config_children_attr(cls):
        for c in CC:
            c.cnt = -1

    def sample_negative(self):
        # avaliable violation
        options = []
        CC_LEFT = CC[:]
        for c in self.children:
            if isinstance(c, SingleToken):
                options.append(c.cc_type)
            elif isinstance(c, LetterToken):
                soft_remove(CC_LEFT, LetterToken)
                soft_remove(CC_LEFT, CapitalToken)
                soft_remove(CC_LEFT, LowerToken)
            elif isinstance(c, CapitalToken):
                soft_remove(CC_LEFT, CapitalToken)
                soft_remove(CC_LEFT, LetterToken)
            elif isinstance(c, LowerToken):
                soft_remove(CC_LEFT, LowerToken)
                soft_remove(CC_LEFT, LetterToken)
            elif isinstance(c, NumToken):
                soft_remove(CC_LEFT, NumToken)
            elif isinstance(c, SpecialToken):
                soft_remove(CC_LEFT, SpecialToken)
        # SingleToken -> Full
        # Digit Letter
        if len(options):
            new_base = random.choice(options)
            new_children = self.children + [new_base()]
            return ComposedByCons(*new_children)
        options.extend(CC_LEFT)
        if len(options):
            new_base = random.choice(options)
            new_children = self.children + [new_base()]
            return ComposedByCons(*new_children)
        else:
            return NoneToken()

class NotOnlyComposedByCons(Constraint):
    pass

class ContainCons(Constraint):
    def logical_form(self):
        return "Contain({})".format(self.children[0].logical_form())
        
    def description(self):
        return "Contain {}".format(self.children[0].description())
    
    def specification(self):
        return "contain({})".format(self.children[0].specification())

    @classmethod
    def generate(cls, allowed=None, ban_orcons=False):
        things = []
        
        containable_helper = ContainableFiledsHelper(allowed)
        things = [containable_helper.generate(ban_orcons=ban_orcons)]
        if things:
            return cls(*things)
        return NoneToken()

class NotContainCons(Constraint):
    @classmethod
    def generate(cls, allowed=None):
        contain_cons = ContainCons.generate(allowed=allowed, ban_orcons=True)
        
        if contain_cons is not None:
            return NotCons(*[contain_cons])
        else:
            return NoneToken()


# (can) contain x, but must be followed by y
# (can) contain x, but must be preceded by y
# y must be a CC

class ConditionalContainCons(Constraint):
    @classmethod
    def generate(cls, allowed=None):
        if allowed:
            cc_s = [y for y in allowed if y in CC_NO_SPEC]
        else:
            cc_s = CC_NO_SPEC[:]
        if not cc_s:
            return NoneToken()
        y = random.choice(cc_s)
        x_s = allowed[:] if allowed else CC[:]
        soft_remove(x_s, y)
        if not x_s:
            return NoneToken()
        x = random.choice(x_s)
        if x in CC:
            x = x()
        if y in CC:
            y = y()
        condition = NotCCCons(y)
        if random_decision(0.5):
            order = [x, condition]
            cons = NotCons(ContainCons(ConcatComp(*order)))
            return AndComp(NotCons(EndwithCons(x)), cons)
        else:
            order = [condition, x]
            cons = NotCons(ContainCons(ConcatComp(*order)))
            return AndComp(NotCons(StartwithCons(x)), cons)

class StartwithCons(Constraint):
    def logical_form(self):
        return "StartWith({})".format(self.children[0].logical_form())
        
    def description(self):
        return "StartWith {}".format(self.children[0].description())
    
    def specification(self):
        return "startwith({})".format(self.children[0].specification())
    
    @classmethod
    def generate(cls, allowed=None, ban_orcons=False):
        things = []
        helper = SEWithFiledHelper(allowed)
        things = [helper.generate(ban_orcons=ban_orcons)]
        if things:
            return cls(*things)
        return NoneToken()

class EndwithCons(Constraint):
    def logical_form(self):
        return "EndWith({})".format(self.children[0].logical_form())
        
    def description(self):
        return "End with {}".format(self.children[0].description())
    
    def specification(self):
        return "endwith({})".format(self.children[0].specification())
    
    @classmethod
    def generate(cls, allowed=None, ban_orcons=False):
        things = []
        
        helper = SEWithFiledHelper(allowed)
        things = [helper.generate(ban_orcons=ban_orcons)]
        if things:
            return cls(*things)
        return NoneToken()

class NotStartwithCons(Constraint):
    @classmethod
    def generate(cls, allowed=None):
        contain_cons = StartwithCons.generate(allowed=allowed, ban_orcons=True)
        
        if contain_cons is not None:
            return NotCons(*[contain_cons])
        else:
            return NoneToken()

class NotEndwithCons(Constraint):
    @classmethod
    def generate(cls, allowed=None):
        contain_cons = EndwithCons.generate(allowed=allowed, ban_orcons=True)
        
        if is_valid(contain_cons):
            return NotCons(*[contain_cons])
        else:
            return NoneToken()

class ConditionalStartwithCons(Constraint):
    @classmethod
    def generate(cls, allowed=None):
        things = []
        helper = SEWithFiledHelper(allowed)
        sup_set, child_set = helper.generate_inclusive_pair()
        if is_valid(sup_set) and is_valid(child_set):
            return AndComp(StartwithCons(sup_set),NotCons(StartwithCons(child_set)))
        return NoneToken()

class ConditionalEndwithCons(Constraint):
    @classmethod
    def generate(cls, allowed=None):
        things = []
        helper = SEWithFiledHelper(allowed)
        sup_set, child_set = helper.generate_inclusive_pair()
        if is_valid(sup_set) and is_valid(child_set):
            return AndComp(EndwithCons(sup_set),NotCons(EndwithCons(child_set)))
        return NoneToken()

class LengthCons(Constraint):
    pass

class LengthOfCons(LengthCons):
    @classmethod
    def generate(cls, rng):
        return RepeatMod.generate(CharacterToken(), rng)

class LengthLessThanCons(LengthCons):
    @classmethod
    def generate(cls, rng):
        return RepeatRangeMod(CharacterToken(), 0, random.randint(rng[0], rng[1]))

class LengthMoreThanCons(LengthCons):
    @classmethod
    def generate(cls, rng):
        return RepeatAtLeastMod.generate(CharacterToken(),rng)

class LengthBetweenCons(LengthCons):
    @classmethod
    def generate(cls, min_range, max_range):
        return RepeatRangeMod.generate(CharacterToken(), min_range, max_range)

class RepeatMod(Modifier):
    def __init__(self, child, x):
        super().__init__(*[child])
        self.params.append(x)

    def logical_form(self):
        if self.params[0] == 1:
            return self.children[0].logical_form()
        else:
            return "Repeat({},{})".format(self.children[0].logical_form(), self.params[0])

    def description(self):
        return "Repeat {}, {} times".format(self.children[0].description(),self.params[0])

    def specification(self):
        if self.params[0] == 1:
            return self.children[0].specification()
        else:
            return "repeat({},{})".format(self.children[0].specification(), self.params[0])

    @classmethod
    def generate(cls, child, rng):
        x = random.randint(rng[0], rng[1])        
        return cls(child, x)
    
    def sample_negative(self):
        x = self.params[0]
        _x = random.choice([x -1, x + 1])
        return RepeatMod(self.children[0], _x)

class RepeatAtLeastMod(Modifier):
    def __init__(self, child, x):
        super().__init__(*[child])
        self.params.append(x)

    def logical_form(self):
        return "RepeatAtLeast({},{})".format(self.children[0].logical_form(), self.params[0])

    def description(self):
        return "Repeat {}, {} times".format(self.children[0].description(),self.params[0])

    def specification(self):
            return "repeatatleast({},{})".format(self.children[0].specification(), self.params[0])

    @classmethod
    def generate(cls, child, rng):
        x = random.randint(rng[0], rng[1])        
        return cls(child, x)
    
    def sample_negative(self):
        x = self.params[0]
        _x = x - 1
        return RepeatMod(self.children[0], _x)

class RepeatRangeMod(Modifier):
    def __init__(self, child, lb, hb):
        super().__init__(*[child])
        self.params.append(lb)
        self.params.append(hb)

    def logical_form(self):
        return "RepeatRange({},{},{})".format(self.children[0].logical_form(), self.params[0], self.params[1])
        
    def description(self):
        return "Repeat range {}, {} to {} times".format(self.children[0].description(), self.params[0], self.params[1])

    def specification(self):
        return "repeatrange({},{},{})".format(self.children[0].specification(), self.params[0], self.params[1])
    
    @classmethod
    def generate(cls, child, min_range, max_range):
        lb = random.randint(min_range[0], min_range[1])
        hb = random.randint(max(max_range[0], lb + 1), max_range[1])
        
        return cls(child, lb, hb)
        
    def sample_negative(self):
        lb = self.params[0]
        hb = self.params[1]
        if lb == 0:
            _x = hb + 1
        else:
            _x = random.choice([lb -1, hb + 1])
        return RepeatMod(self.children[0], _x)

class FieldHelper:
    def __init__(self, allowed=None):
        self.allowed = allowed

    def sample_single(self):
        if self.allowed:
            return SingleToken.generate(self.allowed, False)
        else:
            return SingleToken.generate(CC, False)

    def sample_string(self):
        if self.allowed:
            return StringToken.generate(self.allowed, False)
        else:
            return StringToken.generate(CC_NO_SPEC, False)

    def sample_single_or_cc(self):
        if self.allowed:
            choices = self.allowed
            picked_type = random.choice(choices)
            if picked_type in CC:
                return picked_type()
            else:
                return SingleToken.generate([picked_type])
        else:
            choices = CC + [SingleToken]
            picked_type = random.choice(choices)
            if picked_type in CC:
                return picked_type()
            else:
                return SingleToken.generate(CC, False)

    def sample_simple_or_fields(self):
        choices = [StringToken, RepeatMod]
        weights = [0.3, 1.0]
        picked_type = weighted_random_decision(choices, weights)
        if picked_type == StringToken:
            if self.allowed:
                cc_candidates = [x for x in self.allowed if x in CC]
                if not cc_candidates:
                    return NoneToken()
            else:
                cc_candidates = CC
            picked_cc = random.choice(cc_candidates)
            return OrComp(StringToken.generate([picked_cc]), StringToken.generate([picked_cc]))
        
        # in repeat type
        else:
            # choose between single token or cc
            tok_type = weighted_random_decision([SingleToken, CCToken], [1.0, 3.5])
            if tok_type == SingleToken:
                if self.allowed:
                    cc_candidates = [x for x in self.allowed if x in CC]
                    if not cc_candidates:
                        return NoneToken()
                else:
                    cc_candidates = CC
                picked_cc = random.choice(cc_candidates)
                tok1 = SingleToken.generate([picked_cc], False)
                tok2 = SingleToken.generate([picked_cc], False)
            else:
                tok1, tok2 = self.sample_cc_pair()
                if isinstance(tok1, NoneToken):
                    return NoneToken()
        
        repeating_time = weighted_random_decision([1, 2, 3], [0.55, 0.25, 0.25])
        # return 
        if repeating_time == 1:
            return OrComp(tok1, tok2)
        else:
            return OrComp(RepeatMod(tok1, repeating_time), RepeatMod(tok2, repeating_time))

    def sample_cc_pair(self):
        if self.allowed:
            cc_candidates = [x for x in self.allowed if x in CC]
        else:
            cc_candidates = CC[:]

        toks = []
        if len(cc_candidates) == 0:
            return NoneToken(), NoneToken()

        if len(cc_candidates) == 1:
            if cc_candidates[0] == LetterToken:
                toks = [LowerToken, CapitalToken]
                random.shuffle(toks)
                return toks[0](), toks[1]()
            else:
                return NoneToken(), NoneToken()
        else:
            if self.allowed and (LetterToken in cc_candidates):
                cc_candidates.append(CapitalToken)
                cc_candidates.append(LowerToken)
        print(cc_candidates)
        tok1 = random.choice(cc_candidates)
        print(tok1)
        soft_remove(cc_candidates, tok1)
        if tok1 == LetterToken:
            soft_remove(cc_candidates, CapitalToken)        
            soft_remove(cc_candidates, LowerToken)
        if tok1 == CapitalToken or tok1 == LowerToken:
            soft_remove(cc_candidates, LetterToken)
        print(cc_candidates)
        tok2 = random.choice(cc_candidates)

        return tok1(), tok2()

    def sample_cc(self):
        if self.allowed:
            cc_list = [x for x in self.allowed if x in CC]
            if not cc_list:
                return NoneToken()
            picked_type = random.choice(cc_list)
            return picked_type()
        else:
            picked_type = random.choice(CC)
            return picked_type()

    def instantiate_cat_candidate(self, picked_type):
        if picked_type == SingleToken:
            return self.sample_single()
        elif picked_type in CC:
            return picked_type()
        elif picked_type == RepeatMod:
            return RepeatMod.generate(self.sample_single_or_cc(), (2,3))

    def sample_concat(self):
        if self.allowed:
            candidates = [SingleToken] + [x for x in self.allowed if x in CC] + [RepeatMod]
        else:
            candidates = [SingleToken] + CC + [RepeatMod]
        
        type_c1 = random.choice(candidates)
        type_c2 = random.choice(candidates)
        
        c1 = self.instantiate_cat_candidate(type_c1)
        c2 = self.instantiate_cat_candidate(type_c2)
        return ConcatComp(c1,c2)
    

# only help constuct
# containabble helper , has to be true subset
class ContainableFiledsHelper(FieldHelper):
    def generate(self, ban_orcons=True):
        OR_WEIGHT = 1.5
        if not self.allowed:
            feasible_list = CC + [SingleToken, StringToken, RepeatMod, SimpleConcat2Token]
        if self.allowed:
            if len(self.allowed) == 1:
                feasible_list = [SingleToken, StringToken]
            else:
                feasible_list = [x for x in self.allowed if x in CC] + [SingleToken, StringToken, RepeatMod]
                # if two distinct Token
                if len(self.allowed) > 2:
                    feasible_list.append(SimpleConcat2Token)
        feasible_weights = [1.0 for _ in feasible_list]
        if not ban_orcons:
            feasible_list.append(SimpleOr2Token)
            feasible_weights.append(OR_WEIGHT)
        
        picked_type = weighted_random_decision(feasible_list, feasible_weights)
        if picked_type == SingleToken:
            return self.sample_single()
        elif picked_type in CC:
            return picked_type()
        elif picked_type == RepeatMod:
            return RepeatMod.generate(self.sample_single_or_cc(), (2,3))
        elif picked_type == SimpleConcat2Token:
            return self.sample_concat()
        elif picked_type == StringToken:
            return self.sample_string()
        elif picked_type == SimpleOr2Token:
            return self.sample_simple_or_fields()

class OptionalCons(Constraint):
    def logical_form(self):
        return "Optional({})".format(self.children[0].logical_form())
        
    def description(self):
        return "Optional {}".format(self.children[0].description())
    
    def specification(self):
        return "optional({})".format(self.children[0].specification())
        
    def sample_negative(self):
        return self.children[0].sample_negative()

# only help constuct
class SEWithFiledHelper(FieldHelper):
    def generate(self, ban_orcons=True):
        OR_WEIGHT = 3.0
        if not self.allowed:
            if random_decision(0.1):
                picked_type = random.choice([SingleToken, StringToken])
            else:
                feasible_list = CC + [RepeatMod, SimpleConcat2Token]
                feasible_weights = [1.0 for _ in feasible_list]
                if not ban_orcons:
                    feasible_list.append(SimpleOr2Token)
                    feasible_weights.append(OR_WEIGHT)
                picked_type = weighted_random_decision(feasible_list, feasible_weights)
        if self.allowed:
            if len(self.allowed) == 1:
                picked_type = random.choice([SingleToken, StringToken])
            else:
                if random_decision(0.1):
                    picked_type = random.choice([SingleToken, StringToken])
                feasible_list = [x for x in self.allowed if x in CC] + [RepeatMod]
                # if two distinct Token
                if len(self.allowed) > 2:
                    feasible_list.append(SimpleConcat2Token)
                feasible_weights = [1.0 for _ in feasible_list]
                if not ban_orcons:
                    feasible_list.append(SimpleOr2Token)
                    feasible_weights.append(OR_WEIGHT)
                picked_type = weighted_random_decision(feasible_list, feasible_weights)
        
        if picked_type == SingleToken:
            return self.sample_single()
        elif picked_type in CC:
            return picked_type()
        elif picked_type == RepeatMod:
            return RepeatMod.generate(self.sample_single_or_cc(), (2,3))
        elif picked_type == SimpleConcat2Token:
            return self.sample_concat()
        elif picked_type == StringToken:
            return self.sample_string()
        elif picked_type == SimpleOr2Token:
            return self.sample_simple_or_fields()
    
    def generate_inclusive_pair(self, exclude=None):
        if not self.allowed:
            # sup_set_type = random.choice(CC + [RepeatMod])
            if random_decision(0.55):
                sup_set_type = random.choice(CC_NO_SPEC)
            else:
                sup_set_type = RepeatMod
        if self.allowed:
            if len(self.allowed) == 1:
                return NoneToken(), NoneToken()
            feasible_list = [x for x in self.allowed if x in CC]
            if not feasible_list:
                return NoneToken(), NoneToken()
            if random_decision(0.55):
                sup_set_type = random.choice(CC_NO_SPEC)
            else:
                sup_set_type = RepeatMod
        
        if sup_set_type in CC:
            sup_set = sup_set_type()
            child_set_type = weighted_random_decision([SingleToken, StringToken], [0.6, 0.45])
            if child_set_type == SingleToken:
                child_set = SingleToken.generate([sup_set_type], False)
            elif child_set_type == StringToken:
                child_set = StringToken.generate([sup_set_type], False)

        elif sup_set_type == RepeatMod:
            if not self.allowed:
                sup_set_child_type = random.choice(CC)
            if self.allowed:
                feasible_list = [x for x in self.allowed if x in CC]
                if not feasible_list:
                    return NoneToken(), NoneToken()
                sup_set_child_type = random.choice(feasible_list)
            sup_set_child = sup_set_child_type()
            sup_set = RepeatMod.generate(sup_set_child, (2,3))
            if random_decision(0.4):
                sup_length = sup_set.params[0]
                child_set = StringToken.generate([sup_set_child_type], length=sup_length)
            else:
                child_set_child = SingleToken.generate([sup_set_child_type], False)
                child_set = RepeatMod(child_set_child, sup_set.params[0])
        return sup_set, child_set

class NotOnlyCons(Constraint):
    pass
