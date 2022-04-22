from base import *
from constraints import *
import random


class UnstructuredField(Function):
    def logical_form(self):
        return "And({})".format(",".join([x.logical_form() for x in self.children]))

    def description(self):
        if len(self.children) == 1:
            return self.children[0].description()
        else:
            return "Constrained by:\n{}".format(",\n" .join([x.description() for x in self.children]))

    def specification(self):
        if len(self.children) == 1:
            return "{}".format(",".join([x.specification() for x in self.children]))
        else:
            r = [x.specification() for x in self.children]
            r.reverse()
            y = r[0]
            for c in r[1:]:
                y = 'and({},{})'.format(c,y)
            return y

    @classmethod
    def generate(cls, MAX_CONS):
        # i = 0
        # num_constrains = random.randint(1, MAX_CONS)
        num_constrains = weighted_random_decision(list(range(1, MAX_CONS + 1)), [0.8,1.15,1.3,1.35,2.2])
        ctrl_logger("num cons", num_constrains)
        constraint_types = cls.generate_constraint_types(num_constrains)
        ctrl_logger(constraint_types)
        arguments = []
        if ComposedByCons in constraint_types:
            cons, allowed = ComposedByCons.generate()
            constraint_types.remove(ComposedByCons)
            arguments.append(cons)
        else:
            allowed = None

        if LengthCons in constraint_types:
            constraint_types.remove(LengthCons)
            cons_type = random.choice([LengthOfCons, LengthBetweenCons, LengthLessThanCons, LengthMoreThanCons])
            if cons_type == LengthOfCons:
                cons = LengthOfCons.generate((6,10))
            elif cons_type == LengthBetweenCons:
                cons = LengthBetweenCons.generate((4,6),(8,10))
            elif cons_type == LengthLessThanCons:
                cons = LengthLessThanCons.generate((8,10))
            elif cons_type == LengthMoreThanCons:
                cons = LengthMoreThanCons.generate((4,6))
            arguments.append(cons)

        if not cls.validate_cons_compatibility(constraint_types, allowed):
            return NoneToken()

        for cons_type in constraint_types:
            cons = cons_type.generate(allowed)
            arguments.append(cons)
        if arguments:
            return cls(*arguments)
        return NoneToken()

    @classmethod
    def generate_constraint_types(cls, num_cons):
        if num_cons == 1:
            choices = [ComposedByCons, ContainCons, NotContainCons, ConditionalContainCons, StartwithCons,
                       NotStartwithCons, ConditionalContainCons, EndwithCons, NotEndwithCons, ConditionalEndwithCons]
            # weights = [1,1,1,0.25,0.5,0.5, 0.25, 0.5,0.5, 0.25]
            # return [weighted_random_decision(choices, weights)]
            return [weighted_random_decision(choices, UnstructuredField.gen_sample_weights(choices))]

        cons_types = []
        # prob of having a composed by cons
        if random_decision(0.5):
            cons_types.append(ComposedByCons)
            num_cons -= 1

        # prob of having a length cons
        if random_decision(0.5):
            cons_types.append(LengthCons)
            num_cons -= 1

        choices = [ContainCons, NotContainCons, ConditionalContainCons, StartwithCons, NotStartwithCons, ConditionalStartwithCons,
                   EndwithCons, NotEndwithCons, ConditionalEndwithCons]
        agg_cons = 0
        while (agg_cons < num_cons):
            picked_type = weighted_random_decision(choices, UnstructuredField.gen_sample_weights(choices))
            cons_types.append(picked_type)
            cls.resolve_avaliable_choices(choices, picked_type)
            if picked_type in [ConditionalContainCons, ConditionalStartwithCons, ConditionalEndwithCons]:
                agg_cons += 2
            else:
                agg_cons += 1
        return cons_types

    @staticmethod
    def gen_sample_weights(cons_types):
        half_list = [StartwithCons, NotStartwithCons, EndwithCons, NotEndwithCons]
        quat_list = [ConditionalContainCons, ConditionalEndwithCons, ConditionalStartwithCons]
        return [0.125 if x in quat_list else (0.5 if x in half_list else 1.0) for x in cons_types]

    @classmethod
    def validate_cons_compatibility(cls, constraint_types, allowed):
        for x in [ContainCons, NotContainCons, NotStartwithCons, NotEndwithCons]:
            if sum([y == x for y in constraint_types]) > 2:
                return False
        return True

    @classmethod
    def resolve_avaliable_choices(cls, choices, picked):
        if picked in [StartwithCons, ConditionalStartwithCons]:
            soft_remove(choices, StartwithCons)
            soft_remove(choices, NotStartwithCons)
            soft_remove(choices, ConditionalStartwithCons)
        elif picked in [EndwithCons, ConditionalEndwithCons]:
            soft_remove(choices, EndwithCons)
            soft_remove(choices, NotEndwithCons)
            soft_remove(choices, ConditionalEndwithCons)
        elif picked == NotStartwithCons:
            soft_remove(choices, StartwithCons)
            soft_remove(choices, ConditionalStartwithCons)
        elif picked == NotEndwithCons:
            soft_remove(choices, EndwithCons)
            soft_remove(choices, ConditionalEndwithCons)
        elif picked == ConditionalContainCons:
            soft_remove(choices, ConditionalContainCons)    

    def sample_negative(self):
        i = random.choice(range(len(self.children)))
        new_children = self.children[:i] + [self.children[i].sample_negative()] + self.children[i+1:]
        return UnstructuredField(*new_children)
    
    def negative_candidates(self):
        neg_regexes = []
        for i in range(len(self.children)):          
            new_children = self.children[:i] + [self.children[i].sample_negative()] + self.children[i+1:]
            neg_regexes.append(UnstructuredField(*new_children))
        return neg_regexes

    def posible_negate_way(self):
        return len(self.children)

class InfSeperatedField(Function):
    def __init__(self, delimeter, field):
        super().__init__(delimeter, field)
        self.delimiter = delimeter
        self.field = field
    
    def logical_form(self):
        return "InfSep(Field({}),By({}))".format(self.field.logical_form() ,self.delimiter.logical_form())

    def description(self):
        return "Seperated fields:\n{}".format(self.field.description())

    def specification(self):
        return "concat({0},star(concat({1},{0})))".format(self.field.specification(), self.delimiter.specification())

    def to_string(self):
        return "InfSeperatedField(Field({}),By({}))".format(self.field.to_string(),self.delimiter.to_string())
  
    def required_special_examples(self):
        return [
            "{0}".format(self.field.specification()),
            "concat({0},concat({1},{0}))".format(self.field.specification(), self.delimiter.specification()),
            "concat({0},repeat(concat({1},{0}),2))".format(self.field.specification(), self.delimiter.specification()),
            "concat({0},repeatatleast(concat({1},{0}),3))".format(self.field.specification(), self.delimiter.specification()),
        ]
    
    def negative_candidates(self):
        neg_candidates = []
        for new_f in self.field.negative_candidates():
            neg_candidates.append(InfSeperatedField(self.delimiter, new_f))
        return neg_candidates

class SeperatedField(Function):
    @staticmethod
    def uns_or_cat():
        return weighted_random_decision([SimpleUnstructuredField, SimpleConcatenationField],[1.0, 1.45])

    def __init__(self, delimeter, fields):
        super().__init__(delimeter,*fields)
        self.delimiter = delimeter
        self.fields = fields

    def logical_form(self):
        return "Sep(Fields({}),By({}))".format(",".join([x.logical_form() for x in self.fields]),self.delimiter.logical_form())

    def description(self):
        return "Three seperated fields by:\n{}".format(",\n" .join([x.logical_form() for x in self.fields]))

    def specification(self):
        if len(self.fields) == 3:
            return ConcatComp.concat_type_specification([self.fields[0], self.delimiter, self.fields[1], self.delimiter, self.fields[2]])
        else:
            return ConcatComp.concat_type_specification([self.fields[0], self.delimiter, self.fields[1]])

    def to_string(self):
        return "SeperatedField(Fields({}),By({}))".format(",".join([x.to_string() for x in self.fields]),self.delimiter.to_string())
   
    @classmethod
    def generate_related_fileds(cls, num):
        # determine type
        filed_type = cls.uns_or_cat()
        # filed_type = SimpleUnstructuredField
        if filed_type == SimpleUnstructuredField:
            num_cons = random.choice([1, 2, 3])
            template = SimpleUnstructuredField.generate(3)
            if num == 2:
                # return, no more cons allowed
                template = template.fit_num_cons(num_cons)
                return [SimpleUnstructuredField(*template.children) for i in range(num)]
            # num should be 3
            if num_cons == 1:
                # one or two of them may have aditional one or two cons
                r = random.random()
                if r < 0.33:
                    template = template.fit_num_cons(1)
                    return [SimpleUnstructuredField(*template.children) for i in range(num)]
                elif r < 0.66:
                    comp1template = template.fit_num_cons(1)
                    comp3template = template
                    return cls.random_mix(comp1template, comp3template, SimpleUnstructuredField)
                else:
                    comp2template = template.fit_num_cons(2)
                    comp1template = comp2template.fit_num_cons(1)
                    return cls.random_mix(comp2template, comp1template, SimpleUnstructuredField)
            elif num_cons == 2:
                r = random.random()
                if r < 0.5:
                    template = template.fit_num_cons(2)
                    return [SimpleUnstructuredField(*template.children) for i in range(num)]
                else:
                    comp2template = template.fit_num_cons(2)
                    comp3template = template
                    return cls.random_mix(comp2template, comp3template, SimpleUnstructuredField)
            elif num_cons == 3:
                # return, no more cons allowed
                return [SimpleUnstructuredField(*template.children) for i in range(num)]

        elif filed_type == SimpleConcatenationField:
            num_comp = random.choice([1, 2, 3])
            template = SimpleConcatenationField.generate(3)
            if num == 2:
                # return, no more cons allowed
                template = template.fit_num_comp(num_comp)
                return [SimpleConcatenationField(*template.children) for i in range(num)]

            # num should be 3
            if num_comp == 1:
                # one or two of them may have aditional one or two cons
                r = random.random()
                if r < 0.33:
                    template = template.fit_num_comp(1)
                    return [SimpleConcatenationField(*template.children) for i in range(num)]
                elif r < 0.66:
                    comp1template = template.fit_num_comp(1)
                    comp3template = template
                    return cls.random_mix(comp1template, comp3template, SimpleConcatenationField)
                else:
                    comp2template = template.fit_num_comp(2)
                    comp1template = comp2template.fit_num_comp(1)
                    return cls.random_mix(comp2template, comp1template, SimpleConcatenationField)
                # random.choice()
            elif num_comp == 2:
                # one or two of them may have aditional one or two cons
                r = random.random()
                if r < 0.5:
                    template = template.fit_num_comp(2)
                    return [SimpleConcatenationField(*template.children) for i in range(num)]
                else:
                    comp2template = template.fit_num_comp(2)
                    comp3template = template
                    return cls.random_mix(comp2template, comp3template, SimpleConcatenationField)
            elif num_comp == 3:
                # return, no more cons allowed
                template = template.fit_num_comp(num_comp)
                return [SimpleConcatenationField(*template.children) for i in range(num)]
        return [] * num

    @staticmethod
    def random_mix(temp1, temp2, cls):
        picked_id = random.choice([0, 1, 2])
        seq = [temp1, temp2]
        random.shuffle(seq)
        return [cls(*seq[0].children) if i == picked_id else cls(*seq[1].children) for i in range(3)]

    @classmethod
    def generate(cls):
        # delimiter = SingleToken.generate([SpecialToken], False)
        delimiter = SpecialToken.gen_div_tok()
        # all same fields
        SpecialToken.screen_tok(delimiter.tok)
        CC.remove(SpecialToken)
        if random_decision(0.175):
            # filed_type = random.choice([SimpleUnstructuredField, SimpleConcatenationField])
            # filed_type = weighted_random_decision([SimpleUnstructuredField, SimpleConcatenationField], [1.0, 1.45])
            filed_type = cls.uns_or_cat()
            field = filed_type.generate(random.choice([1, 2, 3]))
            sep_fields = [field]
        else:
            num_related_fileds = random.choice([1, 2, 3])
            # num_related_fileds = 3
            if num_related_fileds == 1:
                # three completely unrelated stuff
                print("unrelated")
                sep_fields = cls.gen_seprated_unrelated_fields()
            elif num_related_fileds == 2:
                print("two related")
                # two related, one I don't care
                sep_fields = cls.generate_related_fileds(2) # I'm done, leave blank
                # filed_type = random.choice([SimpleUnstructuredField, SimpleConcatenationField])
                filed_type = cls.uns_or_cat()
                leftout_one = filed_type.generate(random.choice([1,2]))
                if random_decision(0.5):
                    sep_fields = sep_fields + [leftout_one]
                else:
                    sep_fields =  [leftout_one] + sep_fields
            elif num_related_fileds == 3:
                print("all related")
                sep_fields = cls.generate_related_fileds(3)
    
        SpecialToken.restore()
        CC.append(SpecialToken)
        # CCToken
        if sep_fields:
            if len(sep_fields) == 1:
                return InfSeperatedField(delimiter, sep_fields[0])
            else:
                return cls(delimiter, sep_fields)
        return NoneToken()

    @classmethod
    def gen_seprated_unrelated_fields(cls):
        num_total_cons = random.choice(list(range(3,6)))
        
        seg_points = random.sample(list(range(1, num_total_cons)), 2)
        seg_points.sort()
        list_num_cons = []
        list_num_cons.append(min(3,seg_points[0]))
        list_num_cons.append(min(3,seg_points[1] - seg_points[0]))
        list_num_cons.append(min(3,num_total_cons - seg_points[1]))
        
        fields = []
        
        for num_cons in list_num_cons:
            # filed_type = random.choice([SimpleUnstructuredField, SimpleConcatenationField])
            filed_type = cls.uns_or_cat()
            fields.append(filed_type.generate(num_cons))

        return fields

    def sample_negative(self):  

        SpecialToken.screen_tok(self.delimiter.tok)
        CC.remove(SpecialToken)
        i = random.choice(range(len(self.fields)))
        new_fields = self.fields[:i] + [self.fields[i].sample_negative()] + self.fields[i+1:]
        SpecialToken.restore()
        CC.append(SpecialToken)

        return SeperatedField(self.delimiter, new_fields)
    
    def negative_candidates(self):
        possible_way = self.posible_negate_way()
        return [self.sample_negative() for _ in range(possible_way)]

    def posible_negate_way(self):
        return sum([x.posible_negate_way() for x in self.fields])

class SimpleUnstructuredField(UnstructuredField):

    def logical_form(self):
        return "And({})".format(",".join([x.logical_form() for x in self.children]))

    def description(self):
        if len(self.children) == 1:
            return self.children[0].description()
        else:
            return "Constrained by:\n{}".format(",\n" .join([x.description() for x in self.children]))

    @classmethod
    def generate(cls, MAX_CONS):
        # i = 0
        num_constrains = MAX_CONS
        constraint_types = cls.generate_constraint_types(num_constrains)
        arguments = []
        if ComposedByCons in constraint_types:
            cons, allowed = ComposedByCons.generate(3)
            constraint_types.remove(ComposedByCons)
            arguments.append(cons)
        else:
            allowed = None
        if LengthCons in constraint_types:
            constraint_types.remove(LengthCons)
            cons_type = random.choice([LengthOfCons, LengthBetweenCons, LengthLessThanCons, LengthMoreThanCons])
            if cons_type == LengthOfCons:
                cons = LengthOfCons.generate((3,5))
            elif cons_type == LengthBetweenCons:
                cons = LengthBetweenCons.generate((1,3),(3,5))
            elif cons_type == LengthLessThanCons:
                cons = LengthLessThanCons.generate((3,5))
            elif cons_type == LengthMoreThanCons:
                cons = LengthMoreThanCons.generate((2,4))
            arguments.append(cons)

        for cons_type in constraint_types:
            if cons_type in [ContainCons, StartwithCons, EndwithCons]:
                cons = cons_type.generate(allowed=allowed)
            else:
                cons = cons_type.generate(allowed)
            arguments.append(cons)
        if arguments:
            return cls(*arguments)
        return cls()

    @classmethod
    def generate_constraint_types(cls, num_cons):
        cons_types = []
        cons_types.append(ComposedByCons)
        num_cons -= 1
        if random_decision(0.4 if num_cons > 2 else 0.4):
            # cons_types.append(random.choice([LengthBetweenCons, LengthBeCons, LengthMoreThanCons, LengthLessThanCons]))
            cons_types.append(LengthCons)
            num_cons -= 1
        choices = [ContainCons, StartwithCons, EndwithCons,
                   NotContainCons, NotStartwithCons, NotEndwithCons]
        # for _ in range(num_cons):
            # cons_types.append(random.choice(choices))
        # return cons_types

        for _ in range(num_cons):
            picked_type = random.choice(choices)
            cons_types.append(picked_type)
            cls.resolve_avaliable_choices(choices, picked_type)
        return cons_types

    def fit_num_cons(self, num_cons):
        if len(self.children) == 3:
            if num_cons == 3:
                return self
            elif num_cons == 2:
                if random_decision(0.5):
                    return SimpleUnstructuredField(*[self.children[0], self.children[1]])
                else:
                    return SimpleUnstructuredField(*[self.children[0], self.children[2]])
            else:
                return SimpleUnstructuredField(*[self.children[0]])
        elif len(self.children) == 2:
            if num_cons == 3 or num_cons == 2:
                return self
            else:
                return SimpleUnstructuredField(*[self.children[0]])
        else:
            return self

class BoundedField(Function):
    POSSIBLE_COMBOS = [[SingleToken, SingleToken], [LetterToken, NumToken], [CapitalToken, NumToken], [LowerToken, NumToken], [CapitalToken, LowerToken]]
    COMBOS_WEIGHTS = [1.0, 2.0, 2.0, 2.0, 2.0]
    @classmethod
    def generate(cls, allow_notcc=True, allow_orcc=True):
        type_candidates = [SingleToken, SingleOrSingleToken, StringOrStringToken,
            RepeatMod, RepeatRangeMod, RepeatAtLeastMod]
        picked_type = weighted_random_decision(type_candidates, [0.05, 0.05, 0.05, 0.3, 0.3, 0.15])
    
        if picked_type == SingleToken:
            return SingleToken.generate(SINGLE_TOKEN_CHOICE), 1
        elif picked_type == SingleOrSingleToken:
            picked_cc = random.choice(CC)
            return OrComp(SingleToken.generate([picked_cc]), SingleToken.generate([picked_cc])), 1
        elif picked_type == StringOrStringToken:
            picked_cc = random.choice(CC)
            return OrComp(StringToken.generate([picked_cc]), StringToken.generate([picked_cc])), 2

        if allow_orcc and random_decision(0.05):
            # pass
            combo = weighted_random_decision(BoundedField.POSSIBLE_COMBOS, BoundedField.COMBOS_WEIGHTS)
            if combo[0] == SingleToken:
                picked_cc = random.choice(CC_NO_SPEC)
                tok0 = SingleToken.generate([picked_cc])
                tok1 = SingleToken.generate([picked_cc])
                if tok0.tok == tok1.tok:
                    return NoneToken, 2
            else:
                random.shuffle(combo)
                tok0 = combo[0]()
                tok1 = combo[1]()
            combo_type = weighted_random_decision(["inside", "outside"], [1.0, 2.0])
            if picked_type == RepeatMod:
                p0 = random.randint(1,4)
                if combo_type == "inside":
                    return RepeatMod(OrComp(tok0, tok1), p0), 2
                else:
                    return OrComp(RepeatMod(tok0, p0), RepeatMod(tok1, p0)), 2
            elif picked_type == RepeatRangeMod:
                p0 = random.randint(1, 3)
                p1 = random.randint(max(2, p0+1), 4)
                if combo_type == "inside":
                    return RepeatRangeMod(OrComp(tok0, tok1), p0, p1), 2
                else:
                    return OrComp(RepeatRangeMod(tok0, p0, p1),RepeatRangeMod(tok1, p0, p1)), 2
            elif picked_type == RepeatAtLeastMod:
                p0 = random.randint(1, 3)
                if combo_type == "inside":
                    return RepeatAtLeastMod(OrComp(tok0, tok1), p0), 2
                else:
                    return OrComp(RepeatAtLeastMod(tok0, p0), RepeatAtLeastMod(tok1, p0)), 2

        # repeat Something
        tok_type = random.choice(CC_NO_SPEC + [SingleToken])
        if tok_type in CC:
            tok = tok_type()
        else:
            tok = SingleToken.generate(CC_NO_SPEC)

        if (not isinstance(tok, SingleToken)) and random_decision(0.075) and allow_notcc:
            tok = NotCCCons(tok)

        if picked_type == RepeatMod:
            return RepeatMod.generate(tok, (1,4)), 1
        elif picked_type == RepeatRangeMod:
            return RepeatRangeMod.generate(tok, (1, 3), (2, 4)), 1
        elif picked_type == RepeatAtLeastMod:
            return RepeatAtLeastMod.generate(tok, (1, 3)), 1
        

class ConcatenationField(Function):
    def logical_form(self):
            return "Concat({})".format(",".join([x.logical_form() for x in self.children]))

    def description(self):
        if len(self.children) == 1:
            return self.children[0].description()
        else:
            return "Concatenation of:\n{}".format(",\n" .join([x.logical_form() for x in self.children]))

    def specification(self):
        return ConcatComp.concat_type_specification(self.children)

    @classmethod
    def generate(cls, MAX_COMP):
        num_components = weighted_random_decision([2,3,4,5,6], [1.75, 1.5, 1.25, 1.13, 1.12])
        if num_components < 3:
            components = cls.gen_randomly_concated_components(num_components)
        else:
            num_repeatable = random.choice(
                list(range(num_components // 2 + 1)))
            if num_repeatable == 0:
                components = cls.gen_randomly_concated_components(num_components)
            else:
                components = cls.gen_partialy_related_components(num_components, num_repeatable)

        if len(components) > 2:
            merg_prob =  1 - [1.0,1.0,0.875,0.915,0.875,0.875][len(components) - 1]
            # merg_prob = merg_prob
            if random_decision(merg_prob):
                if len(components) >= 3:
                    num_to_merge = weighted_random_decision([2, 3], [3.0, 1.0])
                else:
                    num_to_merge = 2
                # pick a position
                pos = random.choice(list(range(len(components) - num_to_merge + 1)))
                m_start = pos
                m_end = m_start + num_to_merge
                merge_comp = components[m_start:m_end]
                arguments_merged = components[:m_start] + [OptionalCons(ConcatComp(*merge_comp))] + components[m_end:]

                # make optional
                if arguments_merged:
                    return cls(*arguments_merged)
                return NoneToken()

        optional_prob = 1 - [1.0,0.85,0.885,0.925,0.93,0.95][len(components) - 1]
        arguments = []

        for comp in components:
            if random_decision(optional_prob):
                comp = OptionalCons(comp)
            arguments.append(comp)

        arguments_merged = []
        arguments_merged.append(arguments[0])
        for arg in arguments[1:]:
            if not isinstance(arg, OptionalCons):
                arguments_merged.append(arg)
                continue
            last = arguments_merged[-1]
            if isinstance(last, OptionalCons) and random_decision(0.85) and len(arguments) > 2:
                arguments_merged.pop()
                if isinstance(last.children[0], ConcatComp):
                    new_children = last.children[0].children + [arg.children[0]]
                else:
                    new_children = [last.children[0], arg.children[0]]
                arguments_merged.append(OptionalCons(
                    ConcatComp(*new_children)))
            else:
                arguments_merged.append(arg)

        # make optional
        if arguments_merged:
            return cls(*arguments_merged)
        return NoneToken()

    @classmethod
    def gen_randomly_concated_components(cls, num_comp):
        arguments = []
        i = 0
        while i < num_comp:
            allow_orcc = (i + 2) <= num_comp
            cons, i_complexity = BoundedField.generate(allow_notcc=allow_orcc)
            arguments.append(cons)
            i += i_complexity
        return arguments

    @classmethod
    def gen_partialy_related_components(cls, num_comp, num_repeatable):
        repeating_comp = cls.gen_repeating_components(num_repeatable)
        # print("\t", repeating_comp)
        
        # special casese
        if num_repeatable * 2 == num_comp:
            if random_decision(0.6):
                return repeating_comp + repeating_comp
            else:
                return cls.gen_randomly_concated_components(num_comp)

        i = 0
        just_repeated = False
        arguments = []
        showed_times = 0
        while i < num_comp:
            # decide whether we can repeat here
            if num_repeatable == 1:
                can_repeat = not just_repeated
            else:
                can_repeat = ((i + num_repeatable) <= num_comp)

            if can_repeat:
                if num_comp == 3:
                    repeat_prob = 0.95
                elif num_comp == 4:
                    repeat_prob = 0.6
                elif num_comp == 5:
                    repeat_prob = 0.55
                else:
                    repeat_prob = 0.375
                if random_decision(repeat_prob):
                    arguments = arguments + repeating_comp
                    i += num_repeatable
                    just_repeated = True
                    showed_times += 1
                else:
                    allow_orcc = (i + 2) <= num_comp
                    comp, i_complexity = BoundedField.generate(allow_orcc=allow_orcc)
                    arguments.append(comp)
                    i += i_complexity
                    just_repeated = False
            else:
                allow_orcc = (i + 2) <= num_comp
                comp, i_complexity = BoundedField.generate(allow_orcc=allow_orcc)
                arguments.append(comp)
                i += i_complexity
                just_repeated = False
    
        return arguments

    @classmethod
    def gen_repeating_components(cls, num_repeatable):
        # if num_repeatable == 1:
        #     # repeat, repeat range, cc
        #     type_candidates = [CCToken, RepeatMod, RepeatRangeMod]
        #     picked_type = random.choice(type_candidates)
        #     if picked_type == CCToken:
        #         picked_cc = random.choice(CC_NO_SPEC)
        #         return [picked_cc()]
        #     elif picked_type == RepeatMod:
        #         tok_type = random.choice(CC_NO_SPEC)
        #         tok = tok_type()
        #         return [RepeatMod.generate(tok, (2,3))]
        #     elif picked_type == RepeatRangeMod:
        #         tok_type = random.choice(CC_NO_SPEC)
        #         tok = tok_type()
        #         return [RepeatRangeMod.generate(tok, (1, 2), (2, 3))]
        # else:
        #     return cls.gen_randomly_concated_components(num_repeatable)
        return cls.gen_randomly_concated_components(num_repeatable)
    
    def sample_negative(self):
        i = random.choice(range(len(self.children)))
        new_children = self.children[:i] + [self.children[i].sample_negative()] + self.children[i+1:]
        if len(new_children) == 0:
            return NoneToken()
        return ConcatenationField(*new_children)

    def negative_candidates(self):
        possible_way = self.posible_negate_way()
        return [self.sample_negative() for _ in range(possible_way)]

    def posible_negate_way(self):
        return 2 * len(self.children)

class SimpleConcatenationField(ConcatenationField):

    def logical_form(self):
        return "Concat({})".format(",".join([x.logical_form() for x in self.children]))

    def description(self):
        if len(self.children) == 1:
            return self.children[0].description()
        else:
            return "Concatenation of:\n{}".format(",\n" .join([x.logical_form() for x in self.children]))

    @classmethod
    def generate(cls, MAX_CONS):
        # i = 0
        arguments = []
        num_components = MAX_CONS
        i = 0
        while i < num_components:
            allow_orcc = (i + 2) <= num_components
            comp, i_complexity = BoundedField.generate(allow_notcc=False, allow_orcc=allow_orcc)
            if random_decision(0.15):
                comp = OptionalCons(comp)
            arguments.append(comp)
            i += i_complexity

        arguments_merged = []
        arguments_merged.append(arguments[0])
        for arg in arguments[1:]:
            if not isinstance(arg, OptionalCons):
                arguments_merged.append(arg)
                continue
            last = arguments_merged[-1]
            if isinstance(last, OptionalCons) and random_decision(0.75) and len(arguments) > 2:
                arguments_merged.pop()
                if isinstance(last.children[0], ConcatComp):
                    new_children = last.children[0].children + [arg.children[0]]
                else:
                    new_children = [last.children[0], arg.children[0]]
                arguments_merged.append(OptionalCons(ConcatComp(*new_children)))
            else:
                arguments_merged.append(arg)
        # make optional
        if arguments_merged:
            return cls(*arguments_merged)
        return NoneToken()

    def fit_num_comp(self, num_comp):
        if len(self.children) == 3:
            if num_comp == 3:
                return self
            elif num_comp == 2:
                if random_decision(0.5):
                    return SimpleConcatenationField(*self.children[:2])
                else:
                    return SimpleConcatenationField(*self.children[1:])
            else:
                id = random.choice([0, 1, 2])
                return SimpleConcatenationField(*self.children[id:id+1])
        elif len(self.children) == 2:
            if num_comp == 3 or num_comp == 2:
                return self
            else:
                if random_decision(0.5):
                    return SimpleConcatenationField(*self.children[:1])
                else:
                    return SimpleConcatenationField(*self.children[1:])
        else:
            return self
