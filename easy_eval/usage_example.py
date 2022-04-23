from eval import check_equiv, check_io_consistency
from streg_utils import parse_spec_to_ast

# check equivalance
print('EXPECTED TRUE', check_equiv('or(<low>,<cap>)', 'or(<cap>,<low>)'))
print('EXPECTED TRUE',check_equiv('or(<low>,<cap>)', '<let>'))
print('EXPECTED FALSE',check_equiv('cat(<low>,<cap>)', 'cat(<cap>,<low>)'))

# check example consistency

spec = 'and(repeatatleast(or(<low>,or(<cap>,<^>)),1),and(not(startwith(<low>)),and(not(startwith(<^>)),not(contain(concat(notcc(<low>),<^>))))))'
good_examples = [('Vg^Peg^', '+'), ('CQzyPeqLgZFMpo^c^', '+'), ('UMoaHHz^', '+'), ('KyugLn^h^', '+'), ('CAhPZtBtvfCcuLtBvnR', '+'), ('MFt^Qh^czg^SMwWg', '+'), (('Vg^Peg^', '+'), '-'), (('CQzyPeqLgZFMpo^c^', '+'), '-'), (('UMoaHHz^', '+'), '-'), (('KyugLn^h^', '+'), '-'), (('CAhPZtBtvfCcuLtBvnR', '+'), '-'), (('MFt^Qh^czg^SMwWg', '+'), '-')]
bad_examples = [(';PP;p!mC!FB;;;TI', '+'), (';!;V!RDq;;;X', '+'), (';X;;;I', '+'), (';MW!usG!FEPBiTP;;;P', '+'), ('PnPw;UaPQPYPY;;;X', '+'), ('SRP!P!;;;AN', '+'), ((';PP;p!mC!FB;;;TI', '+'), '-'), ((';!;V!RDq;;;X', '+'), '-'), ((';X;;;I', '+'), '-'), ((';MW!usG!FEPBiTP;;;P', '+'), '-'), (('PnPw;UaPQPYPY;;;X', '+'), '-'), (('SRP!P!;;;AN', '+'), '-')]
print('EXPECTED TRUE',check_io_consistency(spec, good_examples))
print('EXPECTED FALSE',check_io_consistency(spec, bad_examples))


# skeleton for converting to standard regex
ast = parse_spec_to_ast(spec)
print(ast.logical_form())
std_regex = ast.standard_regex()
