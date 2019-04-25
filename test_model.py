from collections import defaultdict
from model import *

sentence_pair = (['a','b'], ['1','2','3'])
data = ModelData({'a','b'}, defaultdict(lambda: defaultdict(lambda: 0.5)))

def test_ctor():
    assert data.translation_probabilities['a']['b'] == 0.5

def test_total_log_likelihood():
    assert total_log_likelihood([sentence_pair], ibm1, data) == -2.079441541679836

def test_print_dictionary():
    assert print_dictionary({'a','b'}, defaultdict(lambda: defaultdict(lambda: 0), {'a': defaultdict(lambda: 1, {'1': 1})})) is None

def test_calculate_aer():
    assert calculate_aer([('a','5')], [({1}, {3})]) == 1.0

# ibm1

def test_train():
    assert ibm1.train([(['a'],['1'])], 1, [(['a'],['5'])], [({1},{1})], ibm1.log_likelihood, ibm1.log_likelihood) == ([-0.6931471805599453], [1.0])

def test_align():
    assert ibm1.align(('a','5')) == {(1,None)}

def test_log_likelihood():
    assert ibm1.log_likelihood(data, *sentence_pair)[0] == -1.3862943611198908
