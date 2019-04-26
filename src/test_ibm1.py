from ibm1 import *

sentence_pair = (['a','b'], ['1','2','3'])

def get_model():
    return IBM1({'a','b'})

def test_ctor():
    model = get_model()
    assert model.translation_probabilities['a']['b'] == 0.5

def test_total_log_likelihood():
    model = get_model()
    assert model.total_log_likelihood([sentence_pair]) == -1.3862943611198906

def test_pair_log_likelihood():
    model = get_model()
    assert model.pair_log_likelihood( sentence_pair) == -1.3862943611198906

def test_log_likelihood():
    model = get_model()
    assert model.     log_likelihood(*sentence_pair)[0] == -1.3862943611198908

def test_train():
    model = get_model()
    assert model.train([(['a'],['1'])], 1, [(['a'],['5'])], [({1},{1})], [(['a'],['5'])], [({1},{1})], 'test') == ([-0.6931471805599453], [1.0])

def test_print_dictionary():
    model = get_model()
    assert model.print_dictionary() is None

def test_align():
    model = get_model()
    assert model.align(('a','5')) == {(1,1)}

def test_calculate_aer():
    model = get_model()
    assert model.calculate_aer([('a','5')], [({1}, {3})]) == 1.0
