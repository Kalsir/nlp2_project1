from dataclasses import dataclass
from typing import Tuple, List, Set, Dict, DefaultDict, Callable
from collections import defaultdict
import math
from aer import AERSufficientStatistics
from tqdm import tqdm, trange
import yaml
from tensorboardX import SummaryWriter
from functools import reduce
import numpy as np
from pdb import set_trace

@dataclass(frozen=True)
class ModelData():
    """data to be used in IBM models"""
    vocab_target: Set[str]
    translation_probabilities: DefaultDict[str, DefaultDict[str, int]]

# @dataclass(frozen=True)
# class IBMModel():
#     """IBM model, implementing the required methods"""
#     # train(model: ModelData, training_corpus: List[Tuple[str, str]], iterations: int, validation_corpus: List[Tuple[str, str]], validation_gold: List[List[Tuple[Set[int], Set[int]]]], log_likelihood_fn: Callable[[ModelData, str, str], int]) -> Tuple[List[int], List[int]]
#     train: Callable[[ModelData, List[Tuple[str, str]], int, List[Tuple[str, str]], List[List[Tuple[Set[int], Set[int]]]], Callable[[ModelData, str, str], int]], Tuple[List[int], List[int]]]
#     # align(target_sentence: str, source_sentence: str, translation_probabilities: DefaultDict[str, DefaultDict[str, int]]) -> Set[Tuple[int, int]]
#     align: Callable[[str, str, DefaultDict[str, DefaultDict[str, int]]], Set[Tuple[int, int]]]
#     # log_likelihood(model: ModelData, target_sentence: str, source_sentence: str) -> int
#     log_likelihood: Callable[[ModelData, str, str], int]

class IBMModel():
    """IBM model, implementing the required methods"""

    def train(self, model: ModelData, training_corpus: List[Tuple[str, str]], iterations: int, validation_corpus: List[Tuple[str, str]], validation_gold: List[List[Tuple[Set[int], Set[int]]]], log_likelihood_fn: Callable[[ModelData, str, str], int]) -> Tuple[List[int], List[int]]:
        raise NotImplementedError

    def align(self, target_sentence: str, source_sentence: str, translation_probabilities: DefaultDict[str, DefaultDict[str, int]]) -> Set[Tuple[int, int]]:
        raise NotImplementedError

    def log_likelihood(self, model: ModelData, target_sentence: str, source_sentence: str) -> int:
        raise NotImplementedError

# ModelData functions

def total_log_likelihood(corpus: List[Tuple[str, str]], ibm_model: IBMModel, model: ModelData) -> int:
    """Calculate log-likelihood of entire corpus"""
    return sum(map(lambda pair: ibm_model.log_likelihood(model, [None] + pair[0], pair[1])[0], tqdm(corpus, desc='corpus')))

def print_dictionary(vocab_target: Set[str], translation_probabilities: DefaultDict[str, DefaultDict[str, int]]) -> None:
    """Print most likely translation for each foreign word"""
    for target_token in vocab_target:
        probs = translation_probabilities[target_token]
        print(target_token, max(zip(probs.values(), probs.keys())) if probs else (0.0, None))

def calculate_aer(validation_corpus: List[Tuple[str, str]], validation_gold: List[List[Tuple[int, int]]], translation_probabilities: DefaultDict[str, DefaultDict[str, int]], ibm_model: IBMModel) -> float:
    """Calculate AER on validation corpus using gold standard"""
    predictions = map(lambda pair: ibm_model.align(*pair, translation_probabilities), validation_corpus)

    # Compute AER
    metric = AERSufficientStatistics()
    for gold, pred in zip(validation_gold, predictions):
        (sure, probable) = gold
        metric.update(sure=sure, probable=probable, predicted=pred)
    return metric.aer()

# IBM1

def _best_align(source_sentence: List[str], probs: Dict[str, float]) -> int:
    """Find best alignment for a target token from a source sentence"""
    probs = map(lambda source_token: probs[source_token], source_sentence)
    # np.argmax errors on empty list
    return (np.argmax(probs)+1) if probs else None

class IBM1(IBMModel):

    def log_likelihood(self, model: ModelData, target_sentence: List[str], source_sentence: List[str]) -> int:
        """Calculate target_token-likelihoods and log-likelihood of sentence pair"""
        target_likelihoods = defaultdict(lambda: 0, {target_token: sum([
                model.translation_probabilities[target_token][source_token]
            for source_token in source_sentence]) for target_token in target_sentence})
        log_likelihood = -math.log(len(source_sentence)**len(target_sentence)) + \
            sum(map(math.log, target_likelihoods.values()))
        return (log_likelihood, target_likelihoods)

    def train(self, model: ModelData, training_corpus: List[Tuple[str, str]], iterations: int, validation_corpus: List[Tuple[str, str]], validation_gold: List[List[Tuple[Set[int], Set[int]]]]) -> Tuple[List[int], List[int]]:
        """Train model"""
        total_log_likelihoods = []
        aer_scores = []

        with SummaryWriter('ibm1') as w:
            for i in trange(iterations, desc='iteration', position=0):
                # Expected number of times target_token is connected to source_token
                expected_count = defaultdict(lambda: defaultdict(lambda: 0))
                # Expected total connections for source_token
                expected_total = defaultdict(lambda: 0)
                total_log_likelihood = 0

                # Calculate expected counts (Expectation step) and log-likelihood
                for pair in tqdm(training_corpus, desc='training_corpus', position=1):
                    # Expand sentence pair
                    target_sentence, source_sentence = pair

                    # Add null token
                    source_sentence = [None] + source_sentence        

                    # Calculate target_token-likelihoods and log-likelihood of sentence pair
                    log_likelihood, target_likelihoods = self.log_likelihood(model, target_sentence, source_sentence)
                    total_log_likelihood += log_likelihood

                    # Collect counts
                    for target_token in target_sentence:
                        for source_token in source_sentence:
                            normalized_count = model.translation_probabilities[target_token][source_token]/target_likelihoods[target_token]
                            expected_count[target_token][source_token] += normalized_count
                            expected_total[source_token] += normalized_count
                
                # Update translation probabilities (Maximization step)
                for target_token in expected_count.keys():
                    for source_token in expected_count[target_token].keys():
                        model.translation_probabilities[target_token][source_token] = expected_count[target_token][source_token]/expected_total[source_token]

                aer = calculate_aer(validation_corpus, validation_gold, model.translation_probabilities, self)
                stats = {
                    'logp': total_log_likelihood,
                    'aer': aer,
                }
                print(yaml.dump(stats))
                w.add_scalars('metrics', stats, i)
                total_log_likelihoods.append(total_log_likelihood/len(training_corpus))
                aer_scores.append(aer)
        return (total_log_likelihoods, aer_scores)

    def align(self, target_sentence: List[str], source_sentence: List[str], translation_probabilities: DefaultDict[str, DefaultDict[str, int]]) -> Set[Tuple[int, int]]:
        """Find best alignment for a sentence pair"""
        aligns = map(lambda target_token: _best_align(source_sentence, translation_probabilities[target_token]), target_sentence)
        # 1-indexed because AER is into that...
        return set(map(lambda k_v: (k_v[0]+1, k_v[1]+1 if k_v[1] else None), enumerate(aligns)))

    def make_data(self, vocab_target: Set[str], translation_probabilities: DefaultDict[str, DefaultDict[str, int]] = None):
        if translation_probabilities is None:
            default_probability = 1/len(vocab_target)
            translation_probabilities = defaultdict(lambda: defaultdict(lambda: default_probability))
        return ModelData(vocab_target, translation_probabilities)
