from typing import Tuple, List, Set, Dict, Any, DefaultDict
from collections import defaultdict
import math
import os
from aer import AERSufficientStatistics
from tqdm import tqdm, trange
import yaml
from tensorboardX import SummaryWriter
from functools import reduce
import numpy as np
from pdb import set_trace

class IBM1():
    def __init__(self, vocab_target: Set[str], translation_probabilities: DefaultDict[str, DefaultDict[str, int]] = None, sampling_method = 'uniform', seed=42):
        self.vocab_target = vocab_target
        n = len(vocab_target)
        default_probability = 1/n
        if translation_probabilities is None:
            if sampling_method == 'random':
                np.random.seed(seed)
                a = np.random.rand(n)
                probabilities = a / np.sum(a)
                self.translation_probabilities = defaultdict(lambda: defaultdict(lambda: default_probability), {k: defaultdict(lambda: p) for k, p in zip(vocab_target, probabilities)})
            else:
                self.translation_probabilities = defaultdict(lambda: defaultdict(lambda: default_probability))
        else:
            self.translation_probabilities = translation_probabilities

    def total_log_likelihood(self, corpus: List[Tuple[List[str], List[str]]]) -> int:
        """Calculate log-likelihood of entire corpus"""
        print('\nCalculating log-likelihood')
        return sum(map(self.pair_log_likelihood, tqdm(corpus, desc='corpus')))

    def pair_log_likelihood(self, pair: Tuple[List[str], List[str]]) -> int:
        # Expand sentence pair
        target_sentence, source_sentence = pair

        # Add null token
        source_sentence = [None] + source_sentence

        # Calculate log_likelihood of pair
        log_likelihood, _ = self.log_likelihood(target_sentence, source_sentence)
        return log_likelihood

    def log_likelihood(self, target_sentence: List[str], source_sentence: List[str]) -> int:    
        """Calculate target_token-likelihoods and log-likelihood of sentence pair"""
        target_likelihoods = defaultdict(lambda: 0, {target_token: sum([
                self.translation_probabilities[target_token][source_token]
            for source_token in source_sentence]) for target_token in target_sentence})
        log_likelihood = -math.log(len(source_sentence)**len(target_sentence)) + \
            sum(map(math.log, target_likelihoods.values()))
        return (log_likelihood, target_likelihoods)

    def train(self, training_corpus: List[Tuple[str, str]], iterations: int, validation_corpus: List[Tuple[str, str]], validation_gold: List[List[Tuple[Set[int], Set[int]]]], test_corpus: List[Tuple[str, str]], test_gold: List[List[Tuple[Set[int], Set[int]]]], name: str = None) -> Tuple[List[int], List[int]]:
        """Train model"""
        if not name:
            name = self.__class__.__name__        
        total_log_likelihoods = []
        aer_scores = []

        with SummaryWriter(self.__class__.__name__) as w:
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
                    log_likelihood, target_likelihoods = self.log_likelihood(target_sentence, source_sentence)
                    total_log_likelihood += log_likelihood

                    # Collect counts
                    for target_token in target_sentence:
                        for source_token in source_sentence:
                            normalized_count = self.translation_probabilities[target_token][source_token]/target_likelihoods[target_token]
                            expected_count[target_token][source_token] += normalized_count
                            expected_total[source_token] += normalized_count
                
                # Update translation probabilities (Maximization step)
                for target_token in expected_count.keys():
                    for source_token in expected_count[target_token].keys():
                        self.translation_probabilities[target_token][source_token] = expected_count[target_token][source_token]/expected_total[source_token]

                average_log_likelihood = total_log_likelihood/len(training_corpus)
                val_aer = self.calculate_aer(validation_corpus, validation_gold)
                test_aer = self.calculate_aer(test_corpus, test_gold)

                stats = {
                    'logp': average_log_likelihood,
                    'val_aer': val_aer,
                    'test_aer': test_aer,
                }
                print(yaml.dump(stats))
                w.add_scalars('metrics', stats, i)
                total_log_likelihoods.append(average_log_likelihood)
                aer_scores.append(val_aer)
        self.write_naacl(test_corpus, test_gold, name)
        return (total_log_likelihoods, aer_scores)

    def print_dictionary(self) -> None:
        """Print most likely translation for each foreign word"""
        for target_token in self.vocab_target:
            probs = self.translation_probabilities[target_token]
            print(target_token, max(zip(probs.values(), probs.keys())) if probs else (0.0, None))

    def align(self, pair: Tuple[str, str]) -> Set[Tuple[int, int]]:
        """Find best alignment for a sentence pair"""
        target_sentence, source_sentence = pair
        source_sentence = [None] + source_sentence
        aligns = map(lambda target_token: self.best_align(source_sentence, target_token), target_sentence)

        # 1-indexed because AER is into that...
        test = set(map(lambda k_v: (k_v[1] if k_v[1] is not None else None, k_v[0]+1), enumerate(aligns))) 
        return test

    def best_align(self, source_sentence: List[str], target_token: str) -> int:
        """Find best alignment for a target token from a source sentence"""
        probs = list(map(lambda source_token: self.translation_probabilities[target_token][source_token], source_sentence))
        # np.argmax errors on empty list
        if probs:
            return np.argmax(probs)
        else:
            return None

    def calculate_aer(self, validation_corpus: List[Tuple[str, str]], validation_gold: List[List[Tuple[Set[int], Set[int]]]]) -> float:
        """Calculate AER on validation corpus using gold standard"""
        predictions = map(self.align, validation_corpus)

        # Compute AER
        metric = AERSufficientStatistics()
        for gold, pred in zip(validation_gold, predictions):
            (sure, probable) = gold
            metric.update(sure=sure, probable=probable, predicted=pred)
        return metric.aer()

    def write_naacl(self, validation_corpus: List[Tuple[str, str]], validation_gold: List[List[Tuple[Set[int], Set[int]]]], name: str) -> float:
        """Calculate AER on validation corpus using gold standard"""
        predictions = map(self.align, validation_corpus)
        lines = []
        # set_trace()
        for sentence_idx, pred in enumerate(predictions):
            for idx_pair in pred:
                (source_token_idx, target_token_idx) = idx_pair
                sentence_id = '%04d' % (sentence_idx + 1)
                status = 'S'
                line = ' '.join(map(str, [sentence_id, source_token_idx, target_token_idx or 0, status]))
                lines.append(line)
        naacl_file = os.path.join('..', 'Results', 'naacl', f'{name}.naacl')
        with open(naacl_file, 'w') as f:
            f.write('\n'.join(lines))
