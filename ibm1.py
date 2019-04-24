from typing import Tuple, List, Set, Dict, Any, DefaultDict
from collections import defaultdict
import math
from aer import AERSufficientStatistics
from tqdm import tqdm, trange
import yaml
from tensorboardX import SummaryWriter
from functools import reduce
import numpy as np

class IBM1():
    def __init__(self, vocab_target: Set[str], translation_probabilities: DefaultDict[str, DefaultDict[str, int]] = None):
        self.vocab_target = vocab_target        
        if translation_probabilities is None:
            default_probability = 1/len(vocab_target)
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

    def train(self, training_corpus: List[Tuple[str, str]], iterations: int, validation_corpus: List[Tuple[str, str]], validation_gold: List[List[Tuple[Set[int], Set[int]]]]) -> Tuple[List[int], List[int]]:
        """Train model"""
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

                aer = self.calculate_aer(validation_corpus, validation_gold)
                stats = {
                    'logp': total_log_likelihood,
                    'aer': aer,
                }
                print(yaml.dump(stats))
                w.add_scalars('metrics', stats, i)
                total_log_likelihoods.append(total_log_likelihood/len(training_corpus))
                aer_scores.append(aer)
        return (total_log_likelihoods, aer_scores)

    def print_dictionary(self) -> None:
        """Print most likely translation for each foreign word"""
        for target_token in self.vocab_target:
            probs = self.translation_probabilities[target_token]
            print(target_token, max(zip(probs.values(), probs.keys())) if probs else (0.0, None))

    def align(self, pair: Tuple[str, str]) -> Set[Tuple[int, int]]:
        """Find best alignment for a sentence pair"""
        # Expand sentence pair
        target_sentence, source_sentence = pair

        # Initialize alignment
        alignment = set()

        # Add best link for every english word
        for target_idx, target_token in enumerate(target_sentence):
            # Default alignment with null token
            best_align = None
            best_prob = self.translation_probabilities[target_token][None]     

            # Check alignments with all other possible words
            for source_idx, source_token in enumerate(source_sentence):
                prob = self.translation_probabilities[target_token][source_token] 
                if prob >= best_prob:  # prefer newer word in case of tie
                    best_align = source_idx + 1 
                    best_prob = prob
            alignment.add((target_idx+1, best_align))

        return alignment

    def calculate_aer(self, validation_corpus: List[Tuple[str, str]], validation_gold: List[List[Tuple[Set[int], Set[int]]]]) -> float:
        """Calculate AER on validation corpus using gold standard"""
        predictions = map(self.align, validation_corpus)

        # Compute AER
        metric = AERSufficientStatistics()
        for gold, pred in zip(validation_gold, predictions):
            (sure, probable) = gold
            metric.update(sure=sure, probable=probable, predicted=pred)
        return metric.aer()
