from typing import Tuple, List, Set, Dict, DefaultDict
from collections import defaultdict
import math
from ibm1 import IBM1
from tqdm import tqdm, trange
import yaml
from tensorboardX import SummaryWriter

class IBM2Jump(IBM1):
    def __init__(self, english_vocab, translation_probabilities = None):
        super(IBM2Jump, self).__init__(english_vocab, translation_probabilities)
        self.jump_counts = defaultdict(lambda: 1)

    def log_likelihood(self, target_sentence: List[str], foreign_sentence: List[str]) -> Tuple[float, List[float]]:
        """Calculate target_token likelihoods and log-likelihood of sentence pair"""
        len_target = len(target_sentence)
        len_source = len(foreign_sentence)
        log_likelihood = 0
        target_likelihoods = defaultdict(lambda: 0)

        for target_idx, target_token in enumerate(target_sentence):    
            normalizer = 0 # Normalize to have sensible log_likelihood
            for source_idx, source_token in enumerate(foreign_sentence):
                normalizer += self.jump_counts[source_idx - int(target_idx*len_source/len_target)]
            for source_idx, source_token in enumerate(foreign_sentence):
                translation_probability = self.translation_probabilities[target_token][source_token]
                alignment_probabilility = self.jump_counts[source_idx - int(target_idx*len_source/len_target)]/normalizer
                target_likelihoods[target_token] += translation_probability*alignment_probabilility
            log_likelihood += math.log(target_likelihoods[target_token])
        return (log_likelihood, target_likelihoods)

    def train(self, training_corpus: List[Tuple[str, str]], iterations: int, validation_corpus: List[Tuple[str, str]], validation_gold: List[List[Tuple[int, int]]]) -> Tuple[List[int], List[int]]:
        """Train model"""
        total_log_likelihoods = []
        aer_scores = []
        with SummaryWriter(self.__class__.__name__) as w:
            for i in trange(iterations):
                expected_count = defaultdict(lambda: defaultdict(lambda: 0)) # Expected number of times target_token is connected to source_token 
                expected_total = defaultdict(lambda: 0) # Expected total connections for source_token
                expected_jump_count = defaultdict(lambda: 0) # Expected connections with a certain jump length
                total_log_likelihood = 0

                # Calculate expected counts (Expectation step) and log-likelihood
                for t in range(len(training_corpus)):
                    # Expand sentence pair
                    target_sentence, foreign_sentence = training_corpus[t]

                    # Add null token
                    foreign_sentence = [None] + foreign_sentence

                    len_target = len(target_sentence)
                    len_source = len(foreign_sentence)

                    # Calculate target_token likelihoods and log-likelihood of sentence pair
                    log_likelihood, target_likelihoods = self.log_likelihood(target_sentence, foreign_sentence)
                    total_log_likelihood += log_likelihood

                    # Collect counts
                    for target_idx, target_token in enumerate(target_sentence):
                        normalizer = 0
                        for source_idx, source_token in enumerate(foreign_sentence):
                            normalizer += self.jump_counts[source_idx - int(target_idx*len_source/len_target)]
                        for source_idx, source_token in enumerate(foreign_sentence):
                            translation_probability = self.translation_probabilities[target_token][source_token]
                            alignment_probabilility = self.jump_counts[source_idx - int(target_idx*len_source/len_target)]/normalizer
                            normalized_count = translation_probability*alignment_probabilility/target_likelihoods[target_token]
                            expected_count[target_token][source_token] += normalized_count
                            expected_total[source_token] += normalized_count
                            expected_jump_count[source_idx - int(target_idx*len_source/len_target)] += normalized_count

                # Update translation probabilities (Maximization step)
                for target_token in expected_count.keys():
                    for source_token in expected_count[target_token].keys():
                        self.translation_probabilities[target_token][source_token] = expected_count[target_token][source_token]/expected_total[source_token]

                # Update jump counts (Maximization step)
                for jump in expected_jump_count.keys():
                    self.jump_counts[jump] = expected_jump_count[jump]

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

    def align(self, pair: Tuple[str, str]) -> Set[Tuple[int, int]]:
        """Find best alignment for a sentence pair"""
        # Expand sentence pair
        target_sentence, foreign_sentence = pair
        len_target = len(target_sentence)
        len_source = len(foreign_sentence)

        # Initialize alignment
        alignment = set()

        # Add best link for every english word
        for target_idx, target_token in enumerate(target_sentence):
            # Default alignment with null token
            best_align = None
            best_prob = self.translation_probabilities[target_token][None]*self.jump_counts[0 - int(target_idx*len_source/len_target)] 

            # Check alignments with all other possible words
            for source_idx, source_token in enumerate(foreign_sentence):
                prob = self.translation_probabilities[target_token][source_token]*self.jump_counts[source_idx - int(target_idx*len_source/len_target)] 
                if prob > best_prob:  # prefer newer word in case of tie
                    best_align = source_idx + 1 
                    best_prob = prob
            alignment.add((target_idx+1, best_align))

        return alignment
