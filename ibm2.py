from collections import defaultdict
import math
from ibm1 import IBM1

class AlignmentProbabilities():
    def __init__(self):
        self.alignment_probabilities = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: dict())))

    def read(self, len_target, len_source, target_idx, source_idx):
        if source_idx not in self.alignment_probabilities[len_target][len_source][target_idx]:
            self.alignment_probabilities[len_target][len_source][target_idx][source_idx] = 1/10
        return self.alignment_probabilities[len_target][len_source][target_idx][source_idx]

    def write(self, len_target, len_source, target_idx, source_idx, value):
        self.alignment_probabilities[len_target][len_source][target_idx][source_idx] = value

class IBM2(IBM1):
    def __init__(self, vocab_target, translation_probabilities = None):
        super(IBM2, self).__init__(vocab_target, translation_probabilities)
        self.alignment_probabilities = AlignmentProbabilities()

    # Calculate target_token likelihoods and log-likelihood of sentence pair
    def log_likelihood(self, target_sentence, foreign_sentence):    
        len_target = len(target_sentence)
        len_source = len(foreign_sentence)
        log_likelihood = 0
        target_likelihoods = defaultdict(lambda: 0)
        for target_idx, target_token in enumerate(target_sentence):                
            for source_idx, source_token in enumerate(foreign_sentence):
                translation_probability = self.translation_probabilities[target_token][source_token]
                alignment_probabilility = self.alignment_probabilities.read(len_target, len_source, target_idx, source_idx)
                target_likelihoods[target_token] += translation_probability*alignment_probabilility
            log_likelihood += math.log(target_likelihoods[target_token])
        return (log_likelihood, target_likelihoods)

    # Train model
    def train(self, training_corpus, iterations, validation_corpus, validation_gold):
        total_log_likelihoods = []
        aer_scores = []
        for i in tqdm(range(iterations)):
            print('\nStarting iteration', i+1)
            expected_count = defaultdict(lambda: defaultdict(lambda: 0)) # Expected number of times target_token is connected to source_token 
            expected_total = defaultdict(lambda: 0) # Expected total connections for source_token
            len_expected_count = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))) # Expected number of times target_token is connected to source_token in pairs of specific length
            len_expected_total = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0))) # Expected number of total connections for source_token in pairs of specific length
            total_log_likelihood = 0

            # Calculate expected counts (Expectation step) and log-likelihood
            for t in range(len(training_corpus)):
                # Expand sentence pair
                target_sentence, foreign_sentence = training_corpus[t]

                # Add null token
                foreign_sentence = [None] + foreign_sentence

                # Calculate sentence lenghts 
                len_target = len(target_sentence)
                len_source = len(foreign_sentence)                    

                # Calculate target_token likelihoods and log-likelihood of sentence pair
                log_likelihood, target_likelihoods = self.log_likelihood(target_sentence, foreign_sentence)
                total_log_likelihood += log_likelihood

                # Collect counts
                for target_idx, target_token in enumerate(target_sentence):
                    for source_idx, source_token in enumerate(foreign_sentence):
                        translation_probability = self.translation_probabilities[target_token][source_token]
                        alignment_probabilility = self.alignment_probabilities.read(len_target, len_source, target_idx, source_idx)
                        normalized_count = translation_probability*alignment_probabilility/target_likelihoods[target_token]
                        expected_count[target_token][source_token] += normalized_count
                        expected_total[source_token] += normalized_count
                        len_expected_count[len_target][len_source][target_idx][source_idx] += normalized_count
                        len_expected_total[len_target][len_source][source_idx] += normalized_count

                # Print progress through training data
                if (t+1)%10000 == 0:
                    print((t+1), 'out of', len(training_corpus), 'done')

            # Update translation probabilities (Maximization step)
            for target_token in expected_count.keys():
                for source_token in expected_count[target_token].keys():
                    self.translation_probabilities[target_token][source_token] = expected_count[target_token][source_token]/expected_total[source_token]

            # Update alignment probabilities (Maximization step)
            for len_target in len_expected_count.keys():
                for len_source in len_expected_count[len_target].keys():
                    for target_idx in len_expected_count[len_target][len_source].keys():
                        for source_idx in len_expected_count[len_target][len_source][target_idx].keys():
                            new_value = len_expected_count[len_target][len_source][target_idx][source_idx]/len_expected_total[len_target][len_source][source_idx]
                            self.alignment_probabilities.write(len_target, len_source, target_idx, source_idx, new_value)

            print('\nIteration', i+1, 'complete')
            print('Log-likelihood before:', total_log_likelihood)
            aer = self.calculate_aer(validation_corpus, validation_gold)
            print('Validation AER after:', aer)
            total_log_likelihoods.append(total_log_likelihood)
            aer_scores.append(aer)
        return (total_log_likelihoods, aer_scores)


    # Find best alignment for a sentence pair
    def align(self, pair):
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
            best_prob = self.translation_probabilities[target_token][None]*self.alignment_probabilities.read(len_target, len_source, target_idx, 0) 

            # Check alignments with all other possible words
            for source_idx, source_token in enumerate(foreign_sentence):
                prob = self.translation_probabilities[target_token][source_token]*self.alignment_probabilities.read(len_target, len_source, target_idx, source_idx)  
                if prob > best_prob:  # prefer newer word in case of tie
                    best_align = source_idx + 1 
                    best_prob = prob
            alignment.add((target_idx+1, best_align))

        return alignment
