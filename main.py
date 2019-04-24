from collections import defaultdict
import sys
import math
import itertools
from typing import Tuple, List, Set, Dict
import matplotlib.pyplot as plt
import aer

class IBM1():
	def __init__(self, vocab_en, translation_probabilities = None):
		self.vocab_en = vocab_en		
		if translation_probabilities is None:
			default_probability = 1/len(vocab_en)
			self.translation_probabilities = defaultdict(lambda: defaultdict(lambda: default_probability))
		else:
			self.translation_probabilities = translation_probabilities
	
	# Calculate log-likelihood of entire corpus
	def total_log_likelihood(self, corpus):
		print('\nCalculating log-likelihood')
		total_log_likelihood = 0

		for t in range(len(corpus)):
			# Expand sentence pair
			english_sentence, foreign_sentence = corpus[t]

			# Add null token
			foreign_sentence = [None] + foreign_sentence

			# Calculate log_likelihood of pair
			log_likelihood, _ = self.log_likelihood(english_sentence, foreign_sentence)
			total_log_likelihood += log_likelihood

			# Print progress through corpus
			if (t+1)%10000 == 0:
				print((t+1), 'out of', len(corpus), 'done')
		return total_log_likelihood
	
	# Calculate e-likelihoods and log-likelihood of sentence pair
	def log_likelihood(self, english_sentence, foreign_sentence):	
		log_likelihood = -math.log(len(foreign_sentence)**len(english_sentence))
		e_likelihoods = defaultdict(lambda: 0)
		for e in english_sentence:				
			for f in foreign_sentence:
				e_likelihoods[e] += self.translation_probabilities[e][f]
			log_likelihood += math.log(e_likelihoods[e])
		return (log_likelihood, e_likelihoods)
	
	# Train model
	def train(self, training_corpus, iterations, validation_corpus, validation_gold):
		total_log_likelihoods = []
		aer_scores = []
		for i in range(iterations):
			print('\nStarting iteration', i+1)
			expected_count = defaultdict(lambda: defaultdict(lambda: 0)) # Expected number of times e is connected to f 
			expected_total = defaultdict(lambda: 0) # Expected total connections for f
			total_log_likelihood = 0
			
			# Calculate expected counts (Expectation step) and log-likelihood
			for t in range(len(training_corpus)):
				# Expand sentence pair
				english_sentence, foreign_sentence = training_corpus[t]

				# Add null token
				foreign_sentence = [None] + foreign_sentence		

				# Calculate e-likelihoods and log-likelihood of sentence pair
				log_likelihood, e_likelihoods = self.log_likelihood(english_sentence, foreign_sentence)
				total_log_likelihood += log_likelihood

				# Collect counts
				for e in english_sentence:
					for f in foreign_sentence:
						normalized_count = self.translation_probabilities[e][f]/e_likelihoods[e]
						expected_count[e][f] += normalized_count
						expected_total[f] += normalized_count

				# Print progress through training data
				if (t+1)%10000 == 0:
					print((t+1), 'out of', len(training_corpus), 'done')

			# Update translation probabilities (Maximization step)
			for e in expected_count.keys():
				for f in expected_count[e].keys():
					self.translation_probabilities[e][f] = expected_count[e][f]/expected_total[f]

			print('\nIteration', i+1, 'complete')
			print('Log-likelihood before:', total_log_likelihood)
			aer = self.calculate_aer(validation_corpus, validation_gold)
			print('Validation AER after:', aer)
			total_log_likelihoods.append(total_log_likelihood)
			aer_scores.append(aer)
		return (total_log_likelihoods, aer_scores)

	# Print most likely translation for each foreign word
	def print_dictionary(self):
		for e in self.vocab_en:
			probs = self.translation_probabilities[e]
			print(e, max(zip(probs.values(), probs.keys())))

	# Find best alignment for a sentence pair
	def align(self, pair):
		# Expand sentence pair
		english_sentence, foreign_sentence = pair

		# Initialize alignment
		alignment = set()

		# Add best link for every english word
		for e_idx, e in enumerate(english_sentence):
			# Default alignment with null token
			best_align = None
			best_prob = self.translation_probabilities[e][None]     

			# Check alignments with all other possible words
			for f_idx, f in enumerate(foreign_sentence):
				prob = self.translation_probabilities[e][f] 
				if prob > best_prob:  # prefer newer word in case of tie
					best_align = f_idx + 1 
					best_prob = prob
			alignment.add((e_idx+1, best_align))

		return alignment

	# Calculate AER on validation corpus using gold standard
	def calculate_aer(self, validation_corpus, validation_gold):
		# Compute predictions
		predictions = []
		for pair in validation_corpus:
			predictions.append(self.align(pair))

		# Compute AER
		metric = aer.AERSufficientStatistics()
		for gold, pred in zip(validation_gold, predictions):
			metric.update(sure=gold[0], probable=gold[1], predicted=pred)
		return metric.aer()

class AlignmentProbabilities():
	def __init__(self):
		self.alignment_probabilities = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: dict())))

	def read(self, len_e, len_f, e_idx, f_idx):
		if f_idx not in self.alignment_probabilities[len_e][len_f][e_idx]:
			self.alignment_probabilities[len_e][len_f][e_idx][f_idx] = 1/10
		return self.alignment_probabilities[len_e][len_f][e_idx][f_idx]

	def write(self, len_e, len_f, e_idx, f_idx, value):
		self.alignment_probabilities[len_e][len_f][e_idx][f_idx] = value

class IBM2(IBM1):
	def __init__(self, vocab_en, translation_probabilities = None):
		super(IBM2, self).__init__(vocab_en, translation_probabilities)
		self.alignment_probabilities = AlignmentProbabilities()

	# Calculate e-likelihoods and log-likelihood of sentence pair
	def log_likelihood(self, english_sentence, foreign_sentence):	
		len_e = len(english_sentence)
		len_f = len(foreign_sentence)
		log_likelihood = 0
		e_likelihoods = defaultdict(lambda: 0)
		for e_idx, e in enumerate(english_sentence):				
			for f_idx, f in enumerate(foreign_sentence):
				translation_probability = self.translation_probabilities[e][f]
				alignment_probabilility = self.alignment_probabilities.read(len_e, len_f, e_idx, f_idx)
				e_likelihoods[e] += translation_probability*alignment_probabilility
			log_likelihood += math.log(e_likelihoods[e])
		return (log_likelihood, e_likelihoods)

	# Train model
	def train(self, training_corpus, iterations, validation_corpus, validation_gold):
		total_log_likelihoods = []
		aer_scores = []
		for i in range(iterations):
			print('\nStarting iteration', i+1)
			expected_count = defaultdict(lambda: defaultdict(lambda: 0)) # Expected number of times e is connected to f 
			expected_total = defaultdict(lambda: 0) # Expected total connections for f
			len_expected_count = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))) # Expected number of times e is connected to f in pairs of specific length
			len_expected_total = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0))) # Expected number of total connections for f in pairs of specific length
			total_log_likelihood = 0
			
			# Calculate expected counts (Expectation step) and log-likelihood
			for t in range(len(training_corpus)):
				# Expand sentence pair
				english_sentence, foreign_sentence = training_corpus[t]

				# Add null token
				foreign_sentence = [None] + foreign_sentence

				# Calculate sentence lenghts 
				len_e = len(english_sentence)
				len_f = len(foreign_sentence)					

				# Calculate e-likelihoods and log-likelihood of sentence pair
				log_likelihood, e_likelihoods = self.log_likelihood(english_sentence, foreign_sentence)
				total_log_likelihood += log_likelihood

				# Collect counts
				for e_idx, e in enumerate(english_sentence):
					for f_idx, f in enumerate(foreign_sentence):
						translation_probability = self.translation_probabilities[e][f]
						alignment_probabilility = self.alignment_probabilities.read(len_e, len_f, e_idx, f_idx)
						normalized_count = translation_probability*alignment_probabilility/e_likelihoods[e]
						expected_count[e][f] += normalized_count
						expected_total[f] += normalized_count
						len_expected_count[len_e][len_f][e_idx][f_idx] += normalized_count
						len_expected_total[len_e][len_f][f_idx] += normalized_count

				# Print progress through training data
				if (t+1)%10000 == 0:
					print((t+1), 'out of', len(training_corpus), 'done')

			# Update translation probabilities (Maximization step)
			for e in expected_count.keys():
				for f in expected_count[e].keys():
					self.translation_probabilities[e][f] = expected_count[e][f]/expected_total[f]

			# Update alignment probabilities (Maximization step)
			for len_e in len_expected_count.keys():
				for len_f in len_expected_count[len_e].keys():
					for e_idx in len_expected_count[len_e][len_f].keys():
						for f_idx in len_expected_count[len_e][len_f][e_idx].keys():
							new_value = len_expected_count[len_e][len_f][e_idx][f_idx]/len_expected_total[len_e][len_f][f_idx]
							self.alignment_probabilities.write(len_e, len_f, e_idx, f_idx, new_value)
				
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
		english_sentence, foreign_sentence = pair
		len_e = len(english_sentence)
		len_f = len(foreign_sentence)

		# Initialize alignment
		alignment = set()

		# Add best link for every english word
		for e_idx, e in enumerate(english_sentence):
			# Default alignment with null token
			best_align = None
			best_prob = self.translation_probabilities[e][None]*self.alignment_probabilities.read(len_e, len_f, e_idx, 0) 

			# Check alignments with all other possible words
			for f_idx, f in enumerate(foreign_sentence):
				prob = self.translation_probabilities[e][f]*self.alignment_probabilities.read(len_e, len_f, e_idx, f_idx)  
				if prob > best_prob:  # prefer newer word in case of tie
					best_align = f_idx + 1 
					best_prob = prob
			alignment.add((e_idx+1, best_align))

		return alignment

class IBM2Jump(IBM1):
	def __init__(self, english_vocab, translation_probabilities = None):
		super(IBM2Jump, self).__init__(english_vocab, translation_probabilities)
		self.jump_counts = defaultdict(lambda: 1)

	# Calculate e-likelihoods and log-likelihood of sentence pair
	def log_likelihood(self, english_sentence, foreign_sentence):	
		len_e = len(english_sentence)
		len_f = len(foreign_sentence)
		log_likelihood = 0
		e_likelihoods = defaultdict(lambda: 0)

		for e_idx, e in enumerate(english_sentence):	
			normalizer = 0 # Normalize to have sensible log_likelihood
			for f_idx, f in enumerate(foreign_sentence):
				normalizer += self.jump_counts[f_idx - int(e_idx*len_f/len_e)]
			for f_idx, f in enumerate(foreign_sentence):
				translation_probability = self.translation_probabilities[e][f]
				alignment_probabilility = self.jump_counts[f_idx - int(e_idx*len_f/len_e)]/normalizer
				e_likelihoods[e] += translation_probability*alignment_probabilility
			log_likelihood += math.log(e_likelihoods[e])
		return (log_likelihood, e_likelihoods)

	# Train model
	def train(self, training_corpus, iterations, validation_corpus, validation_gold):
		for i in range(iterations):
			print("\nStarting iteration", i+1)
			expected_count = defaultdict(lambda: defaultdict(lambda: 0)) # Expected number of times e is connected to f 
			expected_total = defaultdict(lambda: 0) # Expected total connections for f
			expected_jump_count = defaultdict(lambda: 0) # Expected connections with a certain jump length
			total_log_likelihood = 0
			
			# Calculate expected counts (Expectation step) and log-likelihood
			for t in range(len(training_corpus)):
				# Expand sentence pair
				english_sentence, foreign_sentence = training_corpus[t]

				# Add null token
				foreign_sentence = [None] + foreign_sentence

				len_e = len(english_sentence)
				len_f = len(foreign_sentence)

				# Calculate e-likelihoods and log-likelihood of sentence pair
				log_likelihood, e_likelihoods = self.log_likelihood(english_sentence, foreign_sentence)
				total_log_likelihood += log_likelihood

				# Collect counts
				for e_idx, e in enumerate(english_sentence):
					for f_idx, f in enumerate(foreign_sentence):
						normalizer += self.jump_counts[f_idx - int(e_idx*len_f/len_e)]
					for f_idx, f in enumerate(foreign_sentence):
						translation_probability = self.translation_probabilities[e][f]
						alignment_probabilility = self.jump_counts[f_idx - int(e_idx*len_f/len_e)]/normalizer
						normalized_count = translation_probability*alignment_probabilility/e_likelihoods[e]
						expected_count[e][f] += normalized_count
						expected_total[f] += normalized_count
						expected_jump_count[f_idx - int(e_idx*len_f/len_e)] += normalized_count

				# Print progress through training data
				if (t+1)%10000 == 0:
					print((t+1), "out of", len(training_corpus), "done")

			# Update translation probabilities (Maximization step)
			for e in expected_count.keys():
				for f in expected_count[e].keys():
					self.translation_probabilities[e][f] = expected_count[e][f]/expected_total[f]

			# Update jump counts (Maximization step)
			for jump in expected_jump_count.keys():
				self.jump_counts[jump] = expected_jump_count[jump]
			
			print("\nIteration", i+1, "complete")
			print("Log-likelihood before:", total_log_likelihood)
			print("Validation AER after:", self.calculate_aer(validation_corpus, validation_gold))

	# Find best alignment for a sentence pair
	def align(self, pair):
		# Expand sentence pair
		english_sentence, foreign_sentence = pair
		len_e = len(english_sentence)
		len_f = len(foreign_sentence)

		# Initialize alignment
		alignment = set()

		# Add best link for every english word
		for e_idx, e in enumerate(english_sentence):
			# Default alignment with null token
			best_align = None
			best_prob = self.translation_probabilities[e][None]*self.jump_counts[0 - int(e_idx*len_f/len_e)] 

			# Check alignments with all other possible words
			for f_idx, f in enumerate(foreign_sentence):
				prob = self.translation_probabilities[e][f]*self.jump_counts[f_idx - int(e_idx*len_f/len_e)] 
				if prob > best_prob:  # prefer newer word in case of tie
					best_align = f_idx + 1 
					best_prob = prob
			alignment.add((e_idx+1, best_align))

		return alignment

def read_tokens(path: str, n:int=None) -> List[List[str]]:
	sentences = open(path, 'r', encoding='utf8').readlines()
	sentences_ = itertools.islice(sentences, n)
	# we split on spaces as the hansards dataset uses explicit spacing between tokens
	return [sentence.split(' ')[:-1] for sentence in sentences_]

def sentence_vocab(tokenized: List[List[str]]) -> Set[str]:
	return set([token for tokens in tokenized for token in tokens])

def main():
	# Read in training data
	tokenized_en = read_tokens('data/training/hansards.36.2.e')
	tokenized_fr = read_tokens('data/training/hansards.36.2.f')
	training_corpus = list(zip(tokenized_en, tokenized_fr))
	vocab_en = sentence_vocab(tokenized_en)
	vocab_fr = sentence_vocab(tokenized_fr)
	print(f'vocabulary size english: {len(vocab_en)}')
	print(f'vocabulary size french:  {len(vocab_fr)}')

	# Read in validation data
	validation_corpus = list(zip(
		read_tokens('data/validation/dev.e'),
		read_tokens('data/validation/dev.f'),
	))
	validation_gold = aer.read_naacl_alignments('data/validation/dev.wa.nonullalign')

	# Read in test data
	test_corpus = list(zip(
		read_tokens('data/testing/test/test.e'),
		read_tokens('data/testing/test/test.f'),
	))
	test_gold = aer.read_naacl_alignments('data/testing/answers/test.wa.nonullalign')

	# Create ibm1 model
	ibm1_model = IBM2Jump(vocab_en)

	# Print initial validation AER	
	initial_aer = ibm1_model.calculate_aer(validation_corpus, validation_gold)
	print('Initial validation AER:', initial_aer)
	
	# Train the model
	iterations = 15
	log_likelihoods, aer_scores = ibm1_model.train(training_corpus, iterations, test_corpus, test_gold)
	#aer_scores = [initial_aer] + aer_scores

	# Print log-likelihood after training
	final_log_likelihood = ibm1_model.total_log_likelihood(training_corpus)
	print('\nFinal log-likelihood:', final_log_likelihood)
	log_likelihoods.append(final_log_likelihood)
	log_likelihoods.pop(0)

	# Plot log-likelihood and aer curves
	fig = plt.figure()
	plt.plot(list(range(iterations+1))[1:], log_likelihoods)
	plt.xlabel('Iterations')
	plt.ylabel('Log-likelihood')
	plt.savefig('IBM1_log_likelihoods.png')
	plt.close(fig)

	fig = plt.figure()
	plt.plot(list(range(iterations+1))[1:], aer_scores)
	plt.xlabel('Iterations')
	plt.ylabel('AER Score')
	plt.savefig('IBM1_aer_scores.png')
	plt.close(fig)

	f = open('run_log.txt','w+')
	for i in range(iterations):
		f.write('Iteration: ' + str(i+1) + ' Log-likelihood: ' + str(log_likelihoods[i]) + ' AER score: ' + str(aer_scores[i]) + '\n')

	# Create ibm2 model and train it
	#ibm2_model = IBM2(vocab_en, ibm1_model.translation_probabilities)
	#ibm2_model.train(training_corpus, 5, validation_corpus, validation_gold)

	# Print log-likelihood after training
	#print('\nFinal log-likelihood:', ibm2_model.total_log_likelihood(training_corpus))

	# Print dictionary
	#ibm1_model.print_dictionary()
	
if __name__ == '__main__':
	main()
