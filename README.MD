# UvA Natural Language Processing II practical I

## Usage

```bash
pip install -r requirements.txt
python main.m
```

## Roadmap

### IBM model 1
- [ ] Implement IBM model 1
    - 1.
        - [ ] a) Implement EM training (Brown et al., 1993) for IBM model 1;
        - [ ] b) All of the tasks below should be performed for both models.
    - [ ] 2. Plot the evolution of training log likelihood as a function of the iteration.
    - [ ] 3. Plot the evolution of alignment error rate (AER) on validation data as a function of the iteration;
    - [ ] 4. Experiment with two criteria for model selection (i.e. deciding on number of training iterations):
        - [ ] 1) convergence in terms of training log likelihood;
        - [ ] 2) best AER on validation data;
    - [ ] 5. For the selected models, obtain Viterbi alignments for every sentence pair in a test corpus and compute AER using a gold-standard provided by the assistant;
        - [ ] IBM1: AER 0.3378 over 10 iterations
        - [ ] IBM2: AER .2428 over 10 iterations (lexical component), then 5 additional iterations (lexical and jump components)

### IBM model 2
- [ ] Implement IBM model 2 using a jump distribution as in Vogel et al. (1996);
    - [ ] 1. Extend your previous model by implementing a full IBM model 2 (Brown et al.,1993), however using the cheaper parameterisation in terms of jumps;
    - [ ] 2. IBM 2 is non-convex, thus you will see that optimising the log-likelihood function isnot as trivial as in the case of IBM model 1, particularly, convergence will depend onhow you initialise the model parameters, you will try
        - [ ] uniform initialisation
        - [ ] random initialisation (try 3 different starting points)
        - [ ] initialise the lexical parameters using the output of a complete run of model 1
    - [ ] 3. Plot training log-likelihood as a function of the iteration for all these cases
    - [ ] 4. Plot validation AER as a function of the iteration for all these cases
    - [ ] 5. Select two models:
        - [ ] 1) one in terms oftraining log likelihood,
        - [ ] 2) another in terms of validation AER;
    - [ ] 6. Compare the selected models to IBM model 1 in terms of AER in the test set.
- [ ] Experiment with maximum likelihood estimation;

### Report
- [ ] Write a technical report where you present the models and an empirical comparison.   Your report should also present learning curves where applicable along with a discussion explaining aspects such as non-convexity, stability and convergence.
    - [ ] We expect short reports (5 pages plus references) written in English.
    - [ ] The typical submission is organised as follows:
        - [ ] abstract: conveys scope and contributions;
        - [ ] introduction: present the problem and relevant background;
        - [ ] model: technical description of models;
        - [ ] experiments: details about the data, experimental setup and findings;
        - [ ] conclusion: a critical take on contributions and limitations.

### Submission
You should submit a tgz file containing a folder (folder namelastname1.lastname2) withthe following content:
- [ ] Test predictions (in naacl format) using your best run for each of the following models
    - [ ] IBM1 MLE (filename:ibm1.mle.naacl)
    - [ ] IBM2 MLE (filename:ibm2.mle.naacl)
    - [ ] output alignments to naacl file
- [ ] Report as a single pdf file (filename:report.pdf)
- [ ] Your report may contain a link to an open-source repository (such as github), but please do not attach code or additional data to your tgz submission.
- [ ] You can complete your project submission on Canvas.

### Assessment
Your report will be assessed according to the following evaluation criteria:
- [ ] 1. Scope (max 2 points):
    - [ ] Is the problem well presented?
    - [ ] Do students understand the challenges/contributions?
- [ ] 2. Theoretical description (max 3 points):
    - [ ] Are the models presented clearly and correctly?
- [ ] 3. Empirical evaluation (max 3 points):
    - [ ] Is the experimental setup sound/convincing?
    - [ ] Are experimental findings presented in an organised and effective manner?
- [ ] 4. Writing style (max 2 points):
    - [ ] use of latex,
    - [ ] structure of report,
    - [ ] use of tables/figures/plots,
    - [ ] command of English.
- [ ] 5. Extra (max 1 point).
    - [ ] neuralibm notebook which we can do as extra
