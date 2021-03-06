# UvA Natural Language Processing II practical I

## Usage

```bash
# local
pip install -r requirements.txt
cd src/
python main.m
# docker
docker build -t ibm .
docker run --rm -ti -v $PWD:/app ibm python main.py --model ibm2
docker run -ti -u (id -u):(id -g) -p 6006:6006 -v $PWD:/logs tensorflow/tensorflow tensorboard --logdir /logs
http://127.0.0.1:6006/
```

## Roadmap

### IBM model 1
- [ ] Implement IBM model 1
    - 1.
        - [x] a) Implement EM training (Brown et al., 1993) for IBM model 1;
        - [x] b) All of the tasks below should be performed for both models.
    - [x] 2. Plot the evolution of training log likelihood as a function of the iteration.
    - [x] 3. Plot the evolution of alignment error rate (AER) on validation data as a function of the iteration;
    - [ ] 4. Experiment with two criteria for model selection (i.e. deciding on number of training iterations):
        - [ ] 1) convergence in terms of training log likelihood;
        - [ ] 2) best AER on validation data;
    - [x] 5. For the selected models, obtain Viterbi alignments for every sentence pair in a test corpus and compute AER using a gold-standard provided by the assistant;
        - [x] IBM1: AER 0.3378 over 10 iterations
        - [x] IBM2: AER .2428 over 10 iterations (lexical component), then 5 additional iterations (lexical and jump components)

### IBM model 2
- [x] Implement IBM model 2 using a jump distribution as in Vogel et al. (1996);
    - [x] 1. Extend your previous model by implementing a full IBM model 2 (Brown et al.,1993), however using the cheaper parameterisation in terms of jumps;
    - [x] 2. IBM 2 is non-convex, thus you will see that optimising the log-likelihood function is not as trivial as in the case of IBM model 1, particularly, convergence will depend on how you initialise the model parameters, you will try
        - [x] uniform initialisation
        - [x] random initialisation (try 3 different starting points)
        - [x] initialise the lexical parameters using the output of a complete run of model 1
    - [x] 3. Plot training log-likelihood as a function of the iteration for all these cases
    - [x] 4. Plot validation AER as a function of the iteration for all these cases
    - [ ] 5. Select two models:
        - [ ] 1) one in terms of training log likelihood,
        - [ ] 2) another in terms of validation AER;
    - [ ] 6. Compare the selected models to IBM model 1 in terms of AER in the test set.
- [ ] Experiment with maximum likelihood estimation;

### Report
- [ ] Write a technical report where you present the models and an empirical comparison. Your report should also present learning curves where applicable along with a discussion explaining aspects such as non-convexity, stability and convergence.
    - [x] We expect short reports (5 pages plus references) written in English.
    - [x] The typical submission is organised as follows:
        - [x] abstract: conveys scope and contributions;
        - [x] introduction: present the problem and relevant background;
        - [x] model: technical description of models;
        - [x] experiments: details about the data, experimental setup and findings;
        - [x] conclusion: a critical take on contributions and limitations.
- [ ] metrics we want to report on:
    - [ ] AER score and log-likelihood averaged over sentence pairs
    - [ ] test AER score when comparing best models

### Submission
You should submit a tgz file containing a folder (folder namelastname1.lastname2) with the following content:
- [ ] Test predictions (in naacl format) using your best run for each of the following models
    - [ ] IBM1 MLE (filename:ibm1.mle.naacl)
    - [ ] IBM2 MLE (filename:ibm2.mle.naacl)
    - [x] output alignments to naacl file
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
    - [x] use of latex,
    - [x] structure of report,
    - [ ] use of tables/figures/plots,
    - [x] command of English.
- [ ] 5. Extra (max 1 point).
    - [ ] [neuralibm notebook](https://uva-slpl.github.io/nlp2/resources/project_ibm/neuralibm.tar.gz) which we can do as extra
