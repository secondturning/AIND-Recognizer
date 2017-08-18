import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ Select best model for self.this_word based on BIC score
        for n between self.min_n_components and self.max_n_components
        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        scores = []
        for states in range(self.min_n_components, self.max_n_components + 1):
            try:
                hmm_model = self.base_model(states)
                log_likelihood = hmm_model.score(self.X, self.lengths)
                data_points = sum(self.lengths)
                free_params = (states ** 2) + (2*states*data_points) - 1
                score = (-2 * log_likelihood) + (free_params * np.log(data_points))
                scores.append(tuple([hmm_model, score]))
            except Exception as e:
                # print(e)
                pass
        return min(scores, key = lambda score: score[1])[0] if scores else None


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # print ("using selector DIC")

        other_words = []
        for word in self.words:
            if word != self.this_word:
                other_words.append(self.hwords[word])
        # print(other_words)

        models_with_logl = []
        scores = []

        try:
            for states in range(self.min_n_components, self.max_n_components + 1):
                hmm_model = self.base_model(states)
                log_likelihood = hmm_model.score(self.X, self.lengths)
                models_with_logl.append((hmm_model, log_likelihood))

        except Exception as e:
            # print(e)
            pass
        for index, model in enumerate(models_with_logl):
            hmm_model, log_likelihood  = model
            score = log_likelihood - np.mean([hmm_model.score(word[0], word[1]) for word in other_words])
            # self.calc_log_likelihood_other_words(model, other_words)
            scores.append(tuple([model[0], score]))

        return max(scores, key = lambda score: score[1])[0] if scores else None            


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''
            
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        k_fold = KFold(n_splits = 3, shuffle = False, random_state = None)
        log_likelihoods = []
        scores = []

        for states in range(self.min_n_components, self.max_n_components + 1):
            hmm_model = self.base_model(states)
            try:
                if len(self.sequences) > 2:
                    for train_index, test_index in k_fold.split(self.sequences):
                        self.X, self.lengths = combine_sequences(train_index, self.sequences)
                        X_test, lengths_test = combine_sequences(test_index, self.sequences)
                        log_likelihood = hmm_model.score(X_test, lengths_test)
                else:
                    log_likelihood = hmm_model.score(self.X, self.lengths)

                log_likelihoods.append(log_likelihood)

                avg_score = np.mean(log_likelihoods)
                scores.append(tuple([hmm_model, avg_score]))
            except Exception as e:
                pass

        return max(scores, key = lambda score: score[1])[0] if scores else None
