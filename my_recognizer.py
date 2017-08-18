import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    # TODO implement the recognizer

    for word in range(0, len(test_set.get_all_Xlengths())):
      feature_lists, sequence_length = test_set.get_item_Xlengths(word)
      log_likelihoods = {}

      for current_word, model in models.items():
        try:
          score = model.score(feature_lists, sequence_length)
          log_likelihoods[current_word] = score
        except:
          log_likelihoods[current_word] = float("-inf")
      
      probabilities.append(log_likelihoods)          
      guesses.append(max(log_likelihoods, key = log_likelihoods.get))

    # return probabilities, guesses
    return probabilities, guesses

