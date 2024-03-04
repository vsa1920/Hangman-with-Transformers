import numpy as np

with open('words_corpus.txt', 'r') as f:
    dictionary = f.read().splitlines()

shuffled_dictionary = np.random.permutation(dictionary)

# Save words in training data
with open('train_words.txt', 'w') as f:
    for word in shuffled_dictionary[:-15000]:
        f.write(word + '\n')

# Save words in validation data
with open('validation_words.txt', 'w') as f:
    for word in shuffled_dictionary[-15000:-5000]:
        f.write(word + '\n')

# Save words in test data
with open('test_words.txt', 'w') as f:
    for word in shuffled_dictionary[-5000:]:
        f.write(word + '\n')