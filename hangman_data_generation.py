import numpy as np

from transformers import CanineTokenizer
tokenizer = CanineTokenizer.from_pretrained('google/canine-s')

CANINE_MASK_TOKEN = tokenizer.mask_token
CANINE_SEP_TOKEN = tokenizer.sep_token

def simulate_game(word, correct_rate=0.4, max_wrong_guesses=6):
    '''
    Play a game of hangman with the given word.
    Inputs:
    word: the word to guess
    correct_rate: the probability that we explicitly correct guess a letter
    max_guesses: the maximum number of guesses allowed
    Returns:
    states: the states of the game
    outputs: the correct guesses for each state encoded as a probability distribution
    '''
    word_idxs = {}
    all_letters = [chr(i) for i in range(ord('a'), ord('z') + 1)]
    for i, c in enumerate(word):
        if c not in word_idxs:
            word_idxs[c] = []
        word_idxs[c].append(i)
    guesses = {}
    encoded_word = "*" * len(word)
    num_guesses = 0
    states = []
    outputs = []
    while encoded_word != word and num_guesses < max_wrong_guesses:
        output = np.zeros(26)
        for ch in word_idxs:
            if ch not in guesses:
                output[ord(ch) - ord('a')] = len(word_idxs[ch])
        out_sum = np.sum(output)
        if out_sum > 0:
            output = output / out_sum
        else:
            raise "Invalid output distribution"
        outputs.append(output)
        states.append(''.join(guesses.keys()) + CANINE_SEP_TOKEN + encoded_word.replace('*', CANINE_MASK_TOKEN))
        correct_guess = np.random.random() < correct_rate
        if correct_guess:
            guess = np.random.choice(list(word_idxs.keys()))
            while guess in guesses:
                guess = np.random.choice(list(word_idxs.keys()))
        else:
            guess = np.random.choice(all_letters)
            while guess in guesses:
                guess = np.random.choice(all_letters)
        if guess in word_idxs:
            for i in word_idxs[guess]:
                encoded_word = encoded_word[:i] + guess + encoded_word[i + 1:]
        else:
            num_guesses += 1
        guesses[guess] = True

    return states, np.array(outputs), int(encoded_word == word)

def encode_game_states(dictionary):
    '''
    Encode the game states and outputs for the given dictionary.
    Inputs:
    dictionary: the dictionary of words to use
    Returns:
    all_states: the states of the game
    all_outputs: the correct guesses for each state encoded as a probability distribution
    '''
    all_states = []
    all_outputs = []
    total = 0
    total_solved = 0
    for i, word in enumerate(dictionary):
        states, outputs, solved = simulate_game(word)
        total += 1
        total_solved += solved
        all_states.extend(states)
        all_outputs.extend(outputs)
    return all_states, np.array(all_outputs), total_solved / total