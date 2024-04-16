Hangman is an interesting language-based game where the player has to guess a word within a given number of wrong tries.
We start with the word completely concealed. Each correct guess reveals all its positions in the word. If you guess all the letters, you win. 
If you guess wrong for a given number of times, say 6, you lose!

This repository contains code to train and evaluate a Hangman playing agent made up of a transformer. Google CANINE-s is a character-level language model.
We initiate the model from random initialization and pre-train on data generated from the game simulations.

CONTENTS:
1) CanineHangmanPlayer.py: This file contains the backbone class for the agent. It can be initiated and trained using the methods defined inside it.
2) hangman_data_generation.py: Given some words, generate the simulation states using a simple Monte-Carlo simulation.
