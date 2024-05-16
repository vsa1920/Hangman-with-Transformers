Hangman is an interesting language-based game where the player has to guess a word within a given number of wrong tries.
We start with the word completely concealed. Each correct guess reveals all its positions in the word. If you guess all the letters, you win. 
If you guess wrong for a given number of times, say 6, you lose!

This repository contains code to train and evaluate a Hangman playing agent made up of a transformer. Google CANINE-s is a character-level language model.
We initiate the model from random initialization and pre-train on data generated from the game simulations.

CONTENTS:
1) CanineHangmanPlayer.py: This file contains the backbone class for the agent. It can be initiated and trained using the methods defined inside it.
2) hangman_data_generation.py: Given some words, generate the simulation states using Biased Random Sampling.
3) Training_Hangman.ipynb: A sample pre-training script.
4) Self_play_finetune.ipynb: A sample self-play finetuning script.
5) Hangman_analysis.ipynb: A notebook comprising analysis of our trained model.
6) Hangman_transformers.pdf and Hangman_report.pdf: Slides and a report on overview of the entire project.
7) words_corpus.txt: Words borrowed from "https://github.com/dwyl/english-words"
8) Baseline.ipynb: A sample code for a baseline model utilising N_grams

The repository also includes a sample slideshow explaining the model called Hangman_Transformers.pdf
