
import numpy as np
from transformers import CanineConfig, CanineTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
from hangman_data_generation import *
from transformers import TrainingArguments, Trainer
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, accuracy_score
from transformers import EvalPrediction
import torch
from tqdm import tqdm

def metrics(predictions, labels):
    '''
    Compute the metrics (accuracy, f1-score) for the predictions and labels. Relevant during Pre-training.
    '''
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(labels, axis=1)
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1': f1_micro_average,
            'accuracy': accuracy}
    return metrics

def compute_metrics(p: EvalPrediction):
    '''
    Wrapper function for metrics.
    '''
    preds = p.predictions[0] if isinstance(p.predictions,
            tuple) else p.predictions
    result = metrics(
        predictions=preds,
        labels=p.label_ids)
    return result

class CanineHangmanPlayer:
    '''
    A Hangman Player based on the Google Canine-s transformer
    '''
    def __init__(self, saved_model_path=None):
        self.tokenizer = CanineTokenizer.from_pretrained('google/canine-s')
        self.CANINE_MASK_TOKEN = self.tokenizer.mask_token
        self.CANINE_SEP_TOKEN = self.tokenizer.sep_token
        # Initialize the Canine-s model from a saved model or random initialisation
        self.configuration = CanineConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if saved_model_path is not None:
            self.model = AutoModelForSequenceClassification.from_pretrained(saved_model_path).to(self.device)
        else:
            self.configuration.num_labels = 26
            self.model = AutoModelForSequenceClassification.from_config(self.configuration).to(self.device)
        # Default training arguments for the model will be set here
        # Learning rate and weight decay for pre-training or self-play training
        self.lr = 1e-5
        self.weight_decay = 1e-4
        self.training_mode = "pre-training" # Set to "pre-training" or "self-play"
        # Relevant arguments when training_mode is set to self-play
        self.optimizer = None
        self.criterion = None

    def eval(self):
        """
        Set the hangman player and model to evaluation mode.
        """
        self.training = False
        self.model.eval()

    def train(self):
        """""
        Set the hangman player and model to training mode.
        """
        self.training = True
        self.model.train()

    def set_training(self, lr, weight_decay, batch_size=64, training_mode="pre-training"):
        """
        Set the training arguments for the model.

        Parameters
        ----------
        lr : float
            The learning rate for the model for pre-training and self-play finetuning.
        weight_decay : float
            The weight decay for the model for pre=training and self-play finetunging.
        batch_size: int
            The batch size for training. Set according to your GPU-VRAM
        """
        self.lr = lr
        self.weight_decay = weight_decay
        self.training_mode = training_mode
        if self.training_mode == "pre-training":
            self.training_args = TrainingArguments(
                output_dir="./canine-pretrained-hangman-log", # Hangman checkpoint directory
                overwrite_output_dir=True,
                evaluation_strategy = "epoch",
                save_strategy = "epoch",
                learning_rate=self.lr,
                per_device_train_batch_size=batch_size, # Set batch-size to fit you GPU VRAM
                per_device_eval_batch_size=batch_size,
                logging_strategy='epoch',
                num_train_epochs=1,
                weight_decay=self.weight_decay
            )
        elif self.training_mode == 'self-play':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Training mode {self.training_mode} is not supported.")

    def simulate_hangman_transformers(self, word, max_wrong_guesses=6, verbose=0):
        '''
        Play a game of hangman with the given word.

        Parameters
        ----------
        word : string
            the word to guess.
        max_guesses: int
            the maximum number of guesses allowed.
        verbose : 0, 1 or 2
            the level of printed information for debugging and deploymeny.
        Outputs
        -------
        if self.training is true (Self-play finetuning):
            outputs_model : torch.Tensor
                the states of the game
            outputs : torch.Tensor
                the correct guesses for each state encoded as a probability distribution
            if_correct : boolean
                whether the model correctly guessed the word.
        else:
            if_correct : boolean
                whether the model correctly guessed the word.
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
        if self.training:
            outputs_model = []
            outputs_true = []
        while encoded_word != word and num_guesses < max_wrong_guesses:
            state= ''.join(guesses.keys()) + self.CANINE_SEP_TOKEN + encoded_word.replace('*', self.CANINE_MASK_TOKEN)
            encoded_state = self.tokenizer(state, padding="max_length", truncation=True, return_tensors="pt", max_length=64).to(self.device)
            output = self.model(**encoded_state)
            if self.training:
                output_true = torch.zeros(26).to(self.device)
                for ch in word_idxs:
                    if ch not in guesses:
                        output_true[ord(ch) - ord('a')] = len(word_idxs[ch])
                out_sum = torch.sum(output_true)
                if out_sum.item() > 0:
                    output_true = output_true / out_sum
                else:
                    raise "Invalid output distribution. Division by zero."
                outputs_model.append(output.logits)
                outputs_true.append(output_true)
            output_np = output.logits.cpu().detach().numpy()
            guess_idx = np.argmax(output_np)
            guess = all_letters[guess_idx]
            while guess in guesses:
                output_np[0][guess_idx] = -float('inf')
                guess_idx = np.argmax(output_np)
                guess = all_letters[guess_idx]
            if guess in word_idxs:
                for i in word_idxs[guess]:
                    encoded_word = encoded_word[:i] + guess + encoded_word[i + 1:]
            else:
                num_guesses += 1
            guesses[guess] = True
            if verbose == 2:
                print("Guessing letter:", guess)
                print("Hangman state:", encoded_word)
                print("Number of wrong guesses:", num_guesses)
        if verbose > 0:
            if encoded_word == word:
                print("Correct word:", word)
                print("You win!")
            else:
                print(f"Correct word: {word}, Guessed word: {encoded_word}")
                print("You lose!")
        if self.training:
            return torch.vstack(outputs_model), torch.vstack(outputs_true), encoded_word == word
        else:
            return encoded_word == word

    def test_accuracy(self, words):
        """
        Test the accuracy of the model at playing hangman on a given list of words.

        Parameters
        ----------
        val_words : list[string]
            A list of words to test the model on.
        Outputs
        -------
        accuracy : float
            The accuracy of the model at winning the game of hangman on the given list of words.
        """
        self.eval()
        total_correct = 0
        for w in words:
            if self.simulate_hangman_transformers(w, max_wrong_guesses=6, verbose=0):
                total_correct += 1
        return total_correct / len(words)

    def pretrain_model(self, train_words, val_words, num_epochs, num_val_samples):
        """
        Train the model with a given list of words.
        In this pre-training task we generate hangman states by a Monte-Carlo simulation at each epoch with the given list of words.
        We then train the model with these states to tune its softmax outputs with the correct probability distribution of letters
        to be guesssed in the hangman state

        Parameters
        ----------
        train_words : list[string]
            A list of words to pre-train our model.
        val_words : list[string]
            A list of words to use for validation to display letter guessing accuracy
        num_val_samples : int
            The number of validation samples to use for validation during gameplay during pre-training.
        """
        self.training_mode = "Pre-training"
        trainer = Trainer(
            self.model,
            self.training_args,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics
        )
        validate_states, validate_outputs, _ = encode_game_states(val_words)
        validate_dataset = Dataset.from_dict({'text': validate_states, 'label': validate_outputs})
        validate_dataset = validate_dataset.map(lambda example: self.tokenizer(example["text"], padding="max_length", max_length=64, truncation=True), batched=True)
        val_accuracies = []
        for epoch in range(num_epochs):
            # Generate new simulation states for each epoch
            self.train()
            print(f"Epoch {epoch + 1}/{num_epochs}")
            train_states, outputs, _ = encode_game_states(train_words)
            train_dataset = Dataset.from_dict({'text': train_states, 'label': outputs})
            train_dataset = train_dataset.map(lambda example: self.tokenizer(example["text"], padding="max_length", max_length = 64, truncation=True), batched=True)
            trainer.train_dataset = train_dataset
            trainer.eval_dataset = validate_dataset
            trainer.train()
            val_accuracies.append(self.test_accuracy(val_words[:num_val_samples]))
            print(f"Validation accuracy: {val_accuracies[-1]}")
            self.model.save_pretrained(".canine-pretrained-hangman=checkpoints/canine-pretrained-hangman-checkpoint-" + str(epoch) + "/")

        # Plot the validation accuracy vs epochs during training. We will use this to choose the best model.
        plt.plot(val_accuracies)
        plt.xlabel("Epoch")
        plt.ylabel("Validation Gameplay Accuracy")
        plt.title("Validation Gameplay Accuracy vs Epoch")
        plt.savefig('./val-accuracy.png')

    def self_play_finetune(self, train_data, validation_data, num_steps):
        """
        Fine-tune the model with a given list of words.
        In this fine-tuning task we generate hangman states by the model playing hangaman at each epoch with the given list of words.
        We then train the model with these states so that it learns from its own mistakes.

        Parameters
        ----------
        train_words : list[string]
            A list of words to fine-tune our model.
        val_words : list[string]
            A list of words to use for validation to display gameplay accuracy.
        """
        self.training_mode = "self-play"
        val_accuracies = []
        train_accuracies = []
        train_losses = []
        val_losses = []
        num_step_samples = len(train_data) // num_steps
        for step in range(num_steps):
            total = 0
            total_correct = 0
            total_loss = 0
            train_set = train_data[step * num_step_samples:(step + 1) * num_step_samples]
            for i in tqdm(range(len(train_set))):
                word = train_set[i]
                self.train()
                outputs, outputs_true, correct = self.simulate_hangman_transformers(word, max_wrong_guesses=6, verbose=0)
                loss = self.criterion(outputs, outputs_true)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total += 1
                total_correct += int(correct)
                total_loss += loss.item()
            self.model.eval()
            total_val = 0
            total_correct_val = 0
            total_loss_val = 0
            for i in tqdm(range(len(validation_data))):
                word = validation_data[i]
                outputs, outputs_true, correct = self.simulate_hangman_transformers(word, max_wrong_guesses=6, verbose=0)
                loss = self.criterion(outputs, outputs_true)
                total_val += 1
                total_correct_val += int(correct)
                total_loss_val += loss.item()
            val_acc = total_correct_val / total_val
            train_acc = total_correct / total
            train_loss = total_loss / total
            val_loss = total_loss_val / total_val
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            print("Step: " + str(step + 1) + ", train-loss: " + str(train_loss) + ", val-loss: " + str(val_loss) + ", train-acc: " + str(train_acc) + ", val-acc: "  + str(val_acc))
            self.model.save_pretrained('./canine-pretrained-hangman-checkpoints/canine-pretrained-hangman-checkpoint-' + str(step))

        # Plot the train and validation losses and accuracies
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Learning-curve")
        plt.legend()
        plt.savefig('./self-play-loss.png')

        plt.plot(train_accuracies, label="Train Accuracy")
        plt.plot(val_accuracies, label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Epoch vs Accuracy")
        plt.legend()
        plt.savefig('./self-play-accuracy.png')