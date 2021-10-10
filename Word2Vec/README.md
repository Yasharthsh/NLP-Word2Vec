# Fall 2021 CSE 538 - Natural Language Processing Assignment 1

This file is best viewed in a Markdown reader (eg. https://jbt.github.io/markdown-editor/)

## Overview

In this assignment, you will be asked to:

  - generate batch for skip-gram model (data.py)
  - implement two loss functions to train word embeddings (model.py)
  - tune the parameters for word embeddings
  - apply best learned word embeddings to word analogy task (word_analogy.py)
  - calculate bias score on your best models (eval_bias.py)
  - create a new task on which you would run WEAT test (custom_weat.json)

## Setup

This assignment is implemented in Python 3.6 and PyTorch 1.9.0. Follow these steps to setup your environment:

1. [Download and install Conda](http://https://conda.io/projects/conda/en/latest/user-guide/install/index.html "Download and install Conda")

2. Create a Conda environment with Python 3.6: `conda create -n nlp-hw1 Python=3.6`

3. Activate the Conda environment. You will need to activate the Conda environment in each terminal in which you want to use this code: `conda activate nlp-hw1`

4. Install the requirements: `pip install -r requirements.txt`. You do not need anything else, so please do not install other packages.

5. Download related resources: `./download_resources.sh` and save them to the data folder: `data/`.

**NOTE:** We will be using this environment to check your code, so please don't work in your default or any other Python environment.

## Boilerplate Code Organisation

When you download the assignment from Blackboard, it will unzip into the following directory structure:

```
538_HW1
├── .gitignore
└── data/
  ├── custom_weat.json
  ├── weat.json
  ├── word_analogy_dev_mturk_answers.txt
  ├── word_analogy_dev_sample_predictions.txt
  ├── word_analogy_dev.txt
  └── word_analogy_test.txt
├── data.py
├── download_resources.sh
├── evaluate_word_analogy.pl
├── main.py
├── model.py
├── README.md
├── requirements.txt
├── train.py
└── word_analogy.py
```

Please only make your code edits in the TODO(students) blocks in the codebase. The exact files you need to work are listed later.

## Generating data

To train word vectors, you need to generate training instances from the given data.
You will implement a method that will generate training instances in batches.

For skip-gram model, you will slide a window and sample training instances from the data inside the window.

[Example]
Suppose that we have a text: "The quick brown fox jumps over the lazy dog."
And batch_size = 8, window_size = 3

"[The quick brown] fox jumps over the lazy dog"

Context word would be 'quick' and predicting words are 'The' and 'brown'.
This will generate training examples:
      context(x), predicted_word(y)
        (quick    ,       The)
        (quick    ,     brown)

And then move the sliding window.
"The [quick brown fox] jumps over the lazy dog"
In the same way, we have two more examples:
    (brown, quick)
    (brown, fox)

Moving the window again:
"The quick [brown fox jumps] over the lazy dog"
We get,
    (fox, brown)
    (fox, jumps)

Finally we get two more instances from the moved window,
"The quick brown [fox jumps over] the lazy dog"
    (jumps, fox)
    (jumps, over)

Since now we have 8 training instances, which is the batch size,
stop generating this batch and return batch data.

data_index is the index of a word. You can access a word using data[data_index].
batch_size is the number of instances in one batch.
num_skips is the number of samples you want to draw in a window (in example, it was 2).
skip_windows decides how many words to consider left and right from a context word(so, skip_windows*2+1 = window_size).
batch will contains word ids for context words. Dimension is [batch_size].
labels will contains word ids for predicting words. Dimension is [batch_size, 1].

You need to fill in `data.py`.

## Loss Functions

You will implement two loss functions to train word vectors: (1) negative log likelihood - recall that we discussed the log likelihood function in class, and (2) negative sampling - an alternate function that uses a set of k negative words.

You need to fill in `model.py`.

### Negative log likelihood

We discussed the log likelihood function in class. This is the negative of the same.
These are called “loss” functions since they measure how bad the current model is from the expected behavior.
Refer to the class notes on this topic.
To understand it better, you may also refer to Section 4.3 [here](http://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes01-wordvecs1.pdf).

Training a word2vec model with this loss and the default settings took ~50 mins on Google Colab with GPU accelarator.
It will take ~10 hrs on a Macbook Pro 2018 CPU.

### Negative Sampling

The negative sampling formulates a slightly different classification task and a corresponding loss.
[This paper](https://papers.nips.cc/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf) describes the method in detail.

The idea here is to build a classifier that can give high probabilities to words that are the correct target words and low probabilities to words that are incorrect target words. 
As with negative log likelihood loss, here we define the classifier using a function that uses the word vectors of the context and target as free parameters.
The key difference however is that instead of using the entire vocabulary, here we sample a set of k negative words for each instance, and create an augmented instance which is a collection of the true target word and k negative words. 
Now the vectors are trained to maximize the probability of this augmented instance.
To understand it better, you may also refer to Section 4.4 [here](http://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes01-wordvecs1.pdf).

Training a word2vec model with this loss and the default settings took ~2h30 mins on Google Colab with GPU accelarator.

For word2vec

- main.py: 
  This file is the main script for training word2vec model.
  Your trained model will be saved as word2vec.model.
  Usage: `python main.py`

- data.py:
  This file contains various operations related to reading data and other files.
  It encapsulates functions relevant to model training into a Dataset class.
  You should fill out the part that generates batch from training data. 

- train.py:
  This file contains a Trainer class to train a model.
  It is a similar formulation as seen in the logistic regression demo in class.

- model.py:
  This file contains code for the word2vec model as well as the two loss function implementations, that need to be filled in.
  - negative_log_likelihood_loss
  - negative_sampling

## Hyperparameter Experiments

You will conduct some hyper-parameter tuning experiments to figure out the best way to learn the word vectors. 
You will learn different word vectors and try them on the analogy task for the dev split. 
You will pick the configuration that works best for each loss function as your best word vector under that method.
Your goal is to understand how the different hyperparameters affect the embeddings. 
You can do just a few or a gazillion experiments here. 
I’d recommend trying at least five configurations for each method. 
There are really very few rule of thumbs at this point on how to do these experiments. 
You can consult any resources to figure out a good way to go through the hyperparameter settings.

## Analogies using word vectors

You will use the word vectors you learned from both approaches in the following word analogy task.

Each question/task is in the following form. 
```
Consider the following word pairs that share the same relation, R:

    pilgrim:shrine, hunter:quarry, assassin:victim, climber:peak

Among these word pairs,

(1) pig:mud
(2) politician:votes
(3) dog:bone
(4) bird:worm

Q1. Which word pairs has the MOST illustrative(similar) example of the relation R?
Q2. Which word pairs has the LEAST illustrative(similar) example of the relation R?
```

For each question, there are examples pairs of a certain relation. 
Your task is to find the most/least illustrative word pair of the relation.
One simple method to answer those questions will be measuring the similarities of difference vectors.

Recall that vectors are representing some direction in space. 
If (a, b) and (c, d) pairs are analogous pairs then the transformation from a to b (i.e., some x vector when added to a gives b: a + x = b) 
should be highly similar to the transformation from c to d (i.e., some y vector when added to c gives d: b + y = d). 
In other words, the difference vector (b-a) should be similar to difference vector (d-c). 

This difference vector can be thought to represent the relation between the two words. 

You need to fill in `word_analogy.py`.

Due to the noisy annotation data, the expected accuracy is not high. 
The NLL default overall accuracy is 33.5% and negative sampling default overall accuracy is 33.6%.
Improving this score 1~3% would be your goal.

For analogy task

  - `word_analogy.py`:
    You will write a code in this file for evaluating relation between pairs of words -- called the [MaxDiff question](https://en.wikipedia.org/wiki/MaxDiff).
    You will generate a file with your predictions following the format of `data/word_analogy_sample_predictions.txt`.

    Example usage: `python word_analogy.py --model_path best_nll_model --loss_model nll`

  - `evaluate_word_analogy.pl`:
    a perl script to evaluate YOUR PREDICTIONS on development data
    Usage: `./evaluate_word_analogy.pl data/word_analogy_mturk_answers.txt <your prediction file> <output file of result>`
    Example usage: `./evaluate_word_analogy.pl data/word_analogy_dev_mturk_answers.txt best_nll_model/dev_preds_nll.txt best_nll_model/dev_score_nll.txt`

    You do not need to submit prediction or score files related to the dev set.

  - `data/word_analogy_dev.txt`:
    Data for development. 
    Each line of this file is divided into "examples" and "choices" by "||".
        [examples]||[choices]
    "Examples" and "choices" are delimited by a comma.
      ex) "tailor:suit","oracle:prophesy","baker:flour"

  - `data/word_analogy_dev_sample_predictions.txt`:
    A sample prediction file. Pay attention to the format of this file. 
    Your prediction file should follow this to use "score_maxdiff.pl" script.
    Each row is in this format:
    
      <pair1> <pair2> <pair3> <pair4> <least_illustrative_pair> <most_illustrative_pair>

    The order of word pairs should match their original order found in `data/word_analogy_dev.txt`.

  - `data/word_analogy_dev_mturk_answers.txt`:
    This is the answers collected using Amazon mechanical turk for `data/word_analogy_dev.txt`. 
    The answers in this file is used as the correct answer and used to evaluate your analogy predictions. (using "evaluate_word_analogy.pl")
    For your information, the answers here are a little bit noisy.

  - `data/word_analogy_test.txt`:
    Test data file. When you are done experimenting with your model, you will generate predictions for this test data using your best models (NLL/negative sampling).
    You will not be able to run the evaluation script on the test set.

    Make sure your submission files are named: `test_preds_nll.txt`, `test_preds_neg.txt`.

## Bias Evaluation using WEAT Test

We now have seen the power of word embeddings to help learn analogies, so it is also appropriate to show the unwanted learnings of the generated embeddings. 
In this task, we will be looking at how to evaluate whether the embeddings are biased or not.

The WEAT test provides a way to measure quantifiably the bias in the word embeddings. [This paper](https://arxiv.org/pdf/1810.03611.pdf) describes the method in detail.

The basic idea is to examine the associations in word embeddings between concepts. 
It measures the degree to which a model associates sets of target words (e.g., African American names, European American names, flowers, insects) with sets of attribute words (e.g., ”stable”, ”pleasant” or ”unpleasant”). 
The association between two given words is defined as the cosine similarity between the embedding vectors for the words.

In order to run bias estimation on generated models:
```
python eval_bias.py 
    --model_path [full_model_path] 
    --weat_file_path [custom_weat_file_path]
```

This will generate the bias scores as evaluated on 5 different tasks with different sets of attributes (A and B) and targets (X and Y) as defined in the file pointed to in the `weat_file_path` (`weat.json` for the given data). This will print and dump the output in the filepath you specify.

For task addition for WEAT Test:
Follow the task definition as done for the other WEAT tasks to add custom task of your own!.

Add to the json file `custom_weat.json`, another task in the following format:
```
{
  # initial tasks....
  "custom_task": {
    "A_key": "A_val",
    "B_key": "B_val",
    "X_key": "X_val",
    "Y_key": "Y_val",
    "A_val": [
      # list of words for attribute A
    ],
    "B_val": [
      # list of words for attribute B
    ],
    "X_val": [
      # list of words for target X
    ],
    "Y_val": [
      # list of words for target Y
    ], 
  }
}
```

Ensure that the task name is `custom_task`, this will be automatically verified. Have a look at the other tasks for more clarity.

To see the scores on tasks as defined in `custom_weat.json`, run the following command:

```
python eval_bias.py --model_path [full_model_path] --weat_file_path [custom_weat_file_path] 
    --out_file [filepath_for_bias_output]
```

Example usage: `python eval_bias.py --model_path best_nll_model/word2vec_nll.model --weat_file_path data/weat.json --out_file nll_bias_output.json`

Your submission bias output files should be named `nll_bias_output.json` and `neg_bias_output.json`.

After you complete the `custom_weat.json` task, you can run the script for the given data as well as your custom data.
Your submission custom bias output files should be named `nll_custom_bias_output.json` and `neg_custom_bias_output.json`.

## Blackboard Submission Code Organisation

**PAY ATTENTION TO THE FILENAMES AND STRUCTURE!!!**

We will use scripts to organize your codes/files, and if filenames and/or the structure are incorrect, they will not be graded.

Please maintain the same organization as the boilerplate code you have been provided.
You should fill in the TODO(students) blocks in-place in the files that already exist (data.py, model.py, custom_weat.json, word_analogy.py).
Create a new folder called `submission/` and place the following files in it:
   - `test_preds_nll.txt` - Your best NLL model predictions for `data/word_analogy_test.txt`
   - `test_preds_neg.txt` - Your best negative sampling model predictions for `data/word_analogy_test.txt`
   - `nll_bias_output.json` - Results for the WEAT task on `weat.json` using your best NLL model
   - `neg_bias_output.json` - Results for the WEAT task on `weat.json` using your best negative sampling model
   - `nll_custom_bias_output.json` - Results for the custom WEAT task on `custom_weat.json` using your best NLL model
   - `neg_custom_bias_output.json` - Results for the custom WEAT task on `custom_weat.json` using your best negative sampling model
   - `gdrive_link.txt` - Should contain a `wget`able to a folder that contains your best models. The model files should be named `word2vec_nll.model` and `word2vec_neg.model`, and the folder should be named `538-hw1-<SBUID>-models`. Please make sure you provide the necessary permissions.
   - `<SBUID>_Report.pdf` - A PDF report as detailed below.

Your PDF report should contain your answer to the following questions.

1. Explanation of your code implementations for each required task.
2. Hyper-parameters explored w/ a description of what each parameter controls and how varying each of them is likely to affect the training. You need experiment with at least three hyperparameters. We suggest: skip_window, batch_size, embedding_size and max_num_steps.
3. Results (the ones you get by running the evaluation perl script - no need to submit the predictions for the DEV data) on the analogy task for five different configurations of hyperparameters, along with the best configuration (on the word pairs dev.txt). Explain these results. Are there any trends you observe? For example, what does increasing dimensions of vectors do? You should say what you learned from exploring these configurations. You are free to consult any resources to support your arguments.
4. Summary of the comparison in bias scores (for the already given data, not your custom task) obtained for the best model as obtained with loss function of NLL and negative sampling. Justification for the bias score along with the attributes and targets used must be mentioned.

Your submission folder should have the following structure:
```
538_HW1_<SBUID>
├── .gitignore
└── data/
  ├── custom_weat.json
  ├── weat.json
  ├── word_analogy_dev_mturk_answers.txt
  ├── word_analogy_dev_sample_predictions.txt
  ├── word_analogy_dev.txt
  └── word_analogy_test.txt
├── data.py
├── download_resources.sh
├── evaluate_word_analogy.pl
├── main.py
├── model.py
├── README.md
├── requirements.txt
└── submission/
  ├── test_preds_nll.txt
  ├── test_preds_neg.txt
  ├── nll_bias_output.json
  ├── neg_bias_output.json
  ├── nll_custom_bias_output.json
  ├── neg_custom_bias_output.json
  ├── gdrive_link.txt
  └── <SBUID>_Report.pdf
├── train.py
└── word_analogy.py
```

Note the name of the main folder, and the exact files that need to be submitted.
Do not submit a `__pycache__`, `models/` or `checkpoints/` folder, or the `text8` file.
Zip your main submission folder and name it `538_HW1_<SBUID>.zip`.
This means that when we unzip your submission, it results in the exact structure outlined above.

## Due Date and Collaboration

  - The assignment is due on Sep 16, 2021 at 11:59 pm EDT. You have a total of three extra days for late submission across all the assignments. You can use all three days for a single assignment, one day for each assignment – whatever works best for you. Submissions between third and fourth day will be docked 20%. Submissions between fourth and fifth day will be docked 40%. Submissions after fifth day won’t be evaluated.
  - You can collaborate to discuss ideas and to help each other for better understanding of concepts and math.
  - You should NOT collaborate on the code level. This includes all implementation activities: design, coding, and debugging.
  - You should NOT not use any code that you did not write to complete the assignment.
  - The homework will be **cross-checked**. Do not cheat at all! It’s worth doing the homework partially instead of cheating and copying your code and get 0 for the whole homework. In previous years, students have faced harsh disciplinary action as a result of the same.

##  Google Colab

Google Colab is a free resource available to everyone.
It allows you to reserve a CPU, GPU or TPU, depending on your use case.
You can run commands in a cell if you add `!` at the start of the line of the command in the cell.
To use it to train your models, all you need to do is upload all the relevant files and make sure you mimic the directory structure of the assignment.
For model training purposes, you don't need to install anything either (although it is advisable).
Note: Do not upload the data/text8 or data/text8.zip file.
Instead, run the download_resources.sh script on Colab itself (uploading is slower than downloading).
The abovementioned setup is ephemeral.
You will lose all the files when the runtime shuts down (due to inactivity or otherwise).
So, save your models when they are trained.
You can link it to your Google Drive (using your stonybrook.edu or cs.stonybrook.edu) account, and the trained models will be saved in your drive.

You will not be able to run the `evaluate_word_analogy.pl` perl script on it.
We recommend using Google Colab for model training, and doing the rest of the assignment on your local machine.

## Some Debugging Tips

You can use your debugger anywhere in your code. You can also put plain python print statements if you don't like debuggers for some reason.

If you don't know how to use a debugger, here is are some basic examples:

1. Put `import pdb; pdb.set_trace()` anywhere in your python code. When the control flow reaches there, the execution will stop and you will be provided python prompt in the terminal. You can now print and interact with variables to see what is causing problem or whether your idea to fix it will work or not.

2. If your code gives error on nth run of some routine, you can wrap it with try and except to catch it with debugger when it occurrs.
```
try:
        erroneous code
except:
        import pdb; pdb.set_trace()
```

You can, of course, use your favorite IDE to debug, that might be better. But this bare-bone debugging would itself make your debugging much more efficient than simple print statements.

## Extra Notes

  - If you add any code apart from the TODOs in the codebase (note that you don't need to), please mark it by commenting in the code itself.
  An example of the same could be:
    ```
    # Adding some_global_var for XXX
    some_global_var
    ```
  - General tips when you work on tensor computations:
    - Break the whole list of operations into smaller ones.
    - Write down the shapes of the tensors

## Credits and Disclaimer

**Credits**: This code is part of the starter package of the assignment/s used in NLP course at Stony Brook University. 
This assignment has been designed, implemented and revamped as required by many NLP TAs to varying degrees.
In chronological order of TAship they include Heeyoung Kwon, Jun Kang, Mohaddeseh Bastan, Harsh Trivedi, Matthew Matero, Nikita Soni, Sharvil Katariya, Yash Kumar Lal, Adithya V. Ganesan and Sounak Mondal. Thanks to all of them!

**Disclaimer/License**: This code is only for school assignment purpose, and **any version of this should NOT be shared publicly on github or otherwise even after semester ends**. 
Public availability of answers devalues usability of the assignment and work of several TAs who have contributed to this. 
We hope you'll respect this restriction.