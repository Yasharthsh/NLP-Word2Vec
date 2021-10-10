"""
author-gh: @adithya8
editor-gh: ykl7
"""

import os
import pickle
import numpy as np
import argparse
np.random.seed(1234)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./NLL', help='Base directory of folder where models are saved')
    parser.add_argument('--input_filepath', type=str, default='./data/word_analogy_test.txt', help='Word analogy file to evaluate on')
    parser.add_argument('--output_filepath', type=str, default='data/test_preds_nll.txt', help='Predictions filepath')
    parser.add_argument("--loss_model", help="The loss function for training the word vector", default="nll", choices=["nll", "neg"])
    args, _ = parser.parse_known_args()
    return args

def read_data(file_path):
    with open(file_path,'r') as f:
        data = f.readlines()
    
    candidate, test = [], []
    for line in data:
        a, b = line.strip().split("||")
        a = [i[1:-1].split(":") for i in a.split(",")]
        b = [i[1:-1].split(":") for i in b.split(",")]
        candidate.append(a)
        test.append(b)
    return candidate, test

def get_embeddings(examples, embeddings):

    """
    For the word pairs in the 'examples' array, fetch embeddings and return.
    You can access your trained model via dictionary and embeddings.
    dictionary[word] will give you word_id
    and embeddings[word_id] will return the embedding for that word.

    word_id = dictionary[word]
    v1 = embeddings[word_id]

    or simply

    v1 = embeddings[dictionary[word_id]]
    """

    norm = np.sqrt(np.sum(np.square(embeddings),axis=1,keepdims=True))
    normalized_embeddings = embeddings/norm

    embs = []
    
    for line in examples:
        
        temp = []
        for pairs in line:
            temp.append([ normalized_embeddings[dictionary[pairs[0]]], normalized_embeddings[dictionary[pairs[1]]] ])
        embs.append(temp)
    result = np.array(embs)

    return result

def evaluate_pairs(candidate_embs, test_embs):


    best_pairs = []
    worst_pairs = []

    
    # print(candidate_embs.shape)
    # print(test_embs.shape) 

    #get embedding for the candidate words and get difference between the vectors
    candidate  = (candidate_embs[:,:,0,:]-candidate_embs[:,:,1,:])
    candidate = np.mean(candidate, axis=1)
    test = (test_embs[:,:,0,:]-test_embs[:,:,1,:])

    #print(candidate.shape)
    #print(test_embs.shape)

    output = []
    i=0

    #while loop is for each of the word_pair (on the right side of the "||" symbol)
    while i<test.shape[1]:
        row_instance = []
        j=0
        # loop for all the rows in the training file
        while j<len(candidate):
            # cosine for each row
            row_instance.append(np.tensordot(candidate[j,:].reshape(-1,1), test[j,i,:].reshape(-1,1)))            
            j+=1
        output.append(row_instance)
        i+=1
    output = np.array(output)
    #get worst and best pairs
    worst_pairs = np.argmin(output, axis=0)
    best_pairs = np.argmax(output, axis=0)
    
    return best_pairs, worst_pairs

def write_solution(best_pairs, worst_pairs, test, path):

    """
    Write best and worst pairs to a file, that can be evaluated by evaluate_word_analogy.pl
    """
    
    ans = []
    for i, line in enumerate(test):
        temp = [f'"{pairs[0]}:{pairs[1]}"' for pairs in line]
        temp.append(f'"{line[worst_pairs[i]][0]}:{line[worst_pairs[i]][1]}"')
        temp.append(f'"{line[best_pairs[i]][0]}:{line[best_pairs[i]][1]}"')
        ans.append(" ".join(temp))

    with open(path, 'w') as f:
        f.write("\n".join(ans))


if __name__ == '__main__':

    args = parse_args()

    loss_model = args.loss_model
    model_path = args.model_path
    input_filepath = args.input_filepath

    print(f'Model file: {model_path}/word2vec_{loss_model}.model')
    model_filepath = os.path.join(model_path, 'word2vec_%s.model'%(loss_model))

    dictionary, embeddings = pickle.load(open(model_filepath, 'rb'))

    candidate, test = read_data(input_filepath)

    candidate_embs = get_embeddings(candidate, embeddings)
    test_embs = get_embeddings(test, embeddings)

    best_pairs, worst_pairs = evaluate_pairs(candidate_embs, test_embs)

    out_filepath = args.output_filepath
    print(f'Output file: {out_filepath}')
    write_solution(best_pairs, worst_pairs, test, out_filepath)