from __future__ import unicode_literals, print_function, division

import numpy as np
import argparse
import random
import math
import string
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from device import device
from helpers import split_every_n_elements
from train import trainIters
from encoder import EncoderRNN
from attnDecoder import AttnDecoderRNN
from evaluation import evaluate, evaluateRandomly, evaluateAll

SOS_token = 0
EOS_token = 1

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

class ModelData:
    def __init__(self, args):
        print(args)
        self.get_samples(args.input)
        self.get_mask(len(self.samples[0]), args.rel_mask)
        self.mask_pop(self.samples, self.mask)
        self.make_inputs_outputs(self.maskedSamples, self.hiddenMarkers)
        self.compute_maf(self.outputs)
        self.pairs = [[self.inputs[i], self.outputs[i]] for i in range(len(self.inputs))]
        self.train, self.test = train_test_split(self.pairs, test_size=args.rel_test)
    
    def make_inputs_outputs(self, maskedSamples, hiddenMarkers):
        inputs = []
        outputs = []
        for i in range(len(maskedSamples)):
            noMissing = list(filter(lambda x: x!='5' ,maskedSamples[i]))
            inputs.append(" ".join(split_every_n_elements(noMissing, 10)))
            output = split_every_n_elements(hiddenMarkers[i], 6)
            outputs.append(" ".join(output))
            
        self.inputs = inputs
        self.outputs = outputs
        return inputs, outputs 
    
    def compute_maf(self, samples):
        sampleMatrix = []
        for sample in samples:
            sampleStr = sample.replace(" ", "").strip()
            if sampleStr == "": continue
            
            sampleMatrix.append(list(sampleStr))
            
        matrix = np.array(sampleMatrix)
        self.mafArr = np.divide(np.count_nonzero(matrix == '1', axis=0), matrix.shape[0])
        return self.mafArr
    
    def mask_pop(self, popData, mask):
        maskedSamples = []
        hiddenMarkers = []
        for sample in popData:
            maskedSamples.append(['5' if mask[index] else marker for index, marker in enumerate(sample)])
            hiddenMarkers.append([marker for index, marker in enumerate(sample) if mask[index] ])
        self.maskedSamples = maskedSamples
        self.hiddenMarkers = hiddenMarkers
        return maskedSamples, hiddenMarkers
    
    def get_mask(self, length, relative_masked):
        random_positions = random.sample(range(0, length), math.ceil(length*relative_masked))
        self.mask = [ 1 if idx in random_positions else 0 for idx in range(length) ]
        return self.mask
    
    def get_samples(self, filename):
        sampleData = self.read(filename)
        self.samples = sampleData.split('\n')
    
    def read(self, filename):
        with open(filename, 'r') as f:
            return f.read()

def main(args):
    data = ModelData(args)
        
    lang_i = Lang("inputs")
    [lang_i.addSentence(input) for input in data.inputs]
    lang_o = Lang("outputs")
    [lang_o.addSentence(output) for output in data.outputs]
        
    encoder1 = EncoderRNN(lang_i.n_words, args.hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN(args.hidden_size, lang_o.n_words, args).to(device)

    if args.verbose > 0:
        print("População ###")
        print(" - Número de samples:", len(data.samples))
        print(" - > 1ª sample:", data.train[0][0])
        print(" - < 1ª sample:", data.train[0][1])
        
        print("\nMáscara ###")
        print(" - Marcadores mascarados", len(data.mask))
        
    if args.verbose > 1:
        print("Número de palavras entrada: ", lang_i.n_words)
        print("Número de palavras saída: ", lang_o.n_words)
        
    trainIters([lang_i, lang_o], data, encoder1, attn_decoder1, args)

if __name__ == '__main__':
    my_parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    my_parser.add_argument('-hidden_size', help='Número de células escondidas', default=512, type=int)
    my_parser.add_argument('-lrate', help='Taxa de aprendizado (learning rate)', default=0.01, type=float)
    my_parser.add_argument('-it', help='número de iterações', default=50000, type=int)
    my_parser.add_argument('-eval_ev', help='Evaluate every <number> of iterations', default=100, type=int)
    my_parser.add_argument('-verbose', help='Verbose 0:None 1:Info 2:Debug', default=0, type=int)
    my_parser.add_argument('-input', help='Verbose, 0: None; 1: Info; 2: Debug', default="result_aux_big.txt", type=str)
    my_parser.add_argument('-rel_mask', help='Relative number of non-observed variants to total of variants (masked variants)', default=0.3, type=float)
    my_parser.add_argument('-rel_test', help='Relative number of test samples from dataset', default=0.2, type=float)
    my_parser.add_argument('-max_length', help='Max length of input/output', default=100, type=int)
    my_parser.add_argument('-length', help='Length of the dataset', default=100, type=int)
    my_parser.add_argument('-dropout', help='Dropout', default=0.1, type=float)
    args = my_parser.parse_args()
    
    main(args)