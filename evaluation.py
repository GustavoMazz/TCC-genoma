import torch
import random
import numpy as np

from tensorHelpers import tensorsFromPair, tensorFromSentence
from device import device

SOS_token = 0
EOS_token = 1

def evaluateRandomly(pairs, lang, encoder, decoder, args, n=0):
    for i in range(n):
        pair = random.choice(pairs)
        output_words, attentions = evaluate(lang, encoder, decoder, pair[0], args)
        output_sentence = ' '.join(output_words)
        print('=', pair[1])
        print('<', output_sentence)
        print('')
        
def evaluateAll(pairs, lang, encoder, decoder, args, n=None):
    acertos = 0
    tentativas = 0
    if n:
        pairs = random.sample(pairs, n)
        
    for idx, pair in enumerate(pairs):
        real = np.array(list(pair[1].replace(" ", "")), dtype=int)
        output_words, attentions = evaluate(lang, encoder, decoder, pair[0], args)
        prediction = np.array(list(''.join(output_words).replace("<EOS>", "")), dtype=int)
        if len(prediction) < len(real):
            teste = np.full(len(real)-len(prediction), 2, dtype=int)
            prediction = np.concatenate((prediction, teste))
            
        tentativas += len(real)
        acertos += np.sum(real == prediction[:len(real)])
    return acertos/tentativas
        
def evaluate(lang, encoder, decoder, sentence, args):
    with torch.no_grad():
        input_tensor = tensorFromSentence(lang[0], sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(args.max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(args.max_length, args.max_length)

        for di in range(args.max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(lang[1].index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]