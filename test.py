# -*- coding: utf-8 -*-
import os

import librosa
import torch
import numpy as np
import time
import random
from model import Wav2Letter
from data.data_loader import SpectrogramDataset
from decoder import GreedyDecoder, PrefixBeamSearchLMDecoder

test_dataset = 'mycsvfile.csv'
model_path = 'models/wav2letter/epoch_15.pth'
beamSearch = '5,0.3,5,1e-3'
decoderVar = 'greedy'
lmPath = ''
print_samples = False
print_all = True


# parser = argparse.ArgumentParser(description='Wav2Letter evaluation')
# parser.add_argument('--test-manifest', metavar='DIR', help='path to test manifest csv', default=test_dataset)
# parser.add_argument('--cuda', default=True, dest='cuda', action='store_true', help='Use cuda to execute model')
# parser.add_argument('--seed', type=int, default=1337)
# parser.add_argument('--print-samples', default=False, action='store_true', help='Print some samples to output')
# parser.add_argument('--print-all', default=True, action='store_true', help='Print all samples to output')
# parser.add_argument('--model-path', type=str, default=model_path, help='Path to model.tar to evaluate')
# parser.add_argument('--decoder', type=str, default='greedy',
#                     help='Type of decoder to use.  "greedy", or "beam". If "beam", can specify LM with to use with "--lm-path"')
# parser.add_argument('--lm-path', type=str, default='',
#                     help='Path to arpa lm file to use for testing. Default is no LM.')
# parser.add_argument('--beam-search-params', type=str, default='5,0.3,5,1e-3',
#                     help='comma separated value for k,alpha,beta,prune. For example, 5,0.3,5,1e-3')


def get_beam_search_params(param_string):
    params = param_string.split(',')
    if len(param_string) != 4:
        return {}
    k, alpha, beta, prune = map(float, params)
    return {"k": k, "alpha": alpha, "beta": beta, "prune": prune}


def get_decoder(decoder_type, lm_path, labels, beam_search_params):
    if decoder_type == 'beam':
        decoder = PrefixBeamSearchLMDecoder(lm_path, labels, **beam_search_params)
    else:
        if not decoder_type == 'greedy':
            print('Decoder type not recognized, defaulting to greedy')
        decoder = GreedyDecoder(labels)
    return decoder


def test():
    set_random_seeds()
    print('starting as %s' % time.asctime())
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = Wav2Letter.load_model(model_path)
    model.to(device)
    model.eval()
    dataset = SpectrogramDataset(test_dataset, model.audio_conf, model.labels)
    decoder = get_decoder(decoderVar, lmPath, model.labels,
                          get_beam_search_params(beamSearch))
    with torch.no_grad():
        num_samples = len(dataset)
        index_to_print = random.randrange(num_samples)
        cer = np.zeros(num_samples)
        wer = np.zeros(num_samples)
        for idx, (data) in enumerate(dataset):
            inputs, targets, file_paths, text = data
            out = model(torch.FloatTensor(inputs).unsqueeze(0).to(device))
            out_sizes = torch.IntTensor([out.size(1)])
            predicted_texts = decoder.decode(probs=out, sizes=out_sizes)[0]
            cer[idx] = decoder.cer_ratio(text, predicted_texts)
            wer[idx] = decoder.wer_ratio(text, predicted_texts)
            if (idx == index_to_print and print_samples) or print_all:
                print(text)
                print('Decoder result: ' + predicted_texts)
                print('Raw acoustic: ' + ''.join(map(lambda i: model.labels[i], torch.argmax(out.squeeze(), 1))))
    print('CER:%f, WER:%f' % (cer.mean(), wer.mean()))


def testForWeb(csvName):
    model_path = os.path.dirname(os.path.abspath(__file__)) + '/models/wav2letter/epoch_15.pth'
    csvfile = os.path.dirname(os.path.abspath(__file__)) + '/UI/' + 'csvFile/' + csvName
    set_random_seeds()
    print('starting as %s' % time.asctime())
    print()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = Wav2Letter.load_model(model_path)
    model.to(device)
    model.eval()
    dataset = SpectrogramDataset(csvfile, model.audio_conf, model.labels)
    decoder = get_decoder(decoderVar, lmPath, model.labels,
                          get_beam_search_params(beamSearch))
    with torch.no_grad():
        num_samples = len(dataset)
        index_to_print = random.randrange(num_samples)
        cer = np.zeros(num_samples)
        wer = np.zeros(num_samples)
        predicted_texts = ""
        for idx, (data) in enumerate(dataset):
            inputs, targets, file_paths, text = data
            out = model(torch.FloatTensor(inputs).unsqueeze(0).to(device))
            out_sizes = torch.IntTensor([out.size(1)])
            predicted_texts = decoder.decode(probs=out, sizes=out_sizes)[0]
            # cer[idx] = decoder.cer_ratio(text, predicted_texts)
            # wer[idx] = decoder.wer_ratio(text, predicted_texts)

        return predicted_texts


def set_random_seeds(seed=1337):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


if __name__ == '__main__':
    # arguments = parser.parse_args()
    test()
