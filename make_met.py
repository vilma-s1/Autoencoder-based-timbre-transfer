"""
Generate speaker embeddings and metadata for training using openL3 library
"""

import os
import pickle
import requests
import soundfile as sf
import torchopenl3
import numpy as np
import torch
from torch.nn import Conv1d

#num_uttrs = 8  # number of wav files used for each speaker/instrument

# Directory containing mel-spectrograms
rootDir = '/content/drive/MyDrive/timbre_transfer/urmp_train_data'
dirName, subdirList, _ = next(os.walk(rootDir))
print('Found directory: %s' % dirName)


rootDir2 = '/content/drive/MyDrive/timbre_transfer/spmel_train'
dirName2, subdirList2, _ = next(os.walk(rootDir2))

speakers = []
for speaker in sorted(subdirList):
    print('Processing speaker: %s' % speaker)
    utterances = []
    utterances.append(speaker)
    _, _, fileList = next(os.walk(os.path.join(dirName, speaker)))

    # make speaker embedding
    #assert len(fileList) >= num_uttrs
    #idx_uttrs = np.random.choice(len(fileList), size=num_uttrs, replace=False)
    idx_uttrs = np.random.choice(len(fileList), size=len(fileList), replace=False)
    embs = []
    for i in range(len(fileList)):
    #for i in range(num_uttrs):

        audio, sr = sf.read(os.path.join(dirName, speaker, fileList[idx_uttrs[i]]))

        emb, ts = torchopenl3.get_audio_embedding(audio, sr, content_type="music", input_repr="mel128",
                                                  embedding_size=512)
        in_chan = len(ts[0])  # size of input channels for the conv layer
        conv = Conv1d(in_chan, 1, 2, stride=2)  # initialise convolution layer
        emb = conv(emb)  # apply dimensionality reduction to original embeddings
        emb = torch.squeeze(emb, 1)  # remove 1 dimension
        embs.append(emb.detach().squeeze().cpu().numpy())

    utterances.append(np.mean(embs, axis=0))


    _, _, fileList2 = next(os.walk(os.path.join(dirName2, speaker)))
    for fileName in sorted(fileList2):
        utterances.append(os.path.join(speaker,fileName))
        print(utterances)
    speakers.append(utterances)

with open(os.path.join(rootDir2, 'train.pkl'), 'wb') as handle:
    pickle.dump(speakers, handle)