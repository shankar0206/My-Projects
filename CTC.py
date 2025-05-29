# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from ctcdecode import CTCBeamDecoder

import numpy as np
import os
import time
import csv

path = '/data/'
os.environ['WSJ_PATH'] = path

device = "cuda" if torch.cuda.is_available() else "cpu"
num_epoch = 20
n = os.cpu_count()


class WSJ():

    def __init__(self):
        self.dev_set = None
        self.train_set = None
        self.test_set = None

    @property
    def dev(self):
        if self.dev_set is None:
            self.dev_set = load_raw(os.environ['WSJ_PATH'], 'wsj0_dev')
        return self.dev_set

    @property
    def train(self):
        if self.train_set is None:
            self.train_set = load_raw(os.environ['WSJ_PATH'], 'wsj0_train')
        return self.train_set

    @property
    def test(self):
        if self.test_set is None:
            self.test_set = (np.load(os.path.join(
                os.environ['WSJ_PATH'], 'wsj0_test.npy'),
                encoding='bytes', allow_pickle=True), None)
        return self.test_set


def load_raw(path, name):
    return (
        np.load(os.path.join(path, '{}.npy'.format(name)),
                encoding='bytes', allow_pickle=True),
        np.load(os.path.join(path, '{}_merged_labels.npy'.format(name)),
                encoding='bytes', allow_pickle=True)
    )


loader = WSJ()

trainX, trainY = loader.train
valX, valY = loader.dev
testX, _ = loader.test

PHONEME_MAP = [
    '_',  # "+BREATH+"
    '+',  # "+COUGH+"
    '~',  # "+NOISE+"
    '!',  # "+SMACK+"
    '-',  # "+UH+"
    '@',  # "+UM+"
    'a',  # "AA"
    'A',  # "AE"
    'h',  # "AH"
    'o',  # "AO"
    'w',  # "AW"
    'y',  # "AY"
    'b',  # "B"
    'c',  # "CH"
    'd',  # "D"
    'D',  # "DH"
    'e',  # "EH"
    'r',  # "ER"
    'E',  # "EY"
    'f',  # "F"
    'g',  # "G"
    'H',  # "HH"
    'i',  # "IH"
    'I',  # "IY"
    'j',  # "JH"
    'k',  # "K"
    'l',  # "L"
    'm',  # "M"
    'n',  # "N"
    'G',  # "NG"
    'O',  # "OW"
    'Y',  # "OY"
    'p',  # "P"
    'R',  # "R"
    's',  # "S"
    'S',  # "SH"
    '.',  # "SIL"
    't',  # "T"
    'T',  # "TH"
    'u',  # "UH"
    'U',  # "UW"
    'v',  # "V"
    'W',  # "W"
    '?',  # "Y"
    'z',  # "Z"
    'Z',  # "ZH"
    ' ',  # "BLANK"
]


class UtteranceDataset(Dataset):
    def __init__(self, utterance, phoneme):
        self.utterance = [torch.tensor(u) for u in utterance]
        self.phoneme = [torch.tensor(p) for p in phoneme]

    def __getitem__(self, i):
        X = self.utterance[i]
        Y = self.phoneme[i]
        return X.to(device), Y.to(device)

    def __len__(self):
        return len(self.utterance)


def collate_lines(seq_list):

    inputs, targets = zip(*seq_list)

    input_len = torch.LongTensor([len(i) for i in inputs])
    target_len = torch.LongTensor([len(t) for t in targets])

    inputs = pad_sequence(inputs)
    targets = pad_sequence(targets, batch_first=True)

    return (inputs, targets, input_len, target_len)


class phonemepredictor(nn.Module):

    def __init__(self, frame_features, num_classes, hidden_size, nlayers):
        super(phonemepredictor, self).__init__()

        self.frame_features = frame_features
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.nlayers = nlayers

        self.rnn = nn.LSTM(input_size=frame_features,
                           hidden_size=hidden_size,
                           num_layers=nlayers,
                           bidirectional=True)

        self.scoring = nn.Linear(
            hidden_size * 2, num_classes)

    def forward(self, seq_batch, input_len):
        packed_X = pack_padded_sequence(
            seq_batch,
            input_len,
            enforce_sorted=False
        )

        packed_out = self.rnn(packed_X)[0]
        out, out_lens = pad_packed_sequence(packed_out)
        out = self.scoring(out).log_softmax(2)
        return out, out_lens


def train_loop(model, train_loader, nepochs):

    model.train()
    print("Training", len(train_loader), "number of batches")
    for epochs in range(nepochs):
        batch_id = 0
        start = time.time()

        if epochs > 14:
            optimizer = optim.RMSprop(model.parameters(), lr=1e-4)

        for inputs, targets, input_len, target_len in train_loader:
            batch_id += 1

            out, out_lens = model(inputs, input_len)

            loss = criterion(out, targets, out_lens, target_len)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_id % 100 == 0:
                end = time.time()
                print(f"Epoch: {epochs + 1}")
                print(f"Batch: {batch_id}")
                print(f"Training loss: {loss.item()}")
                print(f"Time elapsed: {end - start}")
                start = end
                torch.no_grad()
        torch.save(
            model, '/hw3p2/model_' + str(epochs + 1) + '.pt')


criterion = nn.CTCLoss()
criterion = criterion.to(device)

model = phonemepredictor(frame_features=40,
                         num_classes=47,
                         hidden_size=512,
                         nlayers=4)
model = model.to(device)

optimizer = optim.RMSprop(model.parameters(), lr=1e-3)

train_dataset = UtteranceDataset(trainX, trainY)
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    collate_fn=collate_lines,
    shuffle=True)

# Training call
train_loop(model, train_loader, num_epoch)


class TestDataset(Dataset):
    def __init__(self, utterance):
        self.utterance = [torch.tensor(u) for u in utterance]

    def __getitem__(self, i):
        X = self.utterance[i]
        return X.to(device)

    def __len__(self):
        return len(self.utterance)


def collate_lines(seq_list):
    input_len = torch.LongTensor([len(seq) for seq in seq_list])
    inputs = pad_sequence(seq_list)
    return (inputs, input_len)


test_dataset = TestDataset(testX)
test_loader = DataLoader(test_dataset, batch_size=64,
                         shuffle=False, collate_fn=collate_lines)


def test_loop(model, test_loader):
    model.eval()

    output = []
    output_length = []

    for inp, length in test_loader:
        out, out_lens = model(inp, length)
        output.append(out)
        output_length.append(out_lens)
    return output, output_length


output, output_length = test_loop(model, test_loader)
decoder = CTCBeamDecoder(PHONEME_MAP, beam_width=200,
                         num_processes=n, log_probs_input=True)

val = []
lt = []

for i, data in enumerate(output):
    a, _, _, b = decoder.decode(data.transpose(0, 1), output_length[i])
    for j in range(len(a)):
        text = a[j, 0, 0:b[j, 0]]
        val.append(text)

for i in val:
    text = ''
    for j in i:
        text += PHONEME_MAP[j]
    lt.append(text)

idx = list(np.arange(523))
temp = [[a, b] for a, b in zip(idx, lt)]
temp.insert(0, ['Id', 'Predicted'])

file = open('/hw3p2/submission.csv',
            'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerows(temp)
