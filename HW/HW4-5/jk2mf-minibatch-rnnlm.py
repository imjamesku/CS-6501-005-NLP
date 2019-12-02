import torch
from torch import nn
import time
import math
import torch.optim as optim


class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers):
        super(SimpleLSTM, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                            n_layers, batch_first=True)
#         self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_dim, output_size)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden):
        batch_size = x.size(0)
        sequence_len = x.size(1)
        x = x.long()
        # print('x device')
        # print(x.device)
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        # lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

#         out = self.dropout(lstm_out)
        out = self.fc(lstm_out)
        # out = self.sigmoid(out)

        out = out.view(batch_size, sequence_len, -1)
#         out = out[:,-1]
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden


def loadFile(path):
    sentences = []
    with open(path, 'r', encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            words = line.split()
            sentences.append(words)
    return sentences


class Vocabulary(object):
    def __init__(self):
        self.word2idx = {'<pad>': 0}
        self.idx2word = ['<pad>']

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    def wordToIdx(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]


def getBatch(data, i, batchSize):
    return data[i*batchSize: (i+1)*batchSize]


def padSentences(data):
    maxLen = max(len(sentence) for sentence in data)
    return [sentence + [0]*(maxLen-len(sentence)) for sentence in data]


def calPpl(data, model):
    crossEntropyLoss = nn.CrossEntropyLoss(reduction='sum')
    hidden = model.init_hidden(1)
    loss = 0
    totalPredictions = 0
    for i, sentence in enumerate(data):
        data, targets = torch.tensor(
            sentence[:-1], device=device).reshape(1, -1), torch.tensor(sentence[1:], device=device)
        totalPredictions += len(targets)
        output, hidden = model(data, hidden)
        loss += crossEntropyLoss(output.view(-1, len(vocab)), targets).item()
    return math.exp(loss/totalPredictions)


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


criterion = nn.CrossEntropyLoss()


def train(model, dataset, batch_size=1, clip=0.25, lr=20, log_interval=50, epoches=1):
    # Turn on training mode which enables dropout.
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    # optimizer = optim.Adagrad(model.parameters(), lr=lr)
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    start_time = time.time()
    ntokens = len(vocab)
    numBatches = len(dataset) // batch_size

    for epoch in range(epoches):
        total_loss = 0.
        hidden = model.init_hidden(batch_size)
        for i in range(numBatches):
            batch = getBatch(dataset, i, batch_size)
            batch = torch.tensor(batch, device=device)
            data, targets = batch[:, :-1], batch[:, 1:]
            # print('data shape: {}'.format(data.shape))
            # print('targets shape: {}'.format(targets.shape))
            hidden = repackage_hidden(hidden)

            # model.zero_grad()
            optimizer.zero_grad()
            output, hidden = model(data, hidden)
            # print('output shape: {}'.format(output.shape))
            # print('hidden shape: {}'.format(hidden[0].shape))
            loss = criterion(output.view(-1, ntokens), targets.reshape(-1))
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            # for p in model.parameters():
            #     p.data.add_(-lr, p.grad.data)

            total_loss += loss.item()

            if i % log_interval == 0 and i > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                      'loss {:5.2f} | ppl {:8.2f}'.format(
                          epoch, i, numBatches, lr,
                          elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()


prefix = '/content/drive/My Drive/data-for-lang-modeling/'
# prefix = 'data-for-lang-modeling/'
trainSentences = loadFile(
    prefix+'trn-wiki.txt')
devSentences = loadFile(
    prefix+'dev-wiki.txt')

vocab = Vocabulary()
for s in trainSentences:
    for word in s:
        vocab.add_word(word)

trainSentences = [[vocab.wordToIdx(word) for word in s]
                  for s in trainSentences]
devSentences = [[vocab.wordToIdx(word) for word in s] for s in devSentences]
trnPadded = padSentences(trainSentences)
# trnPadded = trnPadded[:200]
print('len vocab={}'.format(len(vocab)))
model = SimpleLSTM(vocab_size=len(vocab), output_size=len(vocab),
                   embedding_dim=32, hidden_dim=32, n_layers=1)
device = torch.device('cuda')
model.to(device)
train(model, trnPadded, lr=5, batch_size=16, epoches=5, log_interval=200)

torch.save(model.state_dict(),
           '/content/drive/My Drive/NLP_HW4-5/models/simpleLSTM_minibatch.pth')

print(calPpl(devSentences, model))
print(calPpl(trainSentences, model))
