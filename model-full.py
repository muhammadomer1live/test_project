import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
from pycocotools.coco import COCO


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet101(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)

    def forward(self, features, captions):
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs

    def sample(self, features, states=None):
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(20):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            _, predicted = outputs.max(1)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)
        sampled_ids = torch.stack(sampled_ids, 1)
        return sampled_ids

# Hyperparameters
embed_size = 256
hidden_size = 512
num_layers = 1
num_epochs = 5
batch_size = 128
learning_rate = 0.001

# Load the dataset (you will need to implement a custom dataset class)
from coco_dataset import CocoDataset

train_dataset = CocoDataset('/mnt/c/Users/muham/Downloads/coco/images/train2014', '/mnt/c/Users/muham/Downloads/coco/annotations/captions_train2014_processed.json', transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=train_dataset.collate_fn)

vocab_size = len(train_dataset.vocab)
encoder = EncoderCNN(embed_size).cuda()
decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers).cuda()

criterion = nn.CrossEntropyLoss()
params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
optimizer = optim.Adam(params, lr=learning_rate)

for epoch in range(num_epochs):
    for i, (images, captions, lengths) in enumerate(train_loader):
        images = images.cuda()
        captions = captions.cuda()
        targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

        features = encoder(images)
        outputs = decoder(features, captions)
        outputs = pack_padded_sequence(outputs, lengths, batch_first=True)[0]

        loss = criterion(outputs, targets)
        decoder.zero_grad()
        encoder.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}')

torch.save(decoder.state_dict(), 'decoder.pth')
torch.save(encoder.state_dict(), 'encoder.pth')

def generate_captions(encoder, decoder, data_loader):
    encoder.eval()
    decoder.eval()
    results = []
    for images, image_ids in data_loader:
        images = images.cuda()
        features = encoder(images)
        sampled_ids = decoder.sample(features)
        sampled_ids = sampled_ids.cpu().numpy()

        for i, image_id in enumerate(image_ids):
            caption = []
            for word_id in sampled_ids[i]:
                word = data_loader.dataset.vocab.idx2word[word_id]
                if word == '<end>':
                    break
                caption.append(word)
            sentence = ' '.join(caption)
            results.append({"image_id": image_id, "caption": sentence})
    return results

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOEvalCap

def evaluate_captions(results, annotation_file):
    coco = COCO(annotation_file)
    coco_results = coco.loadRes(results)
    coco_eval = COCOEvalCap(coco, coco_results)
    coco_eval.evaluate()
    return coco_eval.eval

val_dataset = CocoDataset('/mnt/c/Users/muham/Downloads/coco/images/val2014', '/mnt/c/Users/muham/Downloads/coco/annotations/captions_val2014_processed.json', transform=transform)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=2)

results = generate_captions(encoder, decoder, val_loader)
eval_results = evaluate_captions(results, '/mnt/c/Users/muham/Downloads/coco/annotations/captions_val2014.json')
print(eval_results)



#_______________________


def compute_rewards(baseline_captions, sampled_captions, coco):
    baseline_results = coco.loadRes(baseline_captions)
    sampled_results = coco.loadRes(sampled_captions)

    coco_eval = COCOEvalCap(coco, baseline_results)
    coco_eval.evaluate()
    baseline_scores = coco_eval.eval['CIDEr']

    coco_eval = COCOEvalCap(coco, sampled_results)
    coco_eval.evaluate()
    sampled_scores = coco_eval.eval['CIDEr']

    rewards = sampled_scores - baseline_scores
    return rewards

# Hyperparameters for SCST
learning_rate_scst = 1e-5
scst_epochs = 3

optimizer_scst = optim.Adam(params, lr=learning_rate_scst)

for epoch in range(scst_epochs):
    for i, (images, captions, lengths) in enumerate(train_loader):
        images = images.cuda()
        captions = captions.cuda()
        targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

        features = encoder(images)
        outputs = decoder(features, captions)
        outputs = pack_padded_sequence(outputs, lengths, batch_first=True)[0]

        loss = criterion(outputs, targets)
        decoder.zero_grad()
        encoder.zero_grad()
        loss.backward()
        optimizer_scst.step()

        if i % 100 == 0:
            print(f'Epoch [{epoch}/{scst_epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}')

        if i % 500 == 0:
            baseline_captions = generate_captions(encoder, decoder, val_loader)
            sampled_captions = generate_captions(encoder, decoder, val_loader)
            rewards = compute_rewards(baseline_captions, sampled_captions, coco)
            print(f'Step [{i}/{len(train_loader)}], Rewards: {rewards:.4f}')

torch.save(decoder.state_dict(), 'decoder_scst.pth')
torch.save(encoder.state_dict(), 'encoder_scst.pth')
