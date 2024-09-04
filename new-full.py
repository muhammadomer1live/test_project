import json
import h5py
import numpy as np
import os
from collections import defaultdict
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet101
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence


# Preprocessing COCO Annotations
def preprocess_coco_annotations(input_json, output_json, output_h5):
    with open(input_json, 'r') as f:
        data = json.load(f)

    annotations = data['annotations']
    images = data['images']

    image_id_to_captions = defaultdict(list)
    for ann in annotations:
        image_id = ann['image_id']
        caption = ann['caption']
        image_id_to_captions[image_id].append(caption)

    id_to_filename = {img['id']: img['file_name'] for img in images}

    processed_data = []
    for image_id, captions in image_id_to_captions.items():
        processed_data.append({
            'file_path': id_to_filename[image_id],
            'captions': captions,
            'image_id': image_id
        })

    with open(output_json, 'w') as f:
        json.dump(processed_data, f)

    max_length = 20
    vocab = defaultdict(int)
    for item in processed_data:
        for caption in item['captions']:
            words = caption.split(' ')
            for word in words:
                vocab[word] += 1

    vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
    vocab = {word: i + 1 for i, (word, _) in enumerate(vocab)}

    with h5py.File(output_h5, 'w') as h5f:
        captions = []
        lengths = []
        for item in processed_data:
            for caption in item['captions']:
                words = caption.split(' ')
                cap = np.zeros((max_length,), dtype='int32')
                for i, word in enumerate(words):
                    if i < max_length:
                        cap[i] = vocab.get(word, 0)
                captions.append(cap)
                lengths.append(len(words))

        captions = np.array(captions, dtype='int32')
        lengths = np.array(lengths, dtype='int32')

        h5f.create_dataset('captions', data=captions)
        h5f.create_dataset('lengths', data=lengths)


# Custom CocoDataset Class
class CocoDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        with open(annotation_file, 'r') as f:
            self.data = json.load(f)

        self.vocab = self.build_vocab()
        # Use the original COCO annotation file to initialize the COCO object
        original_annotation_file = annotation_file.replace('_processed.json', '.json')
        self.coco = COCO(original_annotation_file)

    def build_vocab(self):
        vocab = defaultdict(int)
        for item in self.data:
            for caption in item['captions']:
                words = caption.split(' ')
                for word in words:
                    vocab[word] += 1
        vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
        return {word: i + 1 for i, (word, _) in enumerate(vocab)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.image_dir, item['file_path'])
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        captions = item['captions']
        tokens = [self.vocab[word] for caption in captions for word in caption.split(' ')]
        return image, torch.tensor(tokens), item['image_id']

    def collate_fn(self, batch):
        images, captions, image_ids = zip(*batch)
        images = torch.stack(images, 0)
        lengths = [len(cap) for cap in captions]
        targets = torch.zeros(len(captions), max(lengths)).long()
        for i, cap in enumerate(captions):
            end = lengths[i]
            targets[i, :end] = cap[:end]
        return images, targets, lengths, image_ids


# Feature Extraction Function
def extract_features(image_root, output_dir, num_images=100):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model = resnet101(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-2])
    model.eval()
    model.cpu()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_paths = [os.path.join(image_root, img) for img in os.listdir(image_root) if img.endswith('.jpg')][:num_images]

    for image_path in tqdm(image_paths):
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0)

        with torch.no_grad():
            features = model(image).squeeze().detach().numpy()

        output_path = os.path.join(output_dir, os.path.basename(image_path).replace('.jpg', '.npy'))
        np.save(output_path, features)


# Model Definitions
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = resnet101(pretrained=True)
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

# Paths
train_image_dir = '/mnt/c/Users/muham/Downloads/coco/images/train2014'
val_image_dir = '/mnt/c/Users/muham/Downloads/coco/images/val2014'
train_annotation_file = '/mnt/c/Users/muham/Downloads/coco/annotations/captions_train2014.json'
val_annotation_file = '/mnt/c/Users/muham/Downloads/coco/annotations_trainval2014/annotations/captions_val2014_processed.json'

# Dataset and DataLoader
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = CocoDataset(train_image_dir, train_annotation_file, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=train_dataset.collate_fn)

val_dataset = CocoDataset(val_image_dir, val_annotation_file, transform=transform)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=2)

# Model Initialization
vocab_size = len(train_dataset.vocab) + 1
encoder = EncoderCNN(embed_size)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
optimizer = torch.optim.Adam(params, lr=learning_rate)

# Training the Models
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, captions, lengths) in enumerate(train_loader):
        images = images.cpu()
        captions = captions.cpu()
        targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

        # Forward pass
        features = encoder(images)
        outputs = decoder(features, captions)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        decoder.zero_grad()
        encoder.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i}/{total_step}], Loss: {loss.item():.4f}')

    # Save the model checkpoints
    torch.save(encoder.state_dict(), f'encoder-{epoch + 1}.ckpt')
    torch.save(decoder.state_dict(), f'decoder-{epoch + 1}.ckpt')

print("Training completed.")

# Evaluation
def evaluate_model(encoder, decoder, data_loader):
    encoder.eval()
    decoder.eval()
    coco_result = []
    with torch.no_grad():
        for images, _, lengths, image_ids in data_loader:
            images = images.cpu()
            features = encoder(images)
            sampled_ids = decoder.sample(features)
            sampled_ids = sampled_ids.cpu().numpy()

            for image_id, sampled_id in zip(image_ids, sampled_ids):
                sentence = ' '.join([train_dataset.vocab.get(word_id, '') for word_id in sampled_id])
                coco_result.append({
                    'image_id': int(image_id),
                    'caption': sentence
                })

    # Save results for COCO evaluation
    with open('coco_results.json', 'w') as f:
        json.dump(coco_result, f)

    # Load results and evaluate
    coco = COCO(val_annotation_file.replace('_processed.json', '.json'))
    coco_res = coco.loadRes('coco_results.json')
    coco_eval = COCOeval(coco, coco_res)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

# Run evaluation
evaluate_model(encoder, decoder, val_loader)
