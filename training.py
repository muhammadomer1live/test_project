
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence

# Hyperparameters
embed_size = 256
hidden_size = 512
num_layers = 1
num_epochs = 5
batch_size = 128
learning_rate = 0.001

# Load the dataset (you will need to implement a custom dataset class)
from coco_dataset import CocoDataset

train_dataset = CocoDataset('data/coco/images/train2014', 'data/coco/annotations/captions_train2014_processed.json', transform=transform)
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
