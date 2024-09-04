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

val_dataset = CocoDataset('data/coco/images/val2014', 'data/coco/annotations/captions_val2014_processed.json', transform=transform)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=2)

results = generate_captions(encoder, decoder, val_loader)
eval_results = evaluate_captions(results, 'data/coco/annotations/captions_val2014.json')
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
