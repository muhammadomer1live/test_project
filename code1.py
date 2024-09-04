import json
import h5py
import numpy as np
from collections import defaultdict

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
            'captions': captions
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

# Process the training and validation annotations
preprocess_coco_annotations(
    '/mnt/c/Users/muham/Downloads/coco/annotations_trainval2014/annotations/captions_train2014.json',
    '/mnt/c/Users/muham/Downloads/coco/annotations_trainval2014/annotations/captions_train2014_processed.json',
    '/mnt/c/Users/muham/Downloads/coco/annotations_trainval2014/annotations/captions_train2014.h5'
)

preprocess_coco_annotations(
    '/mnt/c/Users/muham/Downloads/coco/annotations_trainval2014/annotations/captions_val2014.json',
    '/mnt/c/Users/muham/Downloads/coco/annotations_trainval2014/annotations/captions_val2014_processed.json',
    '/mnt/c/Users/muham/Downloads/coco/annotations_trainval2014/annotations/captions_val2014.h5'
)
