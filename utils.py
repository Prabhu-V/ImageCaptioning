import os
import numpy as np
import pandas as pd
from PIL import Image
import h5py
import json
import torch
from cv2 import imread, resize
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample


def create_input_files(dataset, karpathy_json_path, image_folder, captions_per_image, min_word_freq, output_folder,
                       max_len=100):
    """
    Creates input files for training, validation, and test data for EfficientNet LSTM model

    Parameters:-
    dataset: name of dataset
    karpathy_json_path: path of Karpathy JSON file with splits and captions
    image_folder: folder with downloaded images
    captions_per_image: number of captions to sample per image
    min_word_freq: words occuring less frequently than this threshold are binned as <unk>s
    output_folder: folder to save files
    max_len: don't sample captions longer than this length
    """

    # Read Karpathy JSON
    with open(karpathy_json_path, 'r') as j:
        data = json.load(j)

    # Read image paths and captions for each image
    train_image_paths = []
    train_image_captions = []
    val_image_paths = []
    val_image_captions = []
    test_image_paths = []
    test_image_captions = []
    word_freq = Counter()

    for img in data['images']:
        captions = []
        for c in img['sentences']:
            # Update word frequency counter
            word_freq.update(c['tokens'])
            if len(c['tokens']) <= max_len:
                captions.append(c['tokens'])

        if len(captions) == 0:
            continue

        path = os.path.join(image_folder, img['filename'])

        if img['split'] in {'train'}:
            train_image_paths.append(path)
            train_image_captions.append(captions)
        elif img['split'] in {'val'}:
            val_image_paths.append(path)
            val_image_captions.append(captions)
        elif img['split'] in {'test'}:
            test_image_paths.append(path)
            test_image_captions.append(captions)

    # Create word map
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq] #avoid less frequent words and mark as <unk>s
    word_map = {k: v + 1 for v, k in enumerate(words)} #create word mapping starting from 1 reserve 0 for <pad>
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    # Create a base/root name for all output files
    base_filename = dataset + '_' + str(captions_per_image) + '_cap_per_img_' + str(min_word_freq) + '_min_word_freq'

    # Save word map to a JSON
    with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w') as j:
        json.dump(word_map, j)

    # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
    seed(123)
    for impaths, imcaps, split in [(train_image_paths, train_image_captions, 'TRAIN'),
                                   (val_image_paths, val_image_captions, 'VAL'),
                                   (test_image_paths, test_image_captions, 'TEST')]:

        with h5py.File(os.path.join(output_folder, split + '_IMAGES_' + base_filename + '.hdf5'), 'a') as h:
            # Make a note of the number of captions we are sampling per image
            h.attrs['captions_per_image'] = captions_per_image

            # Create dataset inside HDF5 file to store images
            images = h.create_dataset('images', (len(impaths), 3, 600, 600), dtype='uint8') #600*600 img size and 3 for rgb

            print("\nReading %s images and captions, storing to file...\n" % split)

            enc_captions = []
            caplens = []

            for i, path in enumerate(tqdm(impaths)):

                # Sample captions
                if len(imcaps[i]) < captions_per_image:
                    captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
                else:
                    captions = sample(imcaps[i], k=captions_per_image)

                # Read images
                img = imread(impaths[i])
                if len(img.shape) == 2:
                    img = img[:, :, np.newaxis]
                    img = np.concatenate([img, img, img], axis=2)
                img = resize(img, (600,600))
                img = img.transpose(2, 0, 1)

                # Save image to HDF5 file
                images[i] = img

                for j, c in enumerate(captions):
                    # Encode captions
                    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                        word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))

                    # Find caption lengths
                    c_len = len(c) + 2

                    enc_captions.append(enc_c)
                    caplens.append(c_len)

            # Save encoded captions and their lengths to JSON files
            with open(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:
                json.dump(enc_captions, j)

            with open(os.path.join(output_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w') as j:
                json.dump(caplens, j)


def create_input_df(karpathy_json_path, image_folder):
    """
    Creates input files for training, validation, and test data for VIT Roberta model

    Parameters:-
    karpathy_json_path: path of Karpathy JSON file with splits and captions
    image_folder: folder with downloaded images
    
    Returns:-
    Dataframe with image path, caption and split
    """
    with open(karpathy_json_path, 'r') as openfile: #Path to json file
        data = json.load(openfile)

    df = pd.DataFrame([])
    images=[]
    captions = []
    split = []
    for img in data['images']:
        for c in img['sentences']:
            captions.append(c['raw'])
            images.append(os.path.join(image_folder, img['filename']))
            split.append(img['split'])
            
    df['images'] = images
    df['captions'] = captions
    df['split'] = split
    return(df)


def decay_learning_rate(optimizer, decay_factor):
    """
    Decays learning rate of optimizer by decay_factor

    Parameters:-
    optimizer: optimizer whose learning rate must be decayed
    decay_factor: factor in interval (0, 1) to decay learning rate with
    """

    print("\nDecaying learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))
    

def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients

    Parameters:-
    optimizer: optimizer with the gradients to be clipped
    grad_clip: value at which to clip the gradients
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, decoder_optimizer,
                    bleu4, is_best):
    """
    Save the model
    
    Parameters:-
    data_name: base name of dataset
    epoch: epoch number
    epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    encoder: encoder model
    decoder: decoder model
    decoder_optimizer: optimizer to update decoder weights
    bleu4: validation BLEU-4 score
    is_best: if checkpoint best or not
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'bleu-4': bleu4,
             'encoder': encoder,
             'decoder': decoder,
             'decoder_optimizer': decoder_optimizer}
    filename = 'checkpoint_' + data_name + '.pth.tar'
    torch.save(state, filename)
    # Save best model
    if is_best:
        torch.save(state, 'BEST_' + filename)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(scores, targets, k):
    """
    Calculates top-k accuracy, from predictions and targets

    Parameters:-
    scores: predictions from the model
    targets: true labels
    k: k in top-k accuracy

    Returns:- top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)