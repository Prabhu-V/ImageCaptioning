import torch
from torch.utils.data import Dataset
from PIL import Image
import h5py
import json
import os


class CaptionDataset(Dataset):
    """
    Dataset class to be used in EfficientNet LSTM DataLoader to create batches
    """

    def __init__(self, data_folder, data_name, split, transform=None):
        """
        Parameters:-
        data_folder: folder where data files are stored
        data_name: base name of processed datasets
        split: TRAIN, TEST, VAL split
        transform: image transform pipeline
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open hdf5 file where images are stored
        self.h = h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')
        self.imgs = self.h['images']

        # Captions per image
        self.cpi = self.h.attrs['captions_per_image']

        # Load encoded captions from json
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths from json
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)

        # Transform the images
        self.transform = transform

        # Total number of captions
        self.dataset_size = len(self.captions)

    def __getitem__(self, i):
        img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.) # ith caption corresponds tp i/cpi th image
        if self.transform is not None:
            img = self.transform(img)

        caption = torch.LongTensor(self.captions[i])

        caplen = torch.LongTensor([self.caplens[i]])

        if (self.split=='TRAIN'):
            return img, caption, caplen
        else:
            # For validation and testing, also return all captions for that image captions to find BLEU score
            all_captions = torch.LongTensor(
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            return img, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size
   
    
class F8kDataset(Dataset):
    """
    Dataset class to be used in VIT Roberta DataLoader to create batches
    """
    def __init__(self, df, tokenizer,feature_extractor, decoder_max_length=50):
        """
        Parameters:-
        df: data frame containing dataset
        tokenizer: tokenizer to tokenize captions
        feature_extractor: feature_extractor to extract features from images
        decoder_max_length: max length for tokenizer
        """
        self.df = df
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.decoder_max_length = decoder_max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get file name + text 
        img_path = self.df['images'][idx]
        caption = self.df['captions'][idx]
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.feature_extractor(image, return_tensors="pt").pixel_values
        # add labels (input_ids) by encoding the text
        labels = self.tokenizer(caption,truncation = True,padding="max_length",max_length=self.decoder_max_length).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding