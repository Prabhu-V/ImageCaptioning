import torch
from torch import nn
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    """
    Encoder model
    """

    def __init__(self):
        super(Encoder, self).__init__()

        effnet = torchvision.models.efficientnet_b7(weights='IMAGENET1K_V1')  # pretrained efficientnet_b7

        # Remove linear and pool layers which are used for classification
        modules = list(effnet.children())[:-2]
        self.effnet = nn.Sequential(*modules)
        
        # dont update weights as we are using pretrainrd model
        for p in self.effnet.parameters():
            p.requires_grad = False

    def forward(self, images):
        """
        Forward propagation

        Parameters:-
        images: images, a tensor of dimensions (batch_size, 3, 600, 600)

        Returns:- encoded images
        """
        out = self.effnet(images)  # (batch_size, 2560, 19, 19)
        out = out.permute(0, 2, 3, 1)  # (batch_size, 19, 19, 2560)
        return out


class Attention(nn.Module):
    """
    Attention Network
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        Parameters:-
        encoder_dim: feature size of encoded images(2560)
        decoder_dim: size of decoder's RNN
        attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmaxed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation
        
        Parameters:-
        encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size*enc_image_size, encoder_dim)
        decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        
        Returns:- attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, 19*19, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, 19*19)
        alpha = self.softmax(att)  # (batch_size, 19*19)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, 2560)

        return attention_weighted_encoding, alpha


class DecoderWithAttention(nn.Module):
    """
    LSTM Decoder
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2560, dropout=0.5):
        """
        Parameters:-
        attention_dim: size of attention network
        embed_dim: embedding size
        decoder_dim: size of decoder's LSTM
        vocab_size: size of vocabulary
        encoder_dim: feature size of encoded images
        dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout) # dropout layer
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoder LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find predicted probability for vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes parameters for easier convergence
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images

        Parameters:-
        encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size*enc_image_size, encoder_dim)
        
        Returns:- hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward propagation

        Parameters:-
        encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        
        Returns:- probability for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim) #(batch_size, 19*19, 2560)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim) will convert captions to embedding

        # I/P to initial LSTM cell
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # discarding end token
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion probability and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, 2560)
            attention_weighted_encoding = gate * attention_weighted_encoding
            #for captions having>0 word apply lstm cell once on first word at first timestep, 
            #for cap with>1 apply second lstm cell on second word at second timestep and so on...
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind