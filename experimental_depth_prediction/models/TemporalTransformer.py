
import sys
import os
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import sys
from experimental_depth_prediction.models.ResNet import ResNet
from experimental_depth_prediction.models.archival.VAE import VanillaVAE



import torch 
import math
def positionalencoding1d(d_model, length):
    """
    Build a sinusoidal positional encoding.

    Args:
        d_model (int): Embedding dimension. Must be even.
        length (int): Number of positions.

    Returns:
        torch.Tensor: (length, d_model) encoding.
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe



def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)




class MultiHeadedSelfAttention(nn.Module):
    """Multi-Headed Dot Product Attention"""
    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.n_heads = num_heads
        self.scores = None # for visualization

    def forward(self, x, mask):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        if mask is not None:
            mask = mask[:, None, None, :].float()
            scores -= 10000.0 * (1.0 - mask)
        # print('scores shape', scores.shape)
        scores = self.drop(F.softmax(scores, dim=-1))
        # plt.imshow(scores[0][0].cpu().detach().numpy())
        # plt.show()
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scores
        return h



class PositionWiseFeedForward(nn.Module):
    """FeedForward Neural Networks for each position"""
    def __init__(self, dim, ff_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, dim)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    """Transformer Block"""
    def __init__(self, dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.attn = MultiHeadedSelfAttention(dim, num_heads, dropout)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.pwff = PositionWiseFeedForward(dim, ff_dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        h = self.drop(self.proj(self.attn(self.norm1(x), mask)))
        x = x + h
        h = self.drop(self.pwff(self.norm2(x)))
        x = x + h
        return x


class Transformer(nn.Module):
    """Transformer with Self-Attentive Blocks"""
    def __init__(self, num_layers, dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for i, block in enumerate(self.blocks):
            x = block(x, mask)
        return x



class CNN(nn.Module):
    def __init__(self, in_channels, num_features):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128, num_features)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.shape[0], -1)        
        x = F.relu(self.fc1(x))

        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TemporalTransformer(nn.Module):
    """
    Transformer over per-frame CNN embeddings of a thermal image sequence.

    Embeds each frame with the backbone, prepends a class token and adds a positional embedding, runs
    the sequence through a transformer and maps the result to end_dimensions outputs with a linear
    layer.

    Args:
        transformer_layers (int): Number of transformer blocks.
        dim (int): Embedding dimension.
        num_heads (int): Attention heads per block.
        seq_len (int): Frames per input sequence. One position is added for the class token.
        ff_dim (int): Hidden size of the feed-forward layer in each block.
        dropout (float): Dropout rate.
        channels (int): Input channels per frame.
        backbone (str): 'ResNet' or 'CNN'. Any other value leaves the model without a backbone.
        end_dimensions (int): Output size, H*W for SDF or 2*H*W for binary.
        encoding (str): 'learnable' or 'sinusoidal' positional embedding.
        init_dim (int): Initial channel count of the ResNet backbone.
        image_size (int): Side length of the input frames, for the ResNet backbone.
        device (str, optional): Device the backbone is moved to. Defaults to CUDA if available.
    """
    
    def __init__(self, 
                     transformer_layers = 4,
                     dim = 256,
                     num_heads = 8,
                     seq_len = 10,
                     ff_dim = 1024,
                     dropout = 0.1,
                     channels = 3,
                     backbone = 'ResNet',
                     end_dimensions = 8192,
                     encoding = 'learnable',
                     init_dim = 16,
                     image_size  = 16,
                     device = None):
            super(TemporalTransformer, self).__init__()
            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dim_mults=(1, 1, 1,)
            if backbone == 'ResNet':
                cnn = ResNet(dim = image_size, init_dim = init_dim, channels = channels, dim_mults  = dim_mults, last_dim  = dim)
            elif backbone == 'CNN':
                cnn = CNN(channels, dim)
            else:
                cnn = None
            if device is None:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.device = device
            # elif backbone == 'VAE':
            #     cnn = VanillaVAE(in_channels = 3, latent_dim = dim)
            #     cnn.load_state_dict(torch.load('vae_ckpt.pth')[0])
            #     cnn.eval()
            self.backbone = backbone
            transformer = Transformer(transformer_layers, dim, num_heads, ff_dim, dropout)
            if cnn is not None:
                self.cnn = cnn.to(self.device)#.cuda()
            self.transformer = transformer#.to(device)
            self.dim = dim
            self.fc4 = nn.Linear(self.dim, end_dimensions)
            self.class_token = nn.Parameter(torch.zeros(1, 1, self.dim))
            self.seq_len = seq_len + 1
            if encoding == 'learnable':
                self.positional_embedding = nn.Parameter(torch.zeros(1, self.seq_len, self.dim))
            elif encoding == 'sinusoidal':
                self.positional_embedding = positionalencoding1d(self.dim, self.seq_len)[None]
            # self.positional_embedding = nn.Parameter(torch.zeros(1, self.seq_len, self.dim))
            self.norm = nn.LayerNorm(self.dim, eps=1e-6)
            # self.tanh = nn.Tanh()
    def embed(self, x):
        if len(x.shape) == 6:
            # All exposures or multiple streams are being used 
            # x_2 = x[:, :, 1, :, :, :]
            # x_1 = x[:, :, 0, :, :, :]
            original_batch_size = x.shape[0]
            x = torch.concat((x[:,:,0], x[:,:,1]))
        # b, s, c, h, w = x.shape
            combined_exposures = True     
        else:
            combined_exposures = False       
        b, s, c, h, w = x.shape
        x = x.flatten(end_dim = 1)
        # patch embeddings in CNN
        if self.backbone == 'VAE':
            cnn_output = self.cnn(x)
            cnn_embeddings = cnn_output[2]
        else:
            cnn_embeddings = self.cnn(x)
        x = cnn_embeddings.unflatten(0, (b, s))
        # add class token 
        x = torch.cat((self.class_token.expand(b, -1, -1).to(self.device), x), dim=1)
        # add positional embedding
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x_ready_for_transformer = x + self.positional_embedding.to(self.device)
        x = self.transformer(x_ready_for_transformer)#.to(device))
        if combined_exposures:
            x = x[:original_batch_size] + x[original_batch_size:]
        return x
    def forward(self, x):
        # combine batch and sequence dimensions
        if len(x.shape) == 6:
            # All exposures or multiple streams are being used 
            # x_2 = x[:, :, 1, :, :, :]
            # x_1 = x[:, :, 0, :, :, :]
            original_batch_size = x.shape[0]
            x = torch.concat((x[:,:,0], x[:,:,1]))
        # b, s, c, h, w = x.shape
            combined_exposures = True     
        else:
            combined_exposures = False 
        
        b, s, c, h, w = x.shape
        x = x.flatten(end_dim = 1)
        # patch embeddings in CNN
        if self.backbone == 'VAE':
            cnn_output = self.cnn(x)
            cnn_embeddings = cnn_output[2]
        else:
            cnn_embeddings = self.cnn(x)
        # decompose batch and sequence dimensions

        x = cnn_embeddings.unflatten(0, (b, s))
        # print(x.shape)
        
        # add class token 
        x = torch.cat((self.class_token.expand(b, -1, -1), x), dim=1)
        # add positional embedding
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x_ready_for_transformer = x + self.positional_embedding.to(self.device)
        x = self.transformer(x_ready_for_transformer)#.to(device))
        if combined_exposures:
            x = x[:original_batch_size] + x[original_batch_size:]
        # print(x.shape)
        x = self.norm(x)[:,0]

    # breakpoint()
        return self.fc4(x)
class DoubleTemporalTransformer(nn.Module):
    def __init__(self, transformer_layers = 4, dim = 256, num_heads = 8, seq_len = 10, ff_dim = 1024, dropout = 0.1,  channels = 3, backbone = 'ResNet', end_dimensions = 8192):
        super(DoubleTemporalTransformer, self).__init__()
      
        # cnn = ResNet()#CNN(channels, dim)
        image_size = 16
        init_dim = 16
       

        # dim= image_size
      

        dim_mults=(1, 1, 1,)
        if backbone == 'ResNet':
            # raise NotImplementedError
            cnn_1 = ResNet(dim = image_size, init_dim = init_dim, channels = channels, dim_mults  = dim_mults, last_dim  = dim)
            cnn_2 = ResNet(dim = image_size, init_dim = init_dim, channels = channels, dim_mults  = dim_mults, last_dim  = dim)

        elif backbone == 'CNN':
            cnn_1 = CNN(channels, dim)
            cnn_2 = CNN(channels, dim)

        transformer_1 = Transformer(transformer_layers, dim, num_heads, ff_dim, dropout)
        transformer_2 = Transformer(transformer_layers, dim, num_heads, ff_dim, dropout)

        self.cnn_1 = cnn_1.cuda()
        self.cnn_2 = cnn_2.cuda()

        self.transformer_1 = transformer_1.cuda()
        self.transformer_2 = transformer_2.cuda()

        self.dim = dim
        self.fc4 = nn.Linear(self.dim, end_dimensions)
        self.class_token_1 = nn.Parameter(torch.zeros(1, 1, self.dim))
        self.class_token_2 = nn.Parameter(torch.zeros(1, 1, self.dim))
        
        self.seq_len = seq_len + 1
        self.positional_embedding_1 = nn.Parameter(torch.zeros(1, self.seq_len, self.dim))
        self.positional_embedding_2 = nn.Parameter(torch.zeros(1, self.seq_len, self.dim))

        self.norm = nn.LayerNorm(self.dim, eps=1e-6)
    def embed(self, x):
        if len(x.shape) == 6:
            # All exposures are being used 
            x_2 = x[:, :, 1, :, :, :]
            x_1 = x[:, :, 0, :, :, :]
            original_batch_size = x.shape[0]
            # x = torch.concat((x[:,:,0], x[:,:,1]))
        # b, s, c, h, w = x.shape
            combined_exposures = True     
        # else:
            # combined_exposures = False       
        b, s, c, h, w = x_1.shape
        x_1 = x_1.flatten(end_dim = 1)
        x_2 = x_2.flatten(end_dim = 1)
        # patch embeddings in CNN
        cnn_embeddings_1 = self.cnn_1(x_1)
        cnn_embeddings_2 = self.cnn_2(x_2)
        # decompose batch and sequence dimensions

        x_1 = cnn_embeddings_1.unflatten(0, (b, s))
        x_2 = cnn_embeddings_2.unflatten(0, (b, s))
        # add class token 
        x_1 = torch.cat((self.class_token_1.expand(b, -1, -1), x_1), dim=1)
        x_2 = torch.cat((self.class_token_2.expand(b, -1, -1), x_2), dim=1)
        # add positional embedding
        x_ready_for_transformer_1 = x_1+ self.positional_embedding_1.cuda()
        x_1 = self.transformer_1(x_ready_for_transformer_1.to('cuda'))
        x_ready_for_transformer_2 = x_2 + self.positional_embedding_2.cuda()
        x_2 = self.transformer_2(x_ready_for_transformer_2.to('cuda'))
       
        x = x_1 + x_2
        return x
    def forward(self, x):
        # combine batch and sequence dimensions
        x = self.embed(x)
        x = self.norm(x)[:,0]
        return self.fc4(x)

class VAETemporalTransformer(TemporalTransformer):
    def __init__(self, vae_path, in_channels = 3,  *args, **kwargs):
        super(VAETemporalTransformer, self).__init__(*args, **kwargs)
        self.cnn = VanillaVAE(in_channels = in_channels, latent_dim = self.dim)
        self.cnn.load_state_dict(torch.load(vae_path)[0])
        self.cnn.eval()
        self.cnn.to(device)
        self.cnn.requires_grad = False
def main():
    temp_transformer = VAETemporalTransformer()
    breakpoint()

if __name__ == '__main__':
    main()