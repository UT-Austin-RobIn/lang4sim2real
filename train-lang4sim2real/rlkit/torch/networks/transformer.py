import numpy as np
import torch
from torch import nn

from rlkit.torch.networks.resnet import ResNet, SpatialSoftmax

"""
Classes based on
https://github.com/Mathux/ACTOR/blob/master/src/models/architectures/transformer.py#L7-L24
"""


class PosEncoding(nn.Module):
    def __init__(self, latent_dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, latent_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, latent_dim, 2).float()
            * (-np.log(10000.0) / latent_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.shape[0], :]
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        in_dim,
        latent_dim,
        num_heads,
        ff_size,
        num_layers,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.ff_size = ff_size
        self.num_layers = num_layers

        # Define modules
        EncoderLayer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=0.0,
            activation="relu",
            batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            EncoderLayer, num_layers=self.num_layers)
        self.emb_act_ff = nn.Linear(self.in_dim, self.latent_dim)
        self.add_pos_embs = PosEncoding(self.latent_dim)
        self.mu_fc = nn.Linear(self.latent_dim, self.latent_dim)
        self.logvar = nn.Linear(self.latent_dim, self.latent_dim)

    def forward(self, batch_dict):
        """
        Assuming x: (bsz, seq, in_dim)
        """
        x = batch_dict['in_seq']
        x = self.emb_act_ff(x)
        x = self.add_pos_embs(x)
        x = self.transformer_encoder(x)

        # TODO: calculate the mu and var potentially directly as model outputs?
        # z = x.mean(axis=1)
        # mu = self.mu_fc(z)
        # logvar = self.logvar_fc(z)
        mu = x[:, 0]
        logvar = x[:, 1]
        std = torch.exp(logvar) ** 0.5
        eps = torch.randn_like(std)
        z = mu + 0.01 * eps * std
        out_dict = dict(
            encoder_embs=x,
            z=z,
            mu=mu,
            logvar=logvar,
            std=std,
            eps=eps,
        )
        if "img" in batch_dict:
            out_dict["img"] = batch_dict["img"]
        batch_dict.update(out_dict)
        return out_dict


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        latent_dim,
        out_dim,
        num_heads,
        ff_size,
        num_layers,
        subtraj_len,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.subtraj_len = subtraj_len

        # Define modules
        DecoderLayer = nn.TransformerDecoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=0.0,
            activation="relu",
            batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(
            DecoderLayer, num_layers=self.num_layers)
        self.add_pos_embs = PosEncoding(self.latent_dim)
        self.final_fc = nn.Linear(self.latent_dim, self.out_dim)

    def forward(self, batch_dict):
        z = batch_dict['z']
        bsz = z.shape[0]
        if 'in_img_embs' not in batch_dict:
            x = torch.zeros(
                bsz, self.subtraj_len, self.latent_dim, device=z.device)
        else:
            x = batch_dict['in_img_embs']
        seq = self.transformer_decoder(tgt=x, memory=z[:, None])
        seq = self.final_fc(seq)
        out_dict = dict(
            out_seq=seq,
        )
        batch_dict.update(out_dict)
        return batch_dict


class TransformerVAE(nn.Module):
    def __init__(
        self,
        in_dim,
        latent_dim,
        num_heads,
        ff_size,
        num_layers,
        subtraj_len,
        img_enc_params={},
    ):
        super().__init__()
        self.encoder = TransformerEncoder(
            in_dim=in_dim,
            latent_dim=latent_dim,
            num_heads=num_heads,
            ff_size=ff_size,
            num_layers=num_layers,
        )
        self.decoder = TransformerDecoder(
            latent_dim=latent_dim,
            out_dim=in_dim,
            num_heads=num_heads,
            ff_size=ff_size,
            num_layers=num_layers,
            subtraj_len=subtraj_len,
        )
        self.img_enc_params = img_enc_params
        if self.img_enc_params:
            device = torch.device('cuda')
            cnn_params = dict()
            cnn_params.update(dict(
                fc_layers=[],
                conv_strides=[2, 2, 1, 1, 1],
                layers=[2, 2, 2, 2],
                num_channels=self.img_enc_params['num_channels'],
                maxpool_stride=2,
                film_emb_dim_list=[],  # [386],
                num_film_inputs=0,  # 1,
                film_hidden_sizes=[],
                film_hidden_activation="identity",
                use_film_attn=False,
            ))
            self.img_enc = ResNet(**cnn_params).to(device)

            test_mat = torch.zeros(1, 3, 128, 128).to(device)
            with torch.no_grad():
                test_mat = self.img_enc(test_mat, output_stage="conv_channels")
            print("test_mat.shape", test_mat.shape)
            self.spatial_softmax = SpatialSoftmax(
                test_mat.shape[2], test_mat.shape[3], test_mat.shape[1])
            self.img_enc_out_dim = 2 * test_mat.shape[1]
            print("self.cnn_out_dim (after potential spatial softmax)",
                  self.img_enc_out_dim)

    def forward(self, batch_dict):
        batch_dict = self.encoder(batch_dict)
        if self.img_enc_params:
            # reshape img batch
            bsz = batch_dict['img'].shape[0]
            batch_dict['img'] = torch.cat(list(batch_dict['img']), dim=0)
            img_embs = self.img_enc(
                batch_dict['img'], output_stage="conv_channels")
            img_embs = self.spatial_softmax(img_embs)
            # pass in as input to decoder
            batch_dict['in_img_embs'] = img_embs.reshape(
                bsz, img_embs.shape[0] // bsz, *img_embs.shape[1:])
        batch_dict = self.decoder(batch_dict)
        return batch_dict
