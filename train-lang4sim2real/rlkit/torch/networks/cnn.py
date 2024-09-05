import numpy as np
import torch
from torch import nn
import torchvision

from rlkit.torch.networks.film import FiLMBlockSequence
from rlkit.torch.networks.image_augmentations import (
    create_aug_transform_fns,
)
from rlkit.util.core import torch_ify
from rlkit.util.pythonplusplus import identity
import rlkit.util.pytorch_util as ptu


class CNN(nn.Module):
    def __init__(
            self,
            input_width,
            input_height,
            input_channels,
            output_size,
            kernel_sizes,
            n_channels,
            strides,
            paddings,
            hidden_sizes=None,
            added_fc_input_size=0,
            # latent_fc_dim=0,
            conv_normalization_type='none',
            fc_normalization_type='none',
            init_w=1e-4,
            reshape_input_on_forward=True,
            hidden_init=nn.init.xavier_uniform_,
            hidden_activation=nn.ReLU(),
            output_activation=identity,
            pool_type='none',
            pool_sizes=None,
            pool_strides=None,
            pool_paddings=None,
            aug_transforms=[],
            image_augmentation_padding=4,
            rnd_erase_prob=0.0,
            fc_dropout=0.0,
            fc_dropout_length=0,
            film_emb_dim_list=[],  # [] == don't use film.
            num_film_inputs=0,  # number of separate embeddings. There will be
            # `num_film_inputs` film blocks for each BasicBlock.
            film_hidden_sizes=[],
            film_hidden_activation="identity",
            use_film_attn=False,
    ):
        if hidden_sizes is None:
            hidden_sizes = []
        assert len(kernel_sizes) == \
               len(n_channels) == \
               len(strides) == \
               len(paddings)
        assert conv_normalization_type in {'none', 'batch', 'layer'}
        assert fc_normalization_type in {'none', 'batch', 'layer'}
        assert pool_type in {'none', 'max2d'}
        if pool_type == 'max2d':
            assert len(pool_sizes) == len(pool_strides) == len(pool_paddings)
        # assert latent_fc_dim <= added_fc_input_size
        super().__init__()

        self.hidden_sizes = hidden_sizes
        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        self.output_size = output_size
        self.output_activation = output_activation
        self.hidden_activation = hidden_activation
        self.conv_normalization_type = conv_normalization_type
        self.fc_normalization_type = fc_normalization_type
        self.added_fc_input_size = added_fc_input_size
        # self.latent_fc_dim = latent_fc_dim
        self.conv_input_length = (
            self.input_width * self.input_height * self.input_channels)
        self.pool_type = pool_type
        self.aug_transforms = aug_transforms
        self.im_aug_pad = image_augmentation_padding
        self.rnd_erase_prob = rnd_erase_prob
        self.fc_dropout = fc_dropout
        self.fc_dropout_length = fc_dropout_length
        self.init_w = init_w
        self.reshape_input_on_forward = reshape_input_on_forward
        self.film_emb_dim_list = film_emb_dim_list
        self.num_film_inputs = num_film_inputs
        self.film_hidden_sizes = film_hidden_sizes
        self.film_hidden_activation = film_hidden_activation

        self.conv_layers = nn.ModuleList()
        self.conv_norm_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()
        self.fc_norm_layers = nn.ModuleList()
        self.film_block_seqs = nn.ModuleList()

        for i, (out_channels, kernel_size, stride, padding) in enumerate(
                zip(n_channels, kernel_sizes, strides, paddings)
        ):
            conv = nn.Conv2d(input_channels,
                             out_channels,
                             kernel_size,
                             stride=stride,
                             padding=padding)
            hidden_init(conv.weight)
            conv.bias.data.fill_(0)

            conv_layer = conv
            self.conv_layers.append(conv_layer)

            if len(film_emb_dim_list) > 0:
                if use_film_attn:
                    num_film_blocks = 1
                else:
                    num_film_blocks = self.num_film_inputs
                film_blocks = FiLMBlockSequence(
                    self.film_emb_dim_list, num_film_blocks, out_channels,
                    self.film_hidden_sizes, self.film_hidden_activation)
                self.film_block_seqs.append(film_blocks)

            input_channels = out_channels

            if pool_type == 'max2d':
                if pool_sizes[i] > 1:
                    self.pool_layers.append(
                        nn.MaxPool2d(
                            kernel_size=pool_sizes[i],
                            stride=pool_strides[i],
                            padding=pool_paddings[i],
                        )
                    )

        # use torch rather than ptu because initially the model is on CPU
        test_mat = torch.zeros(
            1,
            self.input_channels,
            self.input_height,
            self.input_width,
        )
        # find output dim of conv_layers by trial and add norm conv layers
        for i, conv_layer in enumerate(self.conv_layers):
            test_mat = conv_layer(test_mat)
            if self.conv_normalization_type == 'batch':
                self.conv_norm_layers.append(nn.BatchNorm2d(test_mat.shape[1]))
            if self.conv_normalization_type == 'layer':
                self.conv_norm_layers.append(nn.LayerNorm(test_mat.shape[1:]))
            if self.pool_type != 'none' and len(self.pool_layers) > i:
                test_mat = self.pool_layers[i](test_mat)

        self.conv_output_flat_size = int(np.prod(test_mat.shape))

        self.fc_layers, self.fc_norm_layers, self.last_fc = (
            ptu.initialize_fc_layers(
                self.hidden_sizes, self.output_size,
                self.fc_normalization_type, self.conv_output_flat_size,
                self.added_fc_input_size, init_w))

        transf_kwargs = {
            "image_size": (input_height, input_width, input_channels),
            "im_aug_pad": self.im_aug_pad,
            "rnd_erase_prob": self.rnd_erase_prob,
            "aug_transforms": self.aug_transforms,
        }
        self.aug_transform_fns = create_aug_transform_fns(transf_kwargs)

        if self.fc_dropout > 0.0:
            self.fc_dropout_layer = nn.Dropout(self.fc_dropout)

    def forward(
            self, input, output_stage="", train_mode=True, film_inputs=None):
        if self.reshape_input_on_forward:
            conv_input = input.narrow(start=0,
                                      length=self.conv_input_length,
                                      dim=1).contiguous()
            # reshape from batch of flattened images into (channels, w, h)
            h = conv_input.view(conv_input.shape[0],
                                self.input_channels,
                                self.input_height,
                                self.input_width)
        else:
            h = input

        if h.shape[0] > 1 and train_mode:
            for aug_transform_fn in self.aug_transform_fns:
                h = aug_transform_fn(h)

        h = self.apply_forward_conv(h, film_inputs=film_inputs)

        if output_stage == "conv_channels":
            return h

        # flatten channels for fc layers
        h = h.reshape(h.size(0), -1)
        if self.added_fc_input_size != 0:
            extra_fc_input = input.narrow(
                start=self.conv_input_length,
                length=self.added_fc_input_size,
                dim=1,
            )
            h = torch.cat((extra_fc_input, h), dim=+1)
        h = self.apply_forward_fc(h)
        if output_stage == "last_activations":
            return h
        return self.output_activation(self.last_fc(h))

    def apply_forward_conv(self, h, film_inputs):
        for i, layer in enumerate(self.conv_layers):
            h = layer(h)
            if self.conv_normalization_type != 'none':
                h = self.conv_norm_layers[i](h)
            if self.pool_type != 'none' and len(self.pool_layers) > i:
                h = self.pool_layers[i](h)
            if film_inputs is not None:
                film_inputs = torch_ify(film_inputs)
                h = self.film_block_seqs[i](h, film_inputs)
            h = self.hidden_activation(h)
        return h

    def apply_forward_fc(self, h):
        if self.fc_dropout > 0.0 and self.fc_dropout_length > 0:
            dropout_input = h.narrow(
                start=0,
                length=self.fc_dropout_length,
                dim=1,
            )
            dropout_output = self.fc_dropout_layer(dropout_input)

            remaining_input = h.narrow(
                start=self.fc_dropout_length,
                length=(
                    self.conv_output_flat_size + self.added_fc_input_size
                    - self.fc_dropout_length),
                dim=1)
            h = torch.cat((dropout_output, remaining_input), dim=1)

        for i, layer in enumerate(self.fc_layers):
            h = layer(h)
            if self.fc_normalization_type != 'none':
                h = self.fc_norm_layers[i](h)
            h = self.hidden_activation(h)
        return h


class ConcatCNN(CNN):
    """
    Concatenate inputs along dimension and then pass through MLP.
    """
    def __init__(self, *args, dim=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=self.dim)
        return super().forward(flat_inputs, **kwargs)


class StandaloneCNN(nn.Module):
    """ A CNN class where the self.cnn is initialized already,
    but with the option to add extra layers at the end."""
    def __init__(
            self,
            cnn,
            input_width,
            input_height,
            input_channels,
            hidden_sizes,
            output_size,
            freeze_cnn,
            added_fc_input_size=0,
            init_w=1e-3,
            hidden_activation=nn.ReLU(),
            aug_transforms=[],
            image_augmentation_padding=4,
            rnd_erase_prob=0.0,
            fc_normalization_type='none',
            output_activation=torch.tanh,
            batchnorm=False,
            cnn_output_stage="",
            reshape_input_on_forward=True,
            **kwargs,  # Will most likely be unused
    ):
        super(StandaloneCNN, self).__init__()
        self.cnn = cnn
        self.image_size = (input_height, input_width, input_channels)
        self.freeze_cnn = freeze_cnn
        self.conv_input_length = np.prod(self.image_size)
        self.added_fc_input_size = added_fc_input_size
        self.hidden_activation = hidden_activation
        self.aug_transforms = aug_transforms
        self.im_aug_pad = image_augmentation_padding
        self.rnd_erase_prob = rnd_erase_prob
        self.fc_normalization_type = fc_normalization_type
        self.output_activation = output_activation
        self.batchnorm = batchnorm
        self.cnn_output_stage = cnn_output_stage
        self.reshape_input_on_forward = reshape_input_on_forward
        self.num_film_inputs = 0  # FiLM not supported

        test_mat = torch.zeros(
            1,
            input_channels,
            input_height,
            input_width,
        )

        from r3m.models.models_r3m import R3M
        self.forward_kwargs = {"output_stage": self.cnn_output_stage}
        if isinstance(self.cnn.module, R3M):
            self.forward_kwargs = {}
            assert self.cnn_output_stage == ""
        test_mat = self.cnn(test_mat, **self.forward_kwargs)
        cnn_out_dim = np.prod(test_mat.shape[1:])
        print("cnn_out_dim", cnn_out_dim)

        if self.batchnorm:
            self.bn_layer = nn.BatchNorm1d(
                cnn_out_dim + self.added_fc_input_size)

        self.fc_layers, self.fc_norm_layers, self.last_fc = (
            ptu.initialize_fc_layers(
                hidden_sizes, output_size,
                self.fc_normalization_type, cnn_out_dim,
                self.added_fc_input_size, init_w))

        transf_kwargs = {
            "image_size": (input_height, input_width, input_channels),
            "im_aug_pad": self.im_aug_pad,
            "rnd_erase_prob": self.rnd_erase_prob,
            "aug_transforms": self.aug_transforms,
        }
        self.aug_transform_fns = create_aug_transform_fns(transf_kwargs)

        if self.freeze_cnn:
            self.freeze_cnn_params()

    def forward(self, input_obs, train_mode=True):
        if self.reshape_input_on_forward:
            conv_input = input_obs.narrow(
                start=0,
                length=self.conv_input_length,
                dim=1).contiguous()

            # (B, 6912) --> (B, 3, 48, 48)
            B, _ = conv_input.shape
            H, W, C = self.image_size
            conv_input = torch.reshape(conv_input, (B, C, H, W))
        else:
            conv_input = input_obs

        if train_mode:
            # h.shape[0] > 1 ensures we apply this only during training
            for aug_transform_fn in self.aug_transform_fns:
                conv_input = aug_transform_fn(conv_input)

        with torch.set_grad_enabled(not self.freeze_cnn):
            h = self.cnn(conv_input, **self.forward_kwargs)

        if len(h.shape) > 2:
            h = torch.flatten(h, start_dim=1)

        if self.added_fc_input_size != 0:
            extra_fc_input = input_obs.narrow(
                start=self.conv_input_length,
                length=self.added_fc_input_size,
                dim=1,
            )
            h = torch.cat((extra_fc_input, h), dim=+1)

            if h.shape[0] > 1 and self.batchnorm:
                # Only during training
                h = self.bn_layer(h)

        h = self.apply_forward_fc(h)
        h = self.last_fc(h)
        h = self.output_activation(h)
        return h

    def apply_forward_fc(self, h):
        # abridged version of CNN.apply_forward_fc
        for i, layer in enumerate(self.fc_layers):
            h = layer(h)
            if self.fc_normalization_type != 'none':
                h = self.fc_norm_layers[i](h)
            h = self.hidden_activation(h)
        return h

    def freeze_cnn_params(self):
        for param in self.cnn.parameters():
            param.requires_grad = False


class ClipWrapper(nn.Module):
    def __init__(
            self, clip_checkpoint, freeze, image_shape, tokenize_scheme,
            image_augmentation_padding, task_lang_list,
            aug_transforms=[], rnd_erase_prob=0.0):
        super().__init__()
        self.freeze = freeze
        self.load_clip_fns(
            clip_checkpoint, tokenize_scheme, freeze_clip=freeze, gpu=0)
        self.visual_outdim = self.clip.module.embed_dim
        self.lang_outdim = self.clip.module.embed_dim
        self.image_shape = image_shape
        self.task_lang_list = task_lang_list
        self.task_lang_tokens_matrix = self.create_task_id_to_lang_tokens_map()
        self.num_film_inputs = 0  # FiLM not supported

        self.aug_transforms = aug_transforms
        if len(self.aug_transforms) > 1:
            transf_kwargs = {
                "image_size": image_shape,
                "im_aug_pad": image_augmentation_padding,
                "rnd_erase_prob": rnd_erase_prob,
                "aug_transforms": self.aug_transforms,
            }
            self.aug_transform_fns = create_aug_transform_fns(transf_kwargs)
        else:
            self.aug_transform_fns = []

    def create_task_id_to_lang_tokens_map(self):
        if len(self.task_lang_list) == 0:
            return torch.tensor([])
        elif isinstance(self.task_lang_list[0], str):
            return self.tokenize_fn(self.task_lang_list)
        elif isinstance(self.task_lang_list[0], list):
            assert all([len(x) == 1 for x in self.task_lang_list])
            task_lang_tokens_matrix = torch.cat([
                self.tokenize_fn(tl) for tl in self.task_lang_list])
            return task_lang_tokens_matrix
        else:
            raise NotImplementedError

    def forward(self, images, texts, train_mode=True):
        """Allow images to potentially == None"""
        if images is not None:
            if train_mode:
                for aug_transform_fn in self.aug_transform_fns:
                    images = aug_transform_fn(images)
            images = self.preprocess(images)
        if texts is not None:
            texts = texts.long().cuda()

        with torch.set_grad_enabled(not self.freeze and train_mode):
            # image_features, text_features are already normalized
            # The logits used for the contrastive loss for CLIP are
            # logit_scale * image @ text

            if texts is None or images is None:
                return self.clip(images, texts)  # only the image/text features
            else:
                # returns image_features, text_features, logit_scale
                return self.clip(images, texts)

    def load_clip_fns(
            self, clip_checkpoint, tokenize_scheme, freeze_clip, gpu=0):
        from heatmaps_clip import load_model_preprocess
        from clip import clip
        self.clip, self.preprocess = load_model_preprocess(
            clip_checkpoint, gpu=gpu, freeze_clip=freeze_clip,
            input_type="tensor")
        self.tokenize_fn = clip.get_tokenize_fn(tokenize_scheme)

    def get_task_lang_tokens_matrix(self, task_idx_list):
        if isinstance(task_idx_list, list) or isinstance(
                task_idx_list, np.ndarray):
            pass
        elif torch.is_tensor(task_idx_list):
            task_idx_list = np.array(task_idx_list.cpu())
        else:
            raise NotImplementedError

        return self.task_lang_tokens_matrix[task_idx_list]


class R3MWrapper(nn.Module):
    def __init__(
            self, out_dim=384, device=torch.device("cuda"),
            strict_param_load=True, adapter_kwargs={}):
        super().__init__()
        from r3m import load_r3m
        self.r3m = load_r3m("resnet18", strict=strict_param_load)
        # self.r3m = self.r3m.eval()
        self.fc_layers = torch.nn.Sequential(
            torch.nn.Linear(512, out_dim)
        ).to(device)
        if len(adapter_kwargs) > 0:
            self.add_conv_adapters(adapter_kwargs)
        self.r3m.to(device)

    def forward(self, img):
        cnn_embs = self.r3m(
            img, obs_shape=torch.tensor(img.shape[-3:]).tolist())
        out = self.fc_layers(cnn_embs)
        return out

    def add_conv_adapters(self, adapter_kwargs):
        return self.r3m.module.convnet.add_conv_adapters(adapter_kwargs)


class ImgNetWrapper(nn.Module):
    def __init__(self, out_dim=384, device=torch.device("cuda")):
        super().__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True).to(device)
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.Normalize(
                (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
        )
        self.fc_layers = torch.nn.Sequential(
            torch.nn.Linear(1000, out_dim)
        ).to(device)

    def forward(self, img, output_stage=""):
        if output_stage != "":
            print("output_stage", output_stage)
            raise NotImplementedError
        img = self.transform(img)

        out = self.resnet(img)
        out = self.fc_layers(out)
        return out

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_last_layer(self):
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True
        for param in self.fc_layers.parameters():
            param.requires_grad = True

    def unfreeze_all_layers(self):
        for param in self.parameters():
            param.requires_grad = True
