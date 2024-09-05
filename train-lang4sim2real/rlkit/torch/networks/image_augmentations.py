import numpy as np
import torch


class RandomCrop:
    """
    Source: # https://github.com/pratogab/batch-transforms
    Applies the :class:`~torchvision.transforms.RandomCrop` transform to
    a batch of images.
    Args:
        size (int): Desired output size of the crop.
        padding (int, optional): Optional padding on each border of the image.
            Default is None, i.e no padding.
        dtype (torch.dtype,optional): The data type of tensors to which
            the transform will be applied.
        device (torch.device,optional): The device of tensors to which
            the transform will be applied.
    """

    def __init__(self, size, padding=None, dtype=torch.float, device='cpu'):
        self.size = size
        self.padding = padding
        self.dtype = dtype
        self.device = device
        self.i = 1  # For debugging purposes

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be cropped.
        Returns:
            Tensor: Randomly cropped Tensor.
        """
        if self.padding is not None:
            padded = torch.zeros((tensor.size(0), tensor.size(1),
                                  tensor.size(2) + self.padding * 2,
                                  tensor.size(3) + self.padding * 2),
                                 dtype=self.dtype, device=self.device)
            padded[:, :, self.padding:-self.padding,
                   self.padding:-self.padding] = tensor
        else:
            padded = tensor

        h, w = padded.size(2), padded.size(3)
        th, tw = self.size
        if w == tw and h == th:
            i, j = 0, 0
        else:
            i = torch.randint(0, h - th + 1, (tensor.size(0),),
                              device=self.device)
            j = torch.randint(0, w - tw + 1, (tensor.size(0),),
                              device=self.device)

        rows = (torch.arange(th, dtype=torch.long, device=self.device)
                + i[:, None])
        columns = (torch.arange(tw, dtype=torch.long, device=self.device)
                   + j[:, None])
        padded = padded.permute(1, 0, 2, 3)
        padded = padded[:, torch.arange(tensor.size(0))[:, None, None],
                        rows[:, torch.arange(th)[:, None]], columns[:, None]]
        padded = padded.permute(1, 0, 2, 3)
        return padded


def torch_rand_unif(lo, hi, size, device):
    return (hi - lo) * torch.rand(*size, device=device) + lo


class BrightnessContrast:
    """Batch version of BrightnessContrast"""
    def __init__(self, brightness, contrast):
        self.brightness = brightness
        self.contrast = contrast

    def torch_rand_unif(self, lo, hi, size, device):
        return (hi - lo) * torch.rand(*size, device=device) + lo

    def __call__(self, tensor):
        assert tensor.dtype == torch.float32
        assert len(tensor.shape) == 4
        assert tensor.shape[1] == 3  # RGB
        b, *_ = tensor.size()
        alpha_vec = 1 + torch_rand_unif(
            -self.contrast, self.contrast,
            size=(b, 1, 1, 1), device=tensor.device)
        # One alpha val per image.
        beta_vec = torch_rand_unif(
            -self.brightness, self.brightness,
            size=(b, 1, 1, 1), device=tensor.device)
        tensor = alpha_vec * tensor + beta_vec
        tensor = torch.clamp(tensor, min=0.0, max=1.0)
        return tensor


class RGBShift:
    """Batch version of RGBShift"""
    def __init__(self, max_shift=0.2):
        self.r_max_shift = max_shift
        self.g_max_shift = max_shift
        self.b_max_shift = max_shift

    def __call__(self, tensor):
        assert tensor.dtype == torch.float32
        assert len(tensor.shape) == 4
        assert tensor.shape[1] == 3  # RGB
        b, *_ = tensor.size()
        r_shifts = torch_rand_unif(
            -self.r_max_shift, self.r_max_shift,
            size=(b, 1, 1, 1), device=tensor.device)
        g_shifts = torch_rand_unif(
            -self.g_max_shift, self.g_max_shift,
            size=(b, 1, 1, 1), device=tensor.device)
        b_shifts = torch_rand_unif(
            -self.b_max_shift, self.b_max_shift,
            size=(b, 1, 1, 1), device=tensor.device)
        rgb_shift = torch.cat([r_shifts, g_shifts, b_shifts], dim=1)
        # size: (b, 3, 1, 1)
        tensor = tensor + rgb_shift
        tensor = torch.clamp(tensor, min=0.0, max=1.0)
        return tensor


class RandomErasing:
    """
    Batch version of RandomErase. First picks a valid cutout size,
    then puts that cutout size on each image in batch at random x,y offsets.
    """
    def __init__(self, image_size, scale_range, ratio_range, rnd_erase_prob):
        self.im_height = image_size[0]
        self.im_width = image_size[1]
        self.scale_range = scale_range  # May be cutoff the edge of an image.
        self.ratio_range = ratio_range  # aspect ratio range, aka width/height
        assert len(self.scale_range) == 2
        assert len(self.ratio_range) == 2
        self.rnd_erase_prob = rnd_erase_prob
        self.valid_cutout_sizes = self.get_valid_rects()  # [(h, w), ...]

    def get_valid_rects(self):
        # Calc (min_h, max_h), (min_w, max_w)
        # iterate through.
        im_area = self.im_height * self.im_width
        min_h = (
            (1 / self.ratio_range[1]) * self.scale_range[0] * im_area) ** 0.5
        max_h = (
            (1 / self.ratio_range[0]) * self.scale_range[1] * im_area) ** 0.5

        min_w = min_h
        max_w = max_h
        min_h, max_h = int(np.floor(min_h)), int(np.ceil(max_h))
        min_w, max_w = int(np.floor(min_w)), int(np.ceil(max_w))

        valid_cutout_sizes = []
        for h in range(min_h, max_h):
            for w in range(min_w, max_w):
                scale_constr_sat = (
                    self.scale_range[0]
                    <= (h * w) / im_area
                    <= self.scale_range[1])
                im_dim_sat = (h <= self.im_height) and (w <= self.im_width)
                ratio_constr_sat = (
                    self.ratio_range[0] <= (w/h) <= self.ratio_range[1])
                if scale_constr_sat and im_dim_sat and ratio_constr_sat:
                    valid_cutout_sizes.append((h, w))
        return valid_cutout_sizes

    def __call__(self, x):
        assert x.dtype == torch.float32
        assert len(x.shape) == 4
        assert x.shape[1:] == (3, self.im_height, self.im_width)
        b, *_ = x.shape

        erasing_coinflips = torch.tensor(np.random.choice(
            [0, 1], size=(b,),
            p=[self.rnd_erase_prob, 1 - self.rnd_erase_prob]))
        # 0 = erase for that image; 1 = no erase.
        num_to_erase = int(torch.sum(1 - erasing_coinflips))
        if num_to_erase == 0:
            return x

        # Randomly choose a cutout size for the whole batch
        cutout_size_idx = np.random.randint(len(self.valid_cutout_sizes))
        cutout_size = self.valid_cutout_sizes[cutout_size_idx]

        # Below lines are based on:
        # https://github.com/mit-han-lab/data-efficient-gans/blob/master/DiffAugment_pytorch.py#L55
        indices_to_not_erase = torch.where(erasing_coinflips > 0)[0]
        grid_batch, grid_x, grid_y = torch.meshgrid(
            torch.arange(x.size(0), dtype=torch.long, device=x.device),
            torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
            torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
        )
        offset_x = torch.randint(
            0, x.size(2) + (1 - cutout_size[0] % 2),
            size=[x.size(0), 1, 1], device=x.device)
        offset_y = torch.randint(
            0, x.size(3) + (1 - cutout_size[1] % 2),
            size=[x.size(0), 1, 1], device=x.device)
        grid_x = torch.clamp(
            grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
        grid_y = torch.clamp(
            grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
        mask = torch.ones(
            x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
        mask[grid_batch, grid_x, grid_y] = 0

        # Make some masks back to 1's.
        if len(indices_to_not_erase) > 0:
            mask[indices_to_not_erase] = 1.0

        x = x * mask.unsqueeze(1)
        return x


def create_aug_transform_fns(transf_kwargs):
    rnd_crop = RandomCrop(
        (transf_kwargs['image_size'][0], transf_kwargs['image_size'][1]),
        # (h, w)
        transf_kwargs['im_aug_pad'], device='cuda',
    )
    # Pytorch versions for ColorJitter and RandomErasing
    # don't apply diff transforms for images within the same batch.
    bright_contr = BrightnessContrast(
        brightness=0.4, contrast=0.5)
    rgb_shift = RGBShift(max_shift=0.3)
    rnd_erase = RandomErasing(
        image_size=transf_kwargs['image_size'],
        scale_range=(0.1, 0.3),
        ratio_range=(0.25, 4.0),
        rnd_erase_prob=transf_kwargs['rnd_erase_prob'],
    )
    aug_to_fn_map = {
        "pad_crop": rnd_crop,
        "bright_contr": bright_contr,
        "rgb_shift": rgb_shift,
        "erase": rnd_erase,
    }
    assert set(transf_kwargs['aug_transforms']).issubset(
        set(aug_to_fn_map.keys()))
    aug_transform_fns = []
    for aug_transform in transf_kwargs['aug_transforms']:
        aug_transform_fn = aug_to_fn_map[aug_transform]
        aug_transform_fns.append(aug_transform_fn)
    return aug_transform_fns
