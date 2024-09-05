import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from distutils.version import LooseVersion as LV
import os
import h5py
import io
import numpy as np
from PIL import Image

torch.manual_seed(42)
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print('Using PyTorch version:', torch.__version__, ' Device:', device)
assert(LV(torch.__version__) >= LV("1.0.0"))

SAMPLE_SIZE = 50

def get_tensorboard(log_name):
    try:
        import tensorboardX
        import os
        import datetime
        logdir = os.path.join(os.getcwd(), "logs",
                              "classify" + log_name + "-" +
                              datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        print('Logging TensorBoard to:', logdir)
        os.makedirs(logdir)
        return tensorboardX.SummaryWriter(logdir)
    except (ImportError, FileExistsError):
        return None


def train(model, loader, criterion, optimizer, epoch, log=None):
    # Set model to training mode
    model.train()
    epoch_loss = 0.
    num_correct = 0

    # Loop over each batch from the training set
    for batch_idx, (data, target) in enumerate(loader):
        # Copy data to GPU if needed
        data = data.to(device).float()
        target = target.to(device).squeeze() # (bsz,)

        # Zero gradient buffers
        optimizer.zero_grad()

        # Pass data through the network
        output = model(data)
        # output = torch.squeeze(output.to(torch.float32))

        # Calculate loss
        loss = criterion(output, target)
        epoch_loss += (data.shape[0] / len(loader.dataset)) * loss.data.item()

        # Backpropagate
        loss.backward()

        # Update weights
        optimizer.step()

        # Compute training accuracy
        pred_labels = torch.argmax(output.detach(), axis=1)
        num_correct += (pred_labels == target).cpu().sum()

    accuracy = 100. * num_correct / len(loader.dataset)

    print('\nEpoch: {}\nTrain Loss: {:.4f}, Train Accuracy: {}/{} ({:.2f}%)'.format(
        epoch, epoch_loss, num_correct, len(loader.dataset), accuracy))

    if log is not None:
        log.add_scalar('loss', epoch_loss, epoch-1)


def evaluate(model, loader, criterion=None, epoch=None, log=None):
    model.eval()
    loss, num_correct = 0, 0
    for data, target in loader:
        data = data.to(device).float()
        target = target.to(device).squeeze()

        output = model(data)

        if criterion is not None:
            loss += (data.shape[0] / len(loader.dataset)) * (
                    criterion(output, target).data.item())

        pred_labels = torch.argmax(output, axis=1)
        num_correct += (pred_labels == target).cpu().sum()

    accuracy = 100. * num_correct / len(loader.dataset)
    accuracy = accuracy.cpu().item()

    print('Val loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        loss, num_correct, len(loader.dataset), accuracy))

    if log is not None and epoch is not None:
        log.add_scalar('val_loss', loss, epoch-1)
        log.add_scalar('val_acc', accuracy, epoch-1)

    return {"val_loss": loss, "val_acc": accuracy}


class HDF5Dataset(Dataset):
    def __init__(self, file_path, dataset_name, transform=None):
        self.file_path = file_path
        self.dataset_name = dataset_name
        self.transform = transform
        self.trajNames = []
        self.length = None
        self._idx_to_name = {} # Maps dataset traj index to trajName (like "demo_0")
        self._open_hdf5()

        self.length = len(self._hf['mask'][dataset_name])
        for trajName in self._hf['mask'][dataset_name]:
            self.trajNames.append(trajName.decode("utf-8"))

        i = 0
        for _, dd in enumerate(self._hf['data'].items()):
            if dd[0] in self.trajNames:
                self._idx_to_name[i] = dd[0]
                i += 1

    def __len__(self):
        assert self.length is not None
        return self.length

    def _open_hdf5(self):
        self._hf = h5py.File(self.file_path, 'r')

    def __getitem__(self, index):
        assert self._idx_to_name is not None
        traj_name = self._idx_to_name[index]

        ds = self._hf['data'][traj_name]
        x = np.array(ds['obs']['object'])
        verb_start_idx, verb_end_idx = 0, 2
        verb_onehot = np.array(ds['obs']['task_id'])[verb_start_idx:verb_end_idx]
        y = np.array([np.argmax(verb_onehot)]) # convert int to (1,) array.

        if self.transform:
            x = self.transform(x)
        else:
            idx = np.round(np.linspace(0, len(x) - 1, SAMPLE_SIZE)).astype(int)
            x = x[idx]
            x = torch.from_numpy(x)

        y = torch.from_numpy(y)
        return (x, y)


def get_loader_hdf5(split, path, batch_size):
    assert split in ["train", "valid"]
    shuffle = split == "train"
    print(f"{split}: ", end="")
    dataset = HDF5Dataset(path, split, transform=None)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=shuffle, num_workers=10)
    print('Found', len(dataset), 'trajectories')
    return loader


def get_train_loader_hdf5(path, batch_size):
    return get_loader_hdf5("train", path, batch_size)


def get_validation_loader_hdf5(path, batch_size):
    return get_loader_hdf5("valid", path, batch_size)


if __name__ == '__main__':
    print('\nThis Python script is only for common functions. *DON\'T RUN IT DIRECTLY!* :-)')
