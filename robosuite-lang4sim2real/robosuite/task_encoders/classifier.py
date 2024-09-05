import argparse
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime

from robosuite.task_encoders.loader import get_train_loader_hdf5, get_validation_loader_hdf5
from robosuite.task_encoders.loader import device, train, evaluate, get_tensorboard

model_file_wo_ext = 'f_verb'
BATCH_SIZE = 128

def l2_unit_normalize(x):
    return F.normalize(x, p=2, dim=-1)

class TrajMLPClassifier(nn.Module):
    def __init__(self, out_dim, token_dim, sample_size, fc_layer_dims, final_activ="logsoftmax"):
        self.token_dim = token_dim
        self.sample_size = sample_size
        super(TrajMLPClassifier, self).__init__()

        self.fc_layers = [] # everything except last FC layer
        self.fc_dropouts = []
        last_dim = token_dim*sample_size
        for fc_layer_dim in fc_layer_dims:
            fc_layer = nn.Linear(last_dim, fc_layer_dim)
            dropout_layer = nn.Dropout(0.2)
            self.fc_layers.append(fc_layer)
            self.fc_dropouts.append(dropout_layer)
            last_dim = fc_layer_dim
        self.fc_layers = nn.ModuleList(self.fc_layers)
        self.fc_dropouts = nn.ModuleList(self.fc_dropouts)

        # Last layer
        self.last_fc_layer = nn.Linear(last_dim, out_dim)

        # self.fc1 = nn.Linear(token_dim*sample_size, 256)
        # self.fc1_drop = nn.Dropout(0.2)
        # self.fc2 = nn.Linear(256, 128)
        # self.fc2_drop = nn.Dropout(0.2)
        # self.fc3 = nn.Linear(128, 64)
        # self.fc3_drop = nn.Dropout(0.2)
        # self.fc4 = nn.Linear(64, output_dim)
        # self.softmax = nn.LogSoftmax(dim=-1)
        assert final_activ in ["logsoftmax", "l2norm"]
        if final_activ == "logsoftmax":
            self.final_activ = nn.LogSoftmax(dim=-1)
        elif final_activ == "l2norm":
            self.final_activ = l2_unit_normalize

    def forward(self, x):
        x = x.view(-1, self.token_dim * self.sample_size)
        for i in range(len(self.fc_layers)):
            x = F.relu(self.fc_layers[i](x))
            x = self.fc_dropouts[i](x)
        x = self.final_activ(self.last_fc_layer(x))
        # x = F.relu(self.fc1(x))
        # x = self.fc1_drop(x)
        # x = F.relu(self.fc2(x))
        # x = self.fc2_drop(x)
        # x = F.relu(self.fc3(x))
        # x = self.fc3_drop(x)
        # x = self.final_activ(self.fc4(x))
        return x


def train_main(args):
    model = TrajMLPClassifier(**args.model_kwargs).to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=0.05)
    criterion = nn.CrossEntropyLoss()

    print(model)

    train_loader = get_train_loader_hdf5(args.dataset, BATCH_SIZE)
    validation_loader = get_validation_loader_hdf5(args.dataset, BATCH_SIZE)

    log = get_tensorboard('simple')
    epochs = 30

    warmup_epochs = 1
    tot_time = 0

    val_metric_history_dict = {"val_loss": [np.inf], "val_acc": [0.0]}
    val_metric_dict = {}
    model_fpaths = []

    for epoch in range(1, epochs + 1):
        start_time = datetime.now()
        train(model, train_loader, criterion, optimizer, epoch, log)

        # Calc val metrics
        with torch.no_grad():
            val_metric_dict = evaluate(model, validation_loader, criterion, epoch, log)
            val_loss = val_metric_dict['val_loss']
            val_acc = val_metric_dict['val_acc']

        # If val metrics look good, save a checkpoint.
        if (args.save_ckpts and
                (val_loss < min(val_metric_history_dict['val_loss'])) and
                (val_acc > max(val_metric_history_dict['val_acc']))):
            model_fname = f"{model_file_wo_ext}_epoch_{str(epoch).rjust(3,'0')}_val_acc_{round(val_acc, 1)}_val_loss_{round(val_loss, 2)}".replace(".", "-")
            model_fname = f"{model_fname}.pt"
            torch.save(model.state_dict(), model_fname)
            model_fpaths.append(model_fname)
            print('Wrote model to', model_fname)

        # Save val metrics to history
        val_metric_history_dict['val_loss'].append(val_loss)
        val_metric_history_dict['val_acc'].append(val_acc)

        # Calc Epoch Time
        end_time = datetime.now()
        epoch_time = (end_time - start_time).total_seconds()
        txt = 'Epoch time: {:.2f}s.'.format(epoch_time)
        if epoch > warmup_epochs:
            tot_time += epoch_time
            secs_per_epoch = tot_time/(epoch-warmup_epochs)
            txt += ' (Running avg: {:.2f}s)'.format(secs_per_epoch)
        print(txt)

    print('Total training time: {:.2f}, {:.2f} secs/epoch.'.format(tot_time, secs_per_epoch))
    return model_fpaths


def test_main(args, model_fpaths):
    if len(model_fpaths) == 0:
        print("No checkpoints to evaluate")
        return
    model_file = model_fpaths[-1]
    print('Reading', model_file)
    model = TrajMLPClassifier(**args.model_kwargs)
    model.load_state_dict(torch.load(model_file))
    model.to(device)

    # test_loader = get_test_loader_hdf5(args.dataset, BATCH_SIZE)
    validation_loader = get_validation_loader_hdf5(args.dataset, BATCH_SIZE)

    print('=========')
    print('Simple:')
    with torch.no_grad():
        evaluate(model, validation_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to input hdf5 dataset",
        required=True,
    )
    parser.add_argument(
        "--save-ckpts",
        action="store_true",
        default=False
    )
    args = parser.parse_args()

    if not args.save_ckpts:
        print("WARNING: Not saving checkpoints.")
        time.sleep(1)

    args.model_kwargs = {
        "num_classes": 2,
        "token_dim": 9,
        "sample_size": SAMPLE_SIZE,
    }

    model_fpaths = train_main(args)
    test_main(args, model_fpaths)
