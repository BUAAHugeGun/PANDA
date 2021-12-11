from data.data_builder import build_data
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torch import autograd
from tqdm import tqdm
import math
import cv2
import torch
import numpy as np
import yaml
import os
import argparse
from nets.base import Resnet18_asigm
import pathlib
import random
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm

RANDOM_SEED = 1337
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True


def to_log(s, output=True, end="\n"):
    global log_file
    if output:
        print(s, end=end)
    print(s, file=log_file, end=end)


def open_config(root):
    f = open(os.path.join(root, "config.yaml"))
    config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def load(models, epoch, root):
    def _detect_latest():
        checkpoints = os.listdir(os.path.join(root, "logs"))
        checkpoints = [f for f in checkpoints if f.startswith("model-epoch-") and f.endswith(".pth")]
        checkpoints = [int(f[len("epoch-"):-len(".pth")]) for f in checkpoints]
        checkpoints = sorted(checkpoints)
        _epoch = checkpoints[-1] if len(checkpoints) > 0 else None
        return _epoch

    if epoch == -1:
        epoch = _detect_latest()
    if epoch is None:
        return -1
    for name, model in models.items():
        ckpt = torch.load(os.path.join(root, "logs/ckpts/" + name + "_epoch-{}.pth".format(epoch)))
        ckpt = {k: v for k, v in ckpt.items()}
        model.load_state_dict(ckpt)
        to_log("load model: {} from epoch: {}".format(name, epoch))
    # print("loaded from epoch: {}".format(epoch))
    return epoch


def train(args, root):
    global log_file
    pathlib.Path(os.path.join(root, 'logs/ckpts/')).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(root, 'logs/event/')).mkdir(parents=True, exist_ok=True)
    log_file = open(os.path.join(root, "logs/log.txt"), "w")
    to_log(args)
    writer = SummaryWriter(os.path.join(root, "logs/event/"))

    args_data = args['data']
    args_train = args['train']
    args_valid = args['valid']
    dataloader = build_data(args_data['data_path'], args_train["bs"], "train",
                            num_worker=args_train["num_workers"], valid_block=args_data['valid_blocks'])
    val_dataloader = build_data(args_data['data_path'], args_valid["bs"], "valid",
                                num_worker=args_valid["num_workers"], valid_block=args_data['valid_blocks'])
    model = Resnet18_asigm(N=args_data['valid_blocks']).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=args_train["lr"])
    sch = torch.optim.lr_scheduler.MultiStepLR(opt, args_train["lr_milestone"], gamma=0.5)

    load_epoch = load({"model": model, "opt": opt, "sch": sch}, args_train["load_epoch"], root)
    tot_iter = (load_epoch + 1) * len(dataloader)
    show_interval = args_train['show_interval']

    criterion = nn.MSELoss()
    accs = []
    qwks = []
    for epoch in range(load_epoch + 1, args_train['epoch']):
        model.train()
        for i, (image, label) in enumerate(dataloader):
            opt.zero_grad()
            tot_iter += 1
            image = image.cuda()
            label = label.cuda()
            fake = model(image)
            losses = criterion(fake, label.float())
            losses.backward()
            opt.step()

            fake = fake.round().int()
            acc = (fake == label).float().sum().item() / label.shape[0]
            accs.append(acc)
            qwk = cohen_kappa_score(fake.detach().cpu(), label.detach().cpu(), weights='quadratic')
            qwks.append(qwk)
            if tot_iter % args_train['show_interval'] == 0:
                to_log(
                    'epoch: {}, batch: {}, lr: {:.2e}, acc: {:.4f},'.format(
                        epoch, i, sch.get_last_lr()[0], sum(accs[-show_interval:]) / show_interval), end=' ')
                to_log("qwk:{:.3f}, loss: {:.4f}".format(sum(qwks[-show_interval:]) / show_interval, losses.item()))
                writer.add_scalar('train/loss', losses.item(), tot_iter)
                writer.add_scalar('train/acc', sum(accs[-show_interval:]) / show_interval, tot_iter)
                writer.add_scalar('train/qwk', sum(qwks[-show_interval:]) / show_interval, tot_iter)
                writer.add_scalar("lr", sch.get_last_lr()[0], tot_iter)

        if epoch % args_train["snapshot_interval"] == 0 or epoch == args_train['epoch'] - 1:
            torch.save(model.state_dict(), os.path.join(root, "logs/ckpts/model_epoch-{}.pth".format(epoch)))
            torch.save(opt.state_dict(), os.path.join(root, "logs/ckpts/opt_epoch-{}.pth".format(epoch)))
            torch.save(sch.state_dict(), os.path.join(root, "logs/ckpts/sch_epoch-{}.pth".format(epoch)))

        if epoch % args_train['test_interval'] == 0:
            preds, labels = [], []
            model.eval()
            with torch.no_grad():
                for (image, label) in tqdm(val_dataloader):
                    image = image.cuda()
                    label = label.cuda()
                    fake = model(image)
                    preds.append(fake)
                    labels.append(label)
                preds = torch.cat(preds, dim=0)
                labels = torch.cat(labels, dim=0)

                loss = criterion(preds, labels.float()).item()
                preds = preds.round().int()
                acc = (preds == labels).float().mean().item()
                qwk = cohen_kappa_score(preds.detach().cpu(), labels.detach().cpu(), weights='quadratic')
                to_log('epoch: {}, [valid] acc: {:.4f}, qwk: {:.3f}, loss: {:.4f}'.format(epoch, acc, qwk, loss))
                writer.add_scalar('valid/loss', loss, epoch)
                writer.add_scalar('valid/acc', acc, epoch)
                writer.add_scalar('valid/qwk', qwk, epoch)
        sch.step()
        writer.flush()
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str)
    args = parser.parse_args()
    train(open_config(args.root), args.root)
