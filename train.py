from data.data_builder import build_data
from torchvision.utils import save_image
from tensorboardX import SummaryWriter
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
from nets.base import Base_net


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
        ckpt = torch.load(os.path.join(root, "logs/" + name + "_epoch-{}.pth".format(epoch)))
        ckpt = {k: v for k, v in ckpt.items()}
        model.load_state_dict(ckpt)
        to_log("load model: {} from epoch: {}".format(name, epoch))
    # print("loaded from epoch: {}".format(epoch))
    return epoch


def train(args, root):
    global log_file
    if not os.path.exists(os.path.join(root, "logs")):
        os.mkdir(os.path.join(root, "logs"))
    if not os.path.exists(os.path.join(root, "logs/result/")):
        os.mkdir(os.path.join(root, "logs/result/"))
    if not os.path.exists(os.path.join(root, "logs/result/event")):
        os.mkdir(os.path.join(root, "logs/result/event"))
    log_file = open(os.path.join(root, "logs/log.txt"), "w")
    to_log(args)
    writer = SummaryWriter(os.path.join(root, "logs/result/event/"))

    args_data = args['data']
    args_train = args['train']
    dataloader = build_data(args_data['data_path'], args_train["bs"], "train",
                            num_worker=args_train["num_workers"], valid_block=args_data['valid_blocks'])
    model = Base_net(image_size=args_train['image_size']).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=args_train["lr"])
    sch = torch.optim.lr_scheduler.MultiStepLR(opt, args_train["lr_milestone"], gamma=0.5)

    load_epoch = load({"model": model, "opt": opt, "sch": sch}, args_train["load_epoch"], root)
    tot_iter = (load_epoch + 1) * len(dataloader)

    opt.step()
    criterion = nn.CrossEntropyLoss()
    accs = []
    for epoch in range(load_epoch + 1, args_train['epoch']):
        sch.step()
        for i, (image, label) in enumerate(dataloader):
            tot_iter += 1
            image = image.cuda()
            label = label.cuda()
            fake = model(image)
            losses = criterion(fake, label)
            losses.backward()
            opt.step()
            acc = torch.eq(torch.max(fake, dim=1)[1], label)
            acc = acc.sum().cpu().float() / label.shape[0]
            accs.append(acc)
            if tot_iter % args_train['show_interval'] == 0:
                to_log(
                    'epoch: {}, batch: {}, lr: {}, acc: {}'.format(
                        epoch, i, sch.get_last_lr()[0], sum(accs[-100:]) / 100.), end=' ')
                to_log("loss: {}".format(losses.item()))
            writer.add_scalar('train/loss', losses, tot_iter)
            writer.add_scalar('train/acc', sum(accs[-100:]) / 100., tot_iter)
            writer.add_scalar("lr", sch.get_last_lr()[0], tot_iter)

        if epoch % args_train["snapshot_interval"] == 0:
            torch.save(model.state_dict(), os.path.join(root, "logs/model_epoch-{}.pth".format(epoch)))
            torch.save(opt.state_dict(), os.path.join(root, "logs/opt_epoch-{}.pth".format(epoch)))
            torch.save(sch.state_dict(), os.path.join(root, "logs/sch_epoch-{}.pth".format(epoch)))
        # if epoch % args_train['test_interval'] == 0:
        #     # label = torch.tensor([5]).expand([64]).cuda()
        #     # input_test[:, 1, :, :] = label.reshape(64, 1, 1).expand(64, image_size, image_size)
        #     # G_out = G(input_test)
        #     image = y.clone().detach()
        #     image = batch_image_merge(image)
        #     image = imagetensor2np(image)
        #     y = y / 2 + 0.5
        #     y = y.clamp(0, 1)
        #     save_image(y, os.path.join(root, "logs/reconstruction-{}.png".format(epoch)))
        #     writer.add_image('image{}/fake'.format(epoch), cv2.cvtColor(image, cv2.COLOR_BGR2RGB), tot_iter,
        #                      dataformats='HWC')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str)
    parser.add_argument("--test", default=False, action='store_true')
    args = parser.parse_args()
    train(open_config(args.root), args.root)
