import argparse
import random
import shutil
import time

import matplotlib
import pytz
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torchmetrics.classification import BinaryAUROC

from models.Generate_Model import GenerateModel

matplotlib.use("Agg")
import datetime

# import open_clip
import warnings

import matplotlib.pyplot as plt
import numpy as np
import tqdm
from clip import clip
from sklearn.metrics import confusion_matrix

from dataloader.video_dataloader import train_data_loader
from dataloader.video_dataloader import val_data_loader

warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--video_path", type=str, default="/data/zonepg/datasets/dataset/frames_1s/train"
)
parser.add_argument("--workers", type=int, default=8)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch-size", type=int, default=48)

parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument("--lr-image-encoder", type=float, default=1e-5)
parser.add_argument(
    "--image-encoder",
    type=str,
    default="clip-ViT-B-32",
    choices=["clip-ViT-B-32", "clip-ViT-B-16"],
)

parser.add_argument("--weight-decay", type=float, default=1e-4)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--print-freq", type=int, default=10)
parser.add_argument("--milestones", nargs="+", type=int, default=[30, 40])

parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--temporal-layers", type=int, default=1)
parser.add_argument("--temporal-net", type=str, default="transformer")
parser.add_argument(
    "--cls-type", type=str, default="cls", choices=["cls", "mean", "double"]
)
parser.add_argument("--set", type=int, default=1)

args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
cudnn.benchmark = True

china_tz = pytz.timezone("Asia/Shanghai")
now = datetime.datetime.now(tz=pytz.utc).astimezone(china_tz)
time_str = now.strftime("[%m-%d]-[%H:%M:%S]")


def main(set):
    data_set = set

    print("************************")
    for k, v in vars(args).items():
        print(k, "=", v)
    print("************************")

    print("*********** Fold  " + str(data_set) + " ***********")
    log_txt_path = "./log/" + time_str + "-set" + str(data_set) + "-log.txt"
    log_curve_path = "./log/" + time_str + "-set" + str(data_set) + "-log.png"
    checkpoint_path = "./checkpoint/" + time_str + "-set" + str(data_set) + "-model.pth"
    best_checkpoint_path = (
        "./checkpoint/" + time_str + "-set" + str(data_set) + "-model_best.pth"
    )
    train_annotation_file_path = "./annotation/set_" + str(data_set) + "_train.txt"
    val_annotation_file_path = "./annotation/set_" + str(data_set) + "_val.txt"

    # Data loading code
    train_data = train_data_loader(
        list_file=train_annotation_file_path,
        num_segments=16,
        duration=1,
        image_size=224,
        args=args,
    )
    val_data = val_data_loader(
        list_file=val_annotation_file_path,
        num_segments=16,
        duration=1,
        image_size=224,
        args=args,
    )

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    best_auc = 0
    recorder = RecorderMeter(args.epochs)
    print("The training name: " + time_str)

    # create model and load pre_trained parameters
    if args.image_encoder == "clip-ViT-B-32":
        CLIP_model, _ = clip.load("ViT-B/32", device="cpu")
    elif args.image_encoder == "clip-ViT-B-16":
        CLIP_model, _ = clip.load("ViT-B/16", device="cpu")

    model = GenerateModel(clip_model=CLIP_model, args=args).cuda()
    print(model.image_encoder)
    print(model.temporal_net)
    print_model_size("image_encoder", model.image_encoder)
    print_model_size("temporal_net", model.temporal_net)

    # define loss function (criterion)
    criterion = nn.BCELoss()
    # define optimizer
    optimizer = torch.optim.SGD(
        [
            {
                "params": [
                    p for n, p in model.named_parameters() if "image_encoder" not in n
                ]
            },
            {
                "params": model.image_encoder.parameters(),
                "lr": args.lr_image_encoder,
            },
        ],
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # define scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.milestones, gamma=0.1
    )

    with open(log_txt_path, "a") as f:
        for k, v in vars(args).items():
            f.write(str(k) + "=" + str(v) + "\n")
        print("train video number: ", len(train_data.video_list))
        print("train class number: ", train_data.class_num)
        print("train class weights: ", train_data.class_weights)
        print("val video number: ", len(val_data.video_list))
        print("val class number: ", val_data.class_num)
        print("val class weights: ", val_data.class_weights)
        f.write("train video number: " + str(len(train_data.video_list)) + "\n")
        f.write("train class number: " + str(train_data.class_num) + "\n")
        f.write("train class weights: " + str(train_data.class_weights) + "\n")
        f.write("test video number: " + str(len(val_data.video_list)) + "\n")
        f.write("test class number: " + str(val_data.class_num) + "\n")
        f.write("test class weights: " + str(val_data.class_weights) + "\n")

    for epoch in range(0, args.epochs):
        inf = "********************" + str(epoch) + "********************"
        start_time = time.time()
        current_learning_rate_0 = optimizer.state_dict()["param_groups"][0]["lr"]
        current_learning_rate_1 = optimizer.state_dict()["param_groups"][1]["lr"]

        with open(log_txt_path, "a") as f:
            f.write(inf + "\n")
            print(inf)
            f.write(
                "Current learning rate: "
                + str(current_learning_rate_0)
                + " "
                + str(current_learning_rate_1)
                + "\n"
            )
            print(
                "Current learning rate: ",
                current_learning_rate_0,
                current_learning_rate_1,
            )

        # train for one epoch
        train_loss = train(
            train_loader, model, criterion, optimizer, epoch, args, log_txt_path
        )

        # evaluate on validation set
        val_auc, val_los = validate(val_loader, model, criterion, args, log_txt_path)

        scheduler.step()

        # remember best acc and save checkpoint
        is_best = val_auc > best_auc
        best_auc = max(val_auc, best_auc)
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_auc": best_auc,
                "optimizer": optimizer.state_dict(),
                "recorder": recorder,
            },
            is_best,
            checkpoint_path,
            best_checkpoint_path,
        )

        # print and save log
        epoch_time = time.time() - start_time
        recorder.update(epoch, train_loss, val_los, val_auc)
        recorder.plot_curve(log_curve_path)

        print(f"The best accuracy: {best_auc:.3f}")
        print(f"An epoch time: {epoch_time:.2f}s")
        with open(log_txt_path, "a") as f:
            f.write("The best accuracy: " + str(best_auc) + "\n")
            f.write("An epoch time: " + str(epoch_time) + "s" + "\n")

    return compute_auc(val_loader, model, best_checkpoint_path, log_txt_path)


def train(train_loader, model, criterion, optimizer, epoch, args, log_txt_path):
    losses = AverageMeter("Loss", ":.4f")
    auc_meter = Meter("AUC", ":6.3f")
    progress = ProgressMeter(
        len(train_loader),
        [losses, auc_meter],
        prefix=f"Epoch: [{epoch}]",
        log_txt_path=log_txt_path,
    )
    auc_metric = BinaryAUROC()

    # switch to train mode
    model.train()

    for i, (images, target) in enumerate(train_loader):
        images = images.cuda()
        target = target.cuda().float()

        # compute output
        output = model(images).squeeze(-1)
        loss = criterion(output, target)

        # measure accuracy and record loss
        auc_score = auc_metric(output, target) * 100
        losses.update(loss.item(), images.size(0))
        auc_meter.update(auc_score)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print loss and accuracy
        if i % args.print_freq == 0 or i == len(train_loader) - 1:
            progress.display(i)

    return losses.avg


def validate(val_loader, model, criterion, args, log_txt_path):
    losses = AverageMeter("Loss", ":.4f")
    auc_meter = Meter("AUC", ":6.3f")
    progress = ProgressMeter(
        len(val_loader), [losses, auc_meter], prefix="Test: ", log_txt_path=log_txt_path
    )
    auc_metric = BinaryAUROC()

    # switch to evaluate mode
    model.eval()

    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda().float()

            # compute output
            output = model(images).squeeze(-1)
            loss = criterion(output, target)

            all_outputs.append(output)
            all_targets.append(target)

            # measure accuracy and record loss
            auc_score = auc_metric(output, target) * 100
            losses.update(loss.item(), images.size(0))
            auc_meter.update(auc_score)

            if i % args.print_freq == 0 or i == len(val_loader) - 1:
                progress.display(i)

        all_outputs = torch.cat(all_outputs)
        all_targets = torch.cat(all_targets)
        total_auc_score = auc_metric(all_outputs, all_targets) * 100

        # TODO: this should also be done with the ProgressMeter
        print(f"Current AUC: {total_auc_score:.3f}")
        with open(log_txt_path, "a") as f:
            f.write(f"Current AUC: {total_auc_score:.3f}" + "\n")
    return total_auc_score, losses.avg


def save_checkpoint(state, is_best, checkpoint_path, best_checkpoint_path):
    torch.save(state, checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, best_checkpoint_path)


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class Meter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0

    def update(self, val):
        self.val = val

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "}"
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    def __init__(self, num_batches, meters, prefix="", log_txt_path=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.log_txt_path = log_txt_path

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print_txt = "\t".join(entries)
        print(print_txt)
        with open(self.log_txt_path, "a") as f:
            f.write(print_txt + "\n")

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches - 1) + "]"


class RecorderMeter:
    """Computes and stores the minimum loss value and its epoch index"""

    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros(
            (self.total_epoch, 2), dtype=np.float32
        )  # [epoch, train/val]
        self.epoch_auc = np.zeros(
            (self.total_epoch, 1), dtype=np.float32
        )  # [epoch, val]

    def update(self, idx, train_loss, val_loss, val_auc):
        self.epoch_losses[idx, 0] = train_loss * 50
        self.epoch_losses[idx, 1] = val_loss * 50
        self.epoch_auc[idx, 0] = val_auc
        self.current_epoch = idx + 1

    def plot_curve(self, save_path):
        title = "the auc/loss curve of train/val"
        dpi = 80
        width, height = 1600, 800
        legend_fontsize = 10
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.total_epoch)])  # epochs
        y_axis = np.zeros(self.total_epoch)

        plt.xlim(0, self.total_epoch)
        plt.ylim(0, 100)
        interval_y = 5
        interval_x = 1
        plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
        plt.yticks(np.arange(0, 100 + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel("the training epoch", fontsize=16)
        plt.ylabel("AUC", fontsize=16)

        y_axis[:] = self.epoch_auc[:, 0]
        plt.plot(x_axis, y_axis, color="y", linestyle="-", label="valid-auc", lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 0]
        plt.plot(x_axis, y_axis, color="g", linestyle=":", label="train-loss-x50", lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 1]
        plt.plot(x_axis, y_axis, color="y", linestyle=":", label="valid-loss-x50", lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            # print('Curve was saved')
        plt.close(fig)


def compute_auc(
    val_loader,
    model,
    best_checkpoint_path,
    log_txt_path,
):
    pre_trained_dict = torch.load(best_checkpoint_path)["state_dict"]
    model.load_state_dict(pre_trained_dict)
    auc_metric = BinaryAUROC()

    model.eval()

    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm.tqdm(val_loader)):
            images = images.cuda()
            target = target.cuda().float()

            output = model(images).squeeze(-1)
            all_outputs.append(output)
            all_targets.append(target)

    all_outputs = torch.cat(all_outputs)
    all_targets = torch.cat(all_targets)
    auc_score = auc_metric(all_outputs, all_targets) * 100

    print(f"Final AUC: {auc_score:.3f}")

    with open(log_txt_path, "a") as f:
        f.write("************************" + "\n")
        f.write(f"AUC: {auc_score:.2f}" + "\n")
        f.write("************************" + "\n")

    return auc_score


def print_model_size(model_name, model):
    print("********* Model Size *********")
    print(model_name)
    print(
        "Total params: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1000000.0)
    )
    print("*****************************")


if __name__ == "__main__":
    auc_score = main(args.set)

    print("********* Final Results *********")
    print(f"AUC: {auc_score:0.2f}")
    print("*********************************")
