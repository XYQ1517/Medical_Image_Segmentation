import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from mypath import Path
from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from utils.loss import Loss
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
from modeling.unet import Unet
from modeling.CENet import CE_Net
from modeling.CPFNet import CPFNet
from modeling.FATNet import FAT_Net
from modeling.GFANet import GFANet
from modeling.HiFormer import HiFormer
from modeling.CASCADE import PVT_CASCADE
from modeling.DconnNet import DconnNet
from modeling.MISSFormer import MISSFormer
from modeling.H_vmunet import H_vmunet
from modeling.VMUNet import VMUNet
from modeling.VMUNetv2 import VMUNetV2
from modeling.AC_MambaSeg import AC_MambaSeg
from modeling.TransUNet import TransUNet
from modeling.EMCAD import EMCADNet
from modeling.EVSSD import EVSSDNet
from modeling.DATransUNet import DA_Transformer
from modeling.nnWNet import WNet2D
from modeling.UWTNet import UWTNet
from modeling.ConDSeg import ConDSeg
from modeling.MoEMixer import MoEMixer
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'


class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()

        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader = make_data_loader(args, **kwargs)

        # Define network
        model = MoEMixer(num_classes=args.num_classes)
        # model.load_from()

        # Define Optimizer
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=args.weight_decay)

        # Define loss
        self.device=torch.device("cuda:{}".format(self.args.gpu_ids[0]))
        loss = Loss(self.device, cuda=True)
        self.criterion = loss.build_loss(mode=args.loss_type)
        self.model, self.optimizer = model, optimizer
        
        # Define Evaluator
        self.evaluator = Evaluator(2)
        # Define lr scheduler
        # self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.to(self.device)

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0
        self.lr = args.lr
        self.loss_number = 0
        self.factor = False
        self.best_IoU = 0

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        self.evaluator.reset()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        size_rates = [1]
        for i, sample in enumerate(tbar):
            for rate in size_rates:
                image, target = sample['image'], sample['label']
                if self.args.cuda:
                    image, target = image.to(self.device), target.to(self.device)
                target_ = torch.unsqueeze(target, 1)
                trainsize = int(round(self.args.image_size[0] * rate / 32) * 32)
                if rate != 1:
                    image = F.interpolate(image, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    target_ = F.interpolate(target_, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

                # output, o0, o1, o2 = self.model(image)
                output = self.model(image)

                # 输出1通道
                output_n = output.data.cpu().numpy()
                output_n[output_n >= 0.5] = 1
                output_n[output_n < 0.5] = 0
                target_n = target_.cpu().numpy()

                # l0 = self.criterion(o0, target_)
                # l1 = self.criterion(o1, target_)
                # l2 = self.criterion(o2, target_)

                # loss = self.criterion(output, target_) + l0 + l1 + l2
                loss = self.criterion(output, target_)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                tbar.set_description('Train loss: %.5f' % (train_loss / (i + 1)))
                self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

                # Add batch sample into evaluator
                self.evaluator.add_batch(target_n, output_n)

        train_loss /= (num_img_tr * 3)
        Acc = self.evaluator.Accuracy()
        IoU = self.evaluator.Intersection_over_Union()
        DSC = self.evaluator.DSC()
        SE = self.evaluator.Sensitivity()
        SP = self.evaluator.Specificity()
        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        self.writer.add_scalar('train/Acc', Acc, epoch)
        self.writer.add_scalar('train/IoU', IoU, epoch)
        self.writer.add_scalar('train/DSC', DSC, epoch)
        self.writer.add_scalar('train/SE', SE, epoch)
        self.writer.add_scalar('train/SP', SP, epoch)
        print('Train:')
        print('[Epoch:{}, Train loss:{}]'.format(epoch, train_loss))
        print("Acc:{}, IoU:{}, DSC:{}, SE:{}, E:{}".format(Acc, IoU, DSC, SE, SP))
        return train_loss

    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        val_loss = 0.0
        num_img_tr = len(self.val_loader)
        for i, sample in enumerate(tbar):
            image, target = sample[0]['image'], sample[0]['label']
            if self.args.cuda:
                image, target = image.to(self.device), target.to(self.device)
            with torch.no_grad():
                # output = self.model(image, "test")
                output = self.model(image)

            # 输出1通道
            target_ = torch.unsqueeze(target, 1)
            output_n = output.data.cpu().numpy()
            output_n[output_n >= 0.5] = 1
            output_n[output_n < 0.5] = 0
            target_n = target_.cpu().numpy()

            loss = self.criterion(output, target_)
            val_loss += loss.item()
            tbar.set_description('Test loss: %.5f' % (val_loss / (i + 1)))

            # Add batch sample into evaluator
            self.evaluator.add_batch(target_n, output_n)

            if i % (num_img_tr // 1) == 0:
                self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, i, split='Val')

        # Fast test during the training
        val_loss /= num_img_tr
        Acc = self.evaluator.Accuracy()
        IoU = self.evaluator.Intersection_over_Union()
        DSC = self.evaluator.DSC()
        SE = self.evaluator.Sensitivity()
        SP = self.evaluator.Specificity()
        self.writer.add_scalar('val/total_loss_epoch', val_loss, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/IoU', IoU, epoch)
        self.writer.add_scalar('val/DSC', DSC, epoch)
        self.writer.add_scalar('val/SE', SE, epoch)
        self.writer.add_scalar('val/SP', SP, epoch)
        print('Validation:')
        print('[Epoch:{}, Val loss:{}]'.format(epoch, val_loss))
        print("Acc:{}, IoU:{}, DSC:{}, SE:{}, SP:{}".format(Acc, IoU, DSC, SE, SP))

        new_IoU = IoU
        if new_IoU > self.best_IoU:
            is_best = True
            self.best_IoU = new_IoU
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_IoU': self.best_IoU,
            }, is_best)

        return val_loss


def main():
    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument('--out-stride', type=int, default=8, help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='ISIC2018', choices=['Kvasir', 'ISIC2018', 'BUSI', 'GLAS'], help='dataset name')
    parser.add_argument('--workers', type=int, default=8, metavar='N', help='dataloader threads')
    parser.add_argument('--num_classes', type=int, default=1, help='output channel of network')
    parser.add_argument('--image_size', type=int, default=(256, 256), help='(256, 256)')
    parser.add_argument('--sync-bn',  type=bool, default=False, help='whether to use sync bn')
    parser.add_argument('--freeze-bn', type=bool, default=False, help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='BCE_Dice',
                        choices=['ce', 'con_ce', 'focal', 'BCE_Dice', 'Focal_Dice', 'CE_Dice'], help='loss func type')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=300, metavar='N', help='number of epochs to train')
    parser.add_argument('--start_epoch', type=int, default=0, metavar='N', help='start epochs (default:1)')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N', help='input batch size for training (default: 8)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=3e-4, metavar='LR', choices=[5e-4, 3e-4, 2e-4])
    parser.add_argument('--lr-scheduler', type=str, default='cos', choices=['loss_lr_1', 'poly', 'step', 'cos'])
    parser.add_argument('--momentum', type=float, default=0.9,  metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False, help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0', help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')            
    # checking point
    parser.add_argument('--resume', type=str, default=None, help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None, help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False, help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1, help='evaluation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False, help='skip validation during training')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must bomma-separated list of integers only')

    if args.checkname is None:
        args.checkname = 'MoEMixer_no_all'
    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        trainer.validation(epoch)

    print("Finish!")
    trainer.writer.close()


if __name__ == "__main__":
   main()
