# -*- codeing = utf-8 -*-
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import time
import torch

from torch.utils.tensorboard import SummaryWriter

from Datasets import create_dataloader
# from classify_model import ResNet as Model   
# from classify_model import ResNeXt as Model    

from classify_model import EfficientNet as Model    
# from classify_model import ConvNext as Model    

# from classify_model import WideResNet as Model    
# from classify_model import ResNet_C6_2 as Model    
# from classify_model import ResNet_2_fc as Model    
# 

from torch.optim import SGD, Adam, lr_scheduler

# 混合精度训练
from torch.cuda import amp

from pathlib import Path
from general import increment_path, ModelEMA, select_device, init_seeds, one_cycle

import argparse
import yaml
from verification import val
from loss import ComputeLoss



def train(opt, device):
    print("训练设备：{}".format(device))

    weights, img_size, batch_size, epochs, is_MAP, is_EMA = opt.weights, opt.imgsz, opt.batch_size, opt.epochs, opt.is_MAP, opt.is_EMA
    save_dir, workers = opt.save_dir, opt.workers
    loss_type = opt.loss_type
    seed = opt.seed


    ## 首先确定随机种子, seed设置为0会关闭cudnn.benchmark，更有确定性和可复现性
    init_seeds(seed=seed)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # 把用到的参数保存下来
    with open(save_dir + '/opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)

    # 最好的精度
    best_accuracy = 0.0   
    best_epoch = 0 

    ## 构建模型
    model = Model().to(device)
    if weights.endswith('.pth') or weights.endswith('.pt'):
        ckpt = torch.load(weights, map_location='cpu')
        print("导入网络模型参数")
        model.load_state_dict(ckpt['model'].float().state_dict())
        
        del ckpt

    # for k,v in model.named_parameters():
    #     print(k, v.requires_grad)        

    # 构建训练集和验证集
    train_root_dir = '../dataset/train_images'
    val_root_dir = '../dataset/val_images'
    train_dataloader, _ = create_dataloader(train_root_dir, img_size=img_size,shuffle=True, batch_size=batch_size, num_workers=workers, mode='train')
    val_dataloader, _ = create_dataloader(val_root_dir, img_size=img_size, shuffle=False, batch_size=batch_size, num_workers=workers, mode='val') 


    # 构建损失函数
    num_classes = model.num_classes
    compute_loss = ComputeLoss(loss_type=loss_type, num_classes=num_classes)


    # 构建优化器
    learning_rate = 1e-3
    if opt.optimizer == 'Adam':
        optimizer = Adam(model.parameters(), lr=learning_rate) # adjust beta1 to momentum
    else:
        optimizer = SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    

    # Scheduler
    if opt.lr_scheduler == 'None':
        scheduler = None
    else:
        lrf = 0.1
        if opt.lr_scheduler == 'linear':
            lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - lrf) + lrf  # linear
        else:
            lf = one_cycle(1, lrf, epochs)  # cosine 1->hyp['lrf']
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)
    

    # tensorboard显示训练情况
    writer = SummaryWriter(save_dir)


    # EMA
    ema = ModelEMA(model) if is_EMA else None

    # 训练参数设置
    train_step = 0
    # if scheduler:
    #     scheduler.last_epoch = -1  # do not move
    scaler = amp.GradScaler(enabled=is_MAP)
    start_time = time.time()
    # 开始训练
    for epoch in range(epochs):
        print('---------第{}轮训练开始----------'.format(epoch))

        # 训练一个epoch的指数平均损失
        mloss = 0
        # 训练步骤开始
        model.train()
        for iter_id, data in enumerate(train_dataloader):
            # targets shape [batch_size]
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            with amp.autocast(enabled=is_MAP):
                outputs = model(imgs)
                # 整个batch的损失
                loss = compute_loss(outputs, targets)

            # 优化器优化模型
            optimizer.zero_grad()
            # Backward
            # 对损失乘以一个scale因子，然后再进行反向传播计算梯度。乘以scale因子，可以避免float16梯度出现underflow的情况
            scaler.scale(loss).backward()
            
            scaler.step(optimizer)  # optimizer.step，梯度更新，会先对梯度进行unscale，再进行梯度更新
            scaler.update()         # 更新scale因子
            if ema:
                ema.update(model)
            
            loss_item = loss.item()
            train_step += 1
            if train_step % 100 == 0:
                end_time = time.time()
                print('训练次数:{}，Loss:{}，训练时长:{}'.format(train_step, loss_item, end_time - start_time))
                
            # 训练损失的指数加权平均
            mloss = (mloss * iter_id + loss_item) / (iter_id + 1)

        lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('lr/lr', lr, epoch)
        if scheduler:
            scheduler.step()

        # 进行验证
        mean_val_loss, total_accuracy, accuracy_classes = val(
            ema.ema if ema else model,
            val_dataloader,
            device,
            compute_loss
        )

        print('整体测试集的Loss：{}'.format(mean_val_loss))
        print('整体测试集的正确率：{}'.format(total_accuracy))


        ## 可视化
        writer.add_scalar('train/loss', mloss, epoch)
        writer.add_scalar('val/loss', mean_val_loss, epoch)
        writer.add_scalar('metrics/total_accuracy', total_accuracy, epoch)
        for class_name, accuracy_class in accuracy_classes.items():
            writer.add_scalar('metrics/'+class_name, accuracy_class, epoch)


        ## 模型的保存
        if total_accuracy > best_accuracy:
            best_accuracy = total_accuracy
            best_epoch = epoch

        ckpt = {
            'model': ema.ema if ema else model,
            'best_accuracy': best_accuracy,
            'epoch':epoch,
        }

        torch.save(ckpt, save_dir+'/last.pt')
        if best_accuracy == total_accuracy:
            torch.save(ckpt, save_dir+'/best.pt')
        

        print(f"模型已保存到路径{save_dir}下")
    
    print(f"best_accuracy:{best_accuracy}, best_epoch:{best_epoch}")
    writer.close()


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./', help='initial weights path')
    # parser.add_argument('--weights', type=str, default='./runs/train/exp47/best.pt', help='initial weights path')

    parser.add_argument('--hyp', type=str, default='./data/hyps/hyp.scratch.s2anet.LAR1024.yaml', help='hyperparameters path')

    parser.add_argument('--data', type=str, default='./data/LAR1024.yaml', help='dataset.yaml path')

    parser.add_argument('--loss-type', type=str, choices=['softmax', 'BCE', 'focal_loss'], default='softmax', help='optimizer')
    parser.add_argument('--seed', type=int, default=1)

    parser.add_argument('--epochs', type=int, default=48)
    parser.add_argument('--batch-size', type=int, default=64, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=tuple, default=(320,240), help='train, val image size (pixels)')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='Adam', help='optimizer')
    parser.add_argument('--lr-scheduler', type=str, choices=['consine', 'linear', 'None'], default="None", help='lr-scheduler')

    # 是否使用混合精度训练，automatic mixed-precision training
    parser.add_argument('--is_MAP', action='store_true', default=True)
    # 是否使用模型平滑
    parser.add_argument('--is_EMA', action='store_true', default=True)

    parser.add_argument('--device', default='1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default='./runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')



    # 梯度裁剪    
    parser.add_argument('--grad-clip', action='store_true', default=True, help='Gradient clipping')

    # 通过梯度累积达到的名义batch-size，为0表示不进行梯度累积；大于0 时，会根据与batch-size的关系，确定累积几次进行一次反向传播
    # parser.add_argument('--nominal-bs', type=int, default=64)
    parser.add_argument('--nominal-bs', type=int, default=0)

    # 是否对训练参数进行分组，不同的参数分组分别确定学习率
    parser.add_argument('--params_groups', action='store_true', default=False)


    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')


    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')

    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    
    
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')


    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def main(opt):

    opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    device = select_device(opt.device, batch_size=opt.batch_size)

    train(opt, device)
    pass


if __name__=="__main__":
    opt = parse_opt()
    main(opt)