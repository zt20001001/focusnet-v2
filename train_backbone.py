import os
# os.environ['CUDA_VISIBLE_DEVICES'] = options.gpu
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import csv, random
import numpy as np
from model.model import s_net
from model.MDANet.net_backbone import MDANet

from model.MDANet.HybridNet import HybridModel
from model.unet_model import UNet3d
from model.MDANet.mcaNet import MCANet
# Ablation
from model.Net.baseline import BaseNet  # M1
from model.Net.baseline_DRSC import BaseNet_wDRSC  # M2
from model.Net.baseline_wCSSA import BaseNet_wCSSA  # M3
from model.Net.baseline_wMSFE import BaseNet_wMSFE  # M4

from model.Net.baseline_resDconv import BaseNet_wSE
from model.Net.baseline_resASPP import BaseNet_wASPP
from model.Net.baseline_wSkip import BaseNet_wSkip
from model.Net.baseline_wDenseSkip import BaseNet_wDenseSkip
from model.Net.baseline_MSCAM import BaseNet_wMSCAM
from model.Net.proposed import Net
from model.Net.proposedv2 import Net as Net2

from model.MDANet.HierarchyNet import HeadNeckSegModel

from dataset.brain_dataset import BrainDataset

from torch.utils import data
from losses import FocalLoss, DiceLoss, Adaptive_Region_Specific_TverskyLoss
from validation import evaluation
from dataset_split import split_dataset
from utils import *
from metrics import *
from optparse import OptionParser
import SimpleITK as sitk

from torch.utils.tensorboard import SummaryWriter
import time
import pdb
from monai.utils import set_determinism
from tqdm import tqdm
from medpy.metric.binary import hd95


DATA_CSV_PATH = '/home/image/nvme/zt/code/focusnet-v2/data_preprocess/pddca/new_data.csv'
TRAIN_CSV= '/home/image/nvme/zt/code/focusnet-v2/data_preprocess/pddca/new_train.csv' 
VAL_CSV = '/home/image/nvme/zt/code/focusnet-v2/data_preprocess/pddca/val.csv'

def train_net(net, options):

    # 获取当前设备
    # device = torch.device(options.gpu)

    data_path = options.data_path
    csv_file = options.data_path + 'new_train.csv'
    origin_spacing_data_path = options.data_path + 'origin_spacing_croped/'
    if  os.path.exists(origin_spacing_data_path) == False:   
        os.makedirs(origin_spacing_data_path)

    
    # z_size is the random crop size along z-axis, you can set it larger if have enough gpu memory
    # trainset = BrainDataset(csv_file, data_path, data_path, mode='train', z_size=48)
    trainset = BrainDataset(TRAIN_CSV, data_path, data_path, mode='train', z_size=40)
    trainLoader = data.DataLoader(trainset, batch_size=options.batch_size, shuffle=True, num_workers=0)
    # valset = BrainDataset(val_csv_file, data_path, data_path, mode='val', z_size=48)
    # valLoader = data.DataLoader(valset, batch_size=options.batch_size, shuffle=False, num_workers=0)

    # for batch in trainLoader:
    #     imgs, lbls, weight = batch
    #     print("Image shape:", imgs.shape)
    #     print("Label shape:", lbls.shape)
    
    # test_data_list, test_label_list = load_val_data(origin_spacing_data_path)
    val_data_list, val_label_list = load_val_data(data_path=data_path)

    writer = SummaryWriter(options.log_path + options.unique_name)
    
    
    # optimizer = optim.SGD(net.parameters(), lr=options.lr, momentum=0.9, weight_decay=0.0005)
    optimizer = optim.AdamW(net.parameters(), lr=options.lr, weight_decay=0.0001)

    org_weight = torch.FloatTensor(options.org_weight).unsqueeze(1).cuda()
    criterion_fl = FocalLoss(10, alpha=org_weight)
    criterion_dl = DiceLoss()
    # criterion_arsl = Adaptive_Region_Specific_TverskyLoss(num_region_per_axis=(5, 12, 12))
    # 新增：初始化训练状态变量
    start_epoch = 0
    best_epoch = 0
    best_dice = 0
    patience = 0
    best_organs = []
    
    checkpoint_path = f'{options.cp_path}{options.unique_name}/checkpoint.pth'
    # 新增：加载断点
    if os.path.exists(checkpoint_path):
        print(f"=> Loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # 从下一个epoch开始
        best_dice = checkpoint['best_dice']
        best_organs = checkpoint['best_organs']
        best_epoch = checkpoint['best_epoch']
        patience = checkpoint['patience']
        print(f"=> Loaded checkpoint (epoch {checkpoint['epoch']})")
    
    for epoch in range(start_epoch, options.epochs):
        # print('Starting epoch {}/{}'.format(epoch+1, options.epochs))
        epoch_loss = 0
        best_epoch_flag = False
        multistep_scheduler = multistep_lr_scheduler_with_warmup(
            optimizer, init_lr=options.lr, epoch=epoch, warmup_epoch=5, 
            lr_decay_epoch=[100, 200], max_epoch=options.epochs, gamma=0.1)
        # print('current lr:', multistep_scheduler)
        
        net.train()
        epoch_iterator = tqdm(trainLoader, desc="Training(Epoch X/X)", dynamic_ncols=True)
        epoch_iterator.set_description("Training (Epoch %d/%d)" % (epoch+1, options.epochs))
        for i, (img, label, weight) in enumerate(epoch_iterator, 0):
        # for i, (img, label, weight) in enumerate(trainLoader, 0):
            img = img.cuda()
            label = label.cuda()
            weight = weight.cuda()

            end = time.time()

            optimizer.zero_grad()

            result = net(img)

            # # 修改损失计算以处理字典输出并包含辅助损失
            # if isinstance(result, dict):
            #     # 深度监督模式
            #     final_out = result['final_out']
            #     aux_out1 = result['aux_out1']
            #     aux_out2 = result['aux_out2']

            #     loss_final = criterion_dl(final_out, label, weight)
            #     loss_aux1 = criterion_dl(aux_out1, label, weight) # 可能需要调整标签大小或上采样辅助输出
            #     loss_aux2 = criterion_dl(aux_out2, label, weight) # 可能需要调整标签大小或上采样辅助输出

            #     # 组合损失，可以调整辅助损失的权重 (例如 0.4)
            #     loss = loss_final + 0.4 * (loss_aux1 + loss_aux2)

            #     if options.rlt > 0:
            #         loss_final_fl = criterion_fl(final_out, label, weight)
            #         loss = loss + loss_final_fl # 如果使用 Focal Loss，也加到主输出上
            # else:
            #     # 原有逻辑
            #     if options.rlt > 0:
            #         # loss = criterion_fl(result, label, weight) + options.rlt * criterion_dl(result, label, weight) + criterion_arsl(result, label)
            #         loss = criterion_fl(result, label, weight) + options.rlt * criterion_dl(result, label, weight)
            #     else:
            #         loss = criterion_dl(result, label, weight)

            # 原有逻辑
            if options.rlt > 0:
                loss = criterion_fl(result, label, weight) + options.rlt * criterion_dl(result, label, weight)
            else:
                loss = criterion_dl(result, label, weight)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_time = time.time() - end
            # print(
            #     "Trainning: Epoch {}/{} Sample {}/{}".format(epoch+1, options.epochs, i+1, len(trainLoader)),
            #     "loss: {:.5f}".format(loss.item()),
            #     "time {:.2f}s".format(batch_time),
            # )
            epoch_iterator.set_postfix(
                {"iter": f"{i+1}/{len(trainLoader)}", "loss": f"{loss.item():.5f}", "lr": f"{multistep_scheduler}"}
            )
        # print('[epoch %d] epoch loss: %.5f'%(epoch+1, epoch_loss/(i+1)))

        writer.add_scalar('Train/Loss', epoch_loss/(i+1), epoch+1)
        writer.add_scalar('LR', multistep_scheduler, epoch+1)

        if os.path.isdir('%s%s/'%(options.cp_path, options.unique_name)):
            pass
        else:
            os.mkdir('%s%s/'%(options.cp_path, options.unique_name))

        # if (epoch+1)%10==0:
        #     torch.save(net.state_dict(), '%s%s/CP%d.pth'%(options.cp_path, options.unique_name, epoch+1))
            
        # 定义器官字典
        pddca_organs_dict = {
            1: "BrainStem",
            2: "Chiasm",
            3: "Mandible",
            4: "OPN_L",
            5: "OPN_R",
            6: "Parotid_L",
            7: "Parotid_R",
            8: "SMG_L",
            9: "SMG_R"
        }
        structseg_organs_dict = {
            1: "Brain Stem", # 脑干
            2: "Eye L", # 左眼
            3: "Eye R", # 右眼
            4: "Lens L", # 左晶状体
            5: "Lens R", # 右晶状体
            6: "Opt Nerve L", # 左视神经
            7: "Opt Nerve R", # 右视神经
            8: "Opt Chiasm", # 视交叉
            9: "Temporal Lobes L", # 左颞叶
            10: "Temporal Lobes R", # 右颞叶
            11: "Pituitary", # 垂体
            12: "Parotid Gland L", # 左腮腺
            13: "Parotid Gland R", # 左腮腺
            14: "Inner Ear L", # 左内耳
            15: "Inner Ear R", # 右内耳
            16: "Mid Ear L", # 左中耳
            17: "Mid Ear R", # 右中耳
            18: "TM Joint L", # 左颞下颌关节
            19: "TM Joint R", # 右颞下颌关节
            20: "Spinal Cord", # 脊髓
            21: "Mandible L", # 左下颌骨
            22: "Mandible R" # 右下颌骨
        }
        
        avg_dice, dice_list, avg_hd95, hd95_list = validation(net, val_data_list, val_label_list)
        writer.add_scalar('Test/AVG_Dice', avg_dice, epoch+1)
        for idx in range(9):
            writer.add_scalar('Test/Dice-%s'%(pddca_organs_dict.get(idx+1)), dice_list[idx], epoch+1)
        
        # 新增：保存检查点（每个epoch都保存）
        checkpoint = {
            'epoch': epoch,  # 当前已完成epoch
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_dice': best_dice,
            'best_organs': best_organs,
            'best_epoch': best_epoch,
            'patience': patience,
        }
        torch.save(checkpoint, '%s%s/checkpoint.pth'%(options.cp_path, options.unique_name))
        
        patience += 1
        if avg_dice >= best_dice:
            patience = 0
            best_epoch_flag = True
            best_dice = avg_dice
            best_organs = dice_list
            torch.save(net.state_dict(), '%s%s/bestmodel.pth'%(options.cp_path, options.unique_name))
            print('Best save done!')
        if best_epoch_flag:
            best_epoch = epoch+1
            
        # samples_dice = []
        # samples_dice = [f"case{i+1}:{x:.5f}" for i,x in enumerate(sample_dice_list)]
        # # print('dice: %.5f/best dice: %.5f'%(avg_dice, best_dice),f"-- best epoch at epoch{best_epoch}")
        # group_samples = [samples_dice[i:i+5] for i in range(0, len(samples_dice), 5)]   
        print('dice: %.5f/best dice: %.5f'%(avg_dice, best_dice),f"-- best epoch at epoch{best_epoch}")
        # 输出每个器官的平均Dice系数
        print("Best organs' mean DSC：")
        for idx, dice in enumerate(best_organs):
            print(f"{pddca_organs_dict.get(idx+1)}: {dice:.5f}")
        

def load_test_data(data_path):
    test_name_list = ['0522c0555', '0522c0576', '0522c0598', '0522c0659', '0522c0661',
                    '0522c0667', '0522c0669', '0522c0708', '0522c0727', '0522c0746',
                    ]
    test_data_list = []
    test_label_list = []

    for name in test_name_list:
        CT = sitk.ReadImage(data_path + name + '_data.nii.gz')
        label = sitk.ReadImage(data_path + name + '_label.nii.gz')

        test_data_list.append(CT)
        test_label_list.append(label)

    return test_data_list, test_label_list

def load_val_data(data_path):
    # val_name_list = split_dataset(DATA_CSV_PATH, 160, 18, TRAIN_CSV_PATH, 0)
    with open(VAL_CSV, 'r') as f:
        reader = csv.reader(f)
        data = [row for row in reader if row]

    val_name_list = [row[0] for row in data] 
    val_data_list = []
    val_label_list = []

    for name in val_name_list:
        CT = sitk.ReadImage(data_path + name + '_data.nii.gz')
        label = sitk.ReadImage(data_path + name + '_label.nii.gz')

        val_data_list.append(CT)
        val_label_list.append(label)

    return val_data_list, val_label_list

def validation(net, test_data_list, test_label_list):
    dicecomputer = sitk.LabelOverlapMeasuresImageFilter() 
    
    # 初始化指标存储
    dice_list = np.zeros(9)
    hd95_list = np.zeros(9)
    sample_dice_list = []
    sample_hd95_list = []

    for i in range(len(test_data_list)):
        tmp_dice = np.zeros(9)
        tmp_hd95 = np.zeros(9)
        
        itkCT = test_data_list[i]
        itkLabel = test_label_list[i]

        itkPred = evaluation(net, itkCT)
        # print("itkPred shape:", itkPred.GetSize())
        # print("spacing:", itkCT.GetSpacing())
        np_pred = sitk.GetArrayFromImage(itkPred)
        np_gt = sitk.GetArrayFromImage(itkLabel)
        # print(
        #     "np_pred shape:", np_pred.shape,
        #     "np_gt shape:", np_gt.shape,
        # )

        for idx in range(1, 10):
            # 原有Dice计算
            dicecomputer.Execute(itkLabel==idx, itkPred==idx)
            tmp_dice[idx-1] = dicecomputer.GetDiceCoefficient()
            
            # 新增HD95计算
            pred_mask = (np_pred == idx).astype(np.uint8)
            gt_mask = (np_gt == idx).astype(np.uint8)
            
            if np.sum(gt_mask) == 0 or np.sum(pred_mask) == 0:
                tmp_hd95[idx-1] = np.nan
            else:
                try:
                    # 获取ITK原始间距 (x,y,z顺序)
                    # original_spacing = itkCT.GetSpacing()
                    spacing = itkCT.GetSpacing()[::-1]  # 转换为(z,y,x)顺序  
                    tmp_hd95[idx-1] = hd95(
                        pred_mask, 
                        gt_mask, 
                        voxelspacing = spacing  # 使用原始物理间距
                    )
                except:
                    tmp_hd95[idx-1] = np.nan

        # 样本级指标处理
        valid_dice = tmp_dice[~np.isnan(tmp_dice)]
        sample_dice = np.mean(valid_dice) if len(valid_dice) > 0 else 0
        
        valid_hd95 = tmp_hd95[~np.isnan(tmp_hd95)]
        sample_hd95 = np.mean(valid_hd95) if len(valid_hd95) > 0 else 0

        print(f"case{i+1}-dice: {sample_dice:.4f}, HD95: {sample_hd95:.2f}mm")
        
        sample_dice_list.append(sample_dice)
        sample_hd95_list.append(sample_hd95)
        dice_list += np.nan_to_num(tmp_dice, nan=0)
        hd95_list += np.nan_to_num(tmp_hd95, nan=0)

    # 计算平均指标
    dice_list /= len(test_data_list)
    hd95_list /= len(test_data_list)
    
    return dice_list.mean(), dice_list, hd95_list.mean(), hd95_list

if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=300, type='int',
            help='number of epochs')
    parser.add_option('-b', '--batch_size', dest='batch_size', default=1,
            type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.001,
            type='float', help='learning rate')
    parser.add_option('-c', '--resume', type='str', dest='load', default=False,
            help='load pretrained model')
    parser.add_option('-p', '--checkpoint-path', type='str', dest='cp_path',
            default='/home/image/nvme/zt/code/focusnet-v2/checkpoint/', help='checkpoint path')
    parser.add_option('-o', '--log-path', type='str', dest='log_path',
            default='/home/image/nvme/zt/code/focusnet-v2/log/', help='log path')
    parser.add_option('--data_path', type='str', dest='data_path',
            default='/home/image/nvme/zt/code/focusnet-v2/data_preprocess/pddca/', help='data_path')
    parser.add_option('-m', type='str', dest='model',
            default='s_net', help='use which model')  # choose the network
    parser.add_option('-u', '--unique_name', type='str', dest='unique_name',
            default='pddca', help='use which model')
    parser.add_option('--rlt', type='float', dest='rlt',
            default=0.5, help='relation between CE/FL and dice')
    parser.add_option('--weight', type='float', dest='org_weight',
            default=[0.5,1,8,1,8,8,1,1,2,2], help='weight of focal loss')
    parser.add_option('--norm', type='str', dest='norm',
            default='bn')
    parser.add_option('--gpu', type='str', dest='gpu',
            default='3')
    

    set_determinism(seed=0)
    (options, args) = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = options.gpu
    
    print('use model:', options.model)
    print('use GPU:', options.gpu)
    print('CUDA device count:', torch.cuda.device_count())
    print('CUDA device name:', torch.cuda.get_device_name(0))
    if options.model == 's_net':
        net = s_net(1, 10, se=True, norm=options.norm)
    elif options.model == 'mda_net':
        net = MDANet(in_channel=1, out_channel=10)
    elif options.model == 'mcaNet':
        net = MCANet(in_channel=1, out_channel=10)

    elif options.model =='OAR':
        net = HeadNeckSegModel(in_channels=1, num_classes=10, base_channels=16)
    elif options.model =='hybrid_net':
        net = HybridModel()
    elif options.model =='3dunet':
        net = UNet3d(in_channel=1, out_channel=10)

    elif options.model =='base':    # M1
        net = BaseNet(1, 10)
    elif options.model =='base_wdrsc': # M2    
        net = BaseNet_wDRSC(1, 10)
    elif options.model =='base_wcssa': # M3
        net = BaseNet_wCSSA(1, 10)
    elif options.model =='base_wmsfe': # M4
        net = BaseNet_wMSFE(1, 10) 
    elif options.model =='proposed': # M5/M6
        net = Net(1, 10)
    elif options.model =='proposedv2': # M5/M6 drsc->mca
        net = Net2(1, 10)   
        
    elif options.model =='base_wse':
        net = BaseNet_wSE(1, 10)    
    elif options.model =='base_waspp':
        net = BaseNet_wASPP(1, 10)
    elif options.model =='base_wskip':
        net = BaseNet_wSkip(1, 10)
    elif options.model =='base_wdenseSkip':
        net = BaseNet_wDenseSkip(1, 10)
    elif options.model =='base_wmscam':
        net = BaseNet_wMSCAM(1, 10)
    elif options.model =='proposed2':
        net = Net2(1, 10)
    else:
        print('wrong model')

    if options.load:
        net.load_state_dict(torch.load(options.load))
        print('Model loaded from {}'.format(options.load))
    net.cuda()
    print(net)
    train_net(net, options)
    print(f"Params of Model:{sum(p.numel() for p in net.parameters()) / 1e6:.2f} M")
    print('done')

