from torch.utils.tensorboard import SummaryWriter
import os, glob, losses, utils
import losses as losseses
import sys
# sys.path.append(r'D:/code/TransMorph_Transformer_for_Medical_Image_Registration-main/utils')
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
from natsort import natsorted
from models.TransMorph import CONFIGS as CONFIGS_TM
import models.TransMorph as TransMorph

from datagenerators import Dataset
import torch.utils.data as Data
from metric import MSE
from Functions import generate_grid, transform_unit_flow_to_flow_cuda
import torch.nn.functional as F
from utilses.losses import neg_Jdet_loss
import logging
import time
from utilses.config import get_args

args = get_args()
from utilses.losses import NCC as NCC_new
from utilses.scheduler import StopCriterion
from utilses.utilize import save_model


def make_dirs():
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)


# class Logger(object):
#     def __init__(self, save_dir):
#         self.terminal = sys.stdout
#         self.log = open(save_dir+"logfile.log", "a")
#
#     def write(self, message):
#         self.terminal.write(message)
#         self.log.write(message)
#
#     def flush(self):
#         pass


def generate_grid(imgshape):
    x = np.arange(imgshape[0])
    y = np.arange(imgshape[1])
    z = np.arange(imgshape[2])
    grid = np.rollaxis(np.array(np.meshgrid(z, y, x)), 0, 4)
    grid = np.swapaxes(grid, 0, 2)
    grid = np.swapaxes(grid, 1, 2)
    return grid


def JacboianDet(y_pred, sample_grid):
    J = y_pred + sample_grid
    dy = J[:, 1:, :-1, :-1, :] - J[:, :-1, :-1, :-1, :]
    dx = J[:, :-1, 1:, :-1, :] - J[:, :-1, :-1, :-1, :]
    dz = J[:, :-1, :-1, 1:, :] - J[:, :-1, :-1, :-1, :]

    Jdet0 = dx[:, :, :, :, 0] * (dy[:, :, :, :, 1] * dz[:, :, :, :, 2] - dy[:, :, :, :, 2] * dz[:, :, :, :, 1])
    Jdet1 = dx[:, :, :, :, 1] * (dy[:, :, :, :, 0] * dz[:, :, :, :, 2] - dy[:, :, :, :, 2] * dz[:, :, :, :, 0])
    Jdet2 = dx[:, :, :, :, 2] * (dy[:, :, :, :, 0] * dz[:, :, :, :, 1] - dy[:, :, :, :, 1] * dz[:, :, :, :, 0])

    Jdet = Jdet0 - Jdet1 + Jdet2

    return Jdet


def neg_Jdet_loss(y_pred, sample_grid):
    neg_Jdet = -1.0 * JacboianDet(y_pred, sample_grid)
    selected_neg_Jdet = F.relu(neg_Jdet)

    return torch.mean(selected_neg_Jdet)


def main():
    batch_size = 1
    # train_dir = 'D:/DATA/JHUBrain/Train/'
    # val_dir = 'D:/DATA/JHUBrain/Val/'
    train_dir = 'D:/datasets/train_val_test/train_160_small'

    train_fixed_folder = os.path.join(train_dir, 'fixed')
    train_moving_folder = os.path.join(train_dir, 'moving')
    train_f_img_file_list = sorted(
        [os.path.join(train_fixed_folder, file_name) for file_name in os.listdir(train_fixed_folder) if
         file_name.lower().endswith('.gz')])
    train_m_img_file_list = sorted(
        [os.path.join(train_moving_folder, file_name) for file_name in os.listdir(train_moving_folder) if
         file_name.lower().endswith('.gz')])

    val_dir = 'D:\datasets/train_val_test/val_160_small'
    val_fixed_folder = os.path.join(val_dir, 'fixed')
    val_moving_folder = os.path.join(val_dir, 'moving')
    val_f_img_file_list = sorted(
        [os.path.join(val_fixed_folder, file_name) for file_name in os.listdir(val_fixed_folder) if
         file_name.lower().endswith('.gz')])
    val_m_img_file_list = sorted(
        [os.path.join(val_moving_folder, file_name) for file_name in os.listdir(val_moving_folder) if
         file_name.lower().endswith('.gz')])

    weights = [1, 0.02]  # loss weights

    # save_dir = 'TransMorph_mse_{}_diffusion_{}/'.format(weights[0], weights[1])
    # save_dir = 'Model/stage'
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    #
    # if not os.path.exists(save_dir+'/logs'):
    #     os.makedirs(save_dir+'/logs')
    # sys.stdout = Logger(save_dir+'/logs')  #输出信息记录在一个特定的日志文件中，而不是直接输出到控制台
    # sys.stdout = Logger(save_dir)

    lr = 0.00001  # learning rate
    epoch_start = 0
    max_epoch = 500  # max traning epoch
    cont_training = False  # if continue training

    image_loss_func_NCC = NCC_new(win=args.win_size)
    loss_Jdet = neg_Jdet_loss
    imgshape = (160, 160, 160)
    grid = generate_grid(imgshape)
    grid = torch.from_numpy(np.reshape(grid, (1,)

                                       + grid.shape)).to(device).float()

    '''
    Initialize model
    '''
    config = CONFIGS_TM['TransMorph']
    model = TransMorph.TransMorph(config)
    model.cuda()

    '''
    Initialize spatial transformation function
    '''
    reg_model = utils.register_model(config.img_size, 'nearest')
    reg_model.cuda()
    reg_model_bilin = utils.register_model(config.img_size, 'bilinear')
    reg_model_bilin.cuda()

    '''
    Initialize training 
    '''

    """
    这段代码主要定义训练数据集的数据预处理方式的。具体使用transform.Compose()将多个数据预处理操作组合在一起，构成一个数据预处理管道。该管道包括两个部分：
    ①trans.RandomFlip
    自定义的数据增强操作，对输入数据进行随机翻转.
    其中 0 参数表示不进行翻转的概率为 0，即一定进行翻转操作。
    如果想要指定不进行翻转的概率，可以将 0 参数替换为一个介于 0 和 1 之间的实数。

    ②trans.NumpyType
    一个自定义的数据预处理操作，用来将输入数据的数据类型转换为指定的数据类型。
    其中 (np.float32, np.float32) 参数表示将输入数据的第一个通道和第二个通道分别转换为 np.float32 类型。
    如果输入数据只有一个通道，可以将参数改为 (np.float32,)。
    ------------------------------------------------------------------------
    需要注意的是，这段代码只是定义了训练数据集的数据预处理方式，
    并没有对数据进行实际的处理。如果想要对数据进行预处理，还需要将这个数据预处理管道传递给对应的数据集对象，
    例如 datasets.JHUBrainDataset。
    """
    # train_composed = transforms.Compose([trans.RandomFlip(0),  #表示进行随机翻转操作，其中参数 0 表示水平和垂直翻转的概率都为 0，即不进行翻转操作。因此，trans.RandomFlip(0) 的结果是不进行翻转操作后的输入数据。
    #                                      trans.NumpyType((np.float32, np.float32)),  #trans.NumpyType() 函数接受一个元组作为参数，因此需要使用 (np.float32, np.float32) 表示包含两个 np.float32 类型数组的元组。
    #                                      ])
    """
    Seg_norm()：
    将图像中的像素值进行标准化处理，以便更好地适应神经网络模型的训练要求。
    此外，还会将标签数据中的像素值重新排列，从而转换为 1 到 46 之间的整数值，以便更好地与模型进行匹配。

    NumpyType((np.float32, np.int16))：
    将图像和标签数据的数据类型分别转换为 np.float32 和 np.int16。
    这一步操作可以确保输入数据的数据类型与模型中定义的数据类型相同，从而提高模型的训练效率和准确性。
    """
    # val_composed = transforms.Compose([trans.Seg_norm(), #rearrange segmentation label to 1 to 46
    #                                    trans.NumpyType((np.float32, np.int16)),
    #                                     ])

    """
    在这段代码中，glob.glob(train_dir + '*.pkl') 返回一个包含所有训练数据文件路径的列表，其中 train_dir 是训练数据文件所在的目录路径，'*.pkl' 表示只选择扩展名为 .pkl 的文件。因此，glob.glob(train_dir + '*.pkl') 的结果是一个包含多个训练数据文件路径的列表。
    然后，使用 datasets.JHUBrainDataset 类的构造函数，将训练数据文件路径列表和数据增强操作序列传递给该类，创建了一个名为 train_set 的训练数据集对象。
    最终，该训练数据集对象可用于加载 JHU Brain 数据集中的训练数据，并应用数据增强操作序列对数据进行增强，以便更好地适应神经网络模型的训练要求。
    """
    # train_set = datasets.JHUBrainDataset(glob.glob(train_dir + '*.pkl'), transforms=train_composed)
    # val_set = datasets.JHUBrainInferDataset(glob.glob(val_dir + '*.pkl'), transforms=val_composed)
    train_dataset = Dataset(moving_files=train_m_img_file_list, fixed_files=train_f_img_file_list)
    val_dataset = Dataset(moving_files=val_m_img_file_list, fixed_files=val_f_img_file_list)
    """
    num_workers：用于数据加载的线程数。通过增加线程数，可以并行读取数据，提高数据读取效率。通常情况下，建议将 num_workers 设置为 CPU 核数的一半左右。
    pin_memory：是否将数据加载到 GPU 的固定内存中。将数据加载到固定内存中可以加速数据读取，因为 GPU 可以直接读取固定内存中的数据，而不需要将数据从主机内存复制到 GPU 内存。但是，固定内存的容量有限，如果数据集过大，可能会导致内存溢出。因此，建议在内存充足的情况下开启该选项。
    drop_last：是否丢弃最后一个批次的数据。如果数据集的大小不能被批次大小整除，那么最后一个批次的数据量会小于批次大小。如果设置 drop_last=True，则会丢弃最后一个批次的数据，否则会保留最后一个批次的数据。
    """
    # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    # val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
    train_loader = Data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    val_loader = Data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    """
    optim.Adam 是一种自适应学习率的优化算法，它可以自动调整每个参数的学习率，以适应不同参数的梯度变化情况。
    params：要优化的参数列表，例如 model.parameters()，表示优化模型的所有参数。
    lr：学习率，用于控制每次参数更新的步长。学习率越大，参数更新的步长越大，模型的收敛速度越快，但可能会导致模型振荡或发散；学习率越小，参数更新的步长越小，模型的收敛速度越慢，但可能会得到更稳定的模型。
    weight_decay：权重衰减系数，用于控制参数的正则化强度。权重衰减可以防止模型过拟合，通过对参数的大小进行惩罚来约束模型的复杂度。通常情况下，建议将权重衰减设置为一个较小的值，例如 1e-4。
    amsgrad：是否使用 AMSGrad 变种。AMSGrad 是 Adam 算法的一种变种，它可以防止学习率下降过快，从而提高模型的收敛速度和稳定性。通常情况下，建议开启该选项。

    在训练过程中，可以通过调用 optimizer.step() 方法来更新参数，通过调用 optimizer.zero_grad() 方法来清除参数的梯度。
    """
    # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0, amsgrad=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = image_loss_func_NCC
    criterions = [criterion]
    criterions += [losseses.Grad3d(
        penalty='l2')]  # criterions 被定义为一个列表，首先包含了一个均方误差损失函数 nn.MSELoss()，然后又添加了一个基于梯度的正则化损失函数 losses.Grad3d()。这个列表中的损失函数将在模型训练过程中被同时使用，以帮助模型学习更好的特征表示和更稳定的模型。
    best_MSE = 0
    stop_criterion = StopCriterion()
    """
    这段代码使用 PyTorch 中的 SummaryWriter 类来创建一个 TensorBoard 日志文件，用于可视化模型训练过程中的各种信息。
    具体来说，SummaryWriter 是 PyTorch 提供的一个用于创建 TensorBoard 日志文件的工具类，它可以将模型训练过程中的各种信息（例如损失函数、准确率、梯度等）写入日志文件中，并使用 TensorBoard 可视化工具进行展示和分析。使用 SummaryWriter 可以帮助我们更好地理解模型的学习过程，找到模型训练中存在的问题，并进行调整和优化。
    在这段代码中，SummaryWriter 的初始化函数接受一个参数 log_dir，用于指定日志文件的保存路径。在这里，日志文件将保存在 'logs/' + save_dir 目录下，其中 save_dir 是保存模型参数的目录名。这样做可以方便我们在训练过程中同时保存模型参数和日志文件，并进行统一管理。
    在模型训练过程中，可以使用 writer.add_scalar() 方法将各种信息写入日志文件中，例如 writer.add_scalar('Loss/train', loss.item(), global_step=step) 表示将训练集的损失函数值 loss.item() 写入名为 'Loss/train' 的标量图中，并指定当前的训练步数 global_step=step。然后我们可以在 TensorBoard 中查看名为 'Loss/train' 的标量图，以了解模型训练过程中损失函数的变化情况。
    """
    writer = SummaryWriter(log_dir='logs/')
    '''
       If continue from previous training
       '''
    if cont_training:
        checkpoint = r'D:\TransMorph_Transformer_for_Medical_Image_Registration_main\TransMorph\model\2023-05-07-15-12-52_TM__133_1.0887best.pth'
        model.load_state_dict(torch.load(checkpoint)['model'])
        optimizer.load_state_dict(torch.load(checkpoint)['optimizer'])

    # writer = SummaryWriter(log_dir = save_dir)
    best_loss = 99.
    for epoch in range(epoch_start, max_epoch):
        print('Training Starts')

        '''
        Training
        '''
        loss_all = utils.AverageMeter()
        idx = 0
        loss_total = []
        # for data in train_loader:
        for batch, (moving, fixed) in enumerate(train_loader):
            idx += 1
            model.train()
            adjust_learning_rate(optimizer, epoch, max_epoch, lr)
            # data = [t.cuda() for t in data]
            # x = data[0]
            # y = data[1]
            x = moving[0].to(device).float()
            y = fixed[0].to(device).float()

            x_in = torch.cat((x, y), dim=1)
            output = model(x_in)
            loss = 0
            loss_vals = []
            for n, loss_function in enumerate(criterions):
                curr_loss = loss_function(output[n], y) * weights[n]
                loss_vals.append(curr_loss)
                loss += curr_loss

            # print("F_X_Y_norm:  ",F_X_Y_norm)
            # print("grad:  ",grid)
            # loss_Jacobian = loss_Jdet(output[1].permute(0, 2, 3, 4, 1), grid)
            # print("loss_Jacobian: ",loss_Jacobian)

            # loss_all.update(loss.item(), y.numel())
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            del x_in
            del output

            # flip fixed and moving images
            loss = 0
            x_in = torch.cat((y, x), dim=1)
            output = model(x_in)
            for n, loss_function in enumerate(criterions):
                curr_loss = loss_function(output[n], x) * weights[n]
                loss_vals[n] += curr_loss
                loss += curr_loss
            loss_all.update(loss.item(), y.numel())
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print('Iter {} of {} loss {:.4f}, Img Sim: {:.6f}, Reg: {:.6f}'.format(idx, len(train_loader), loss.item(), loss_vals[0].item()/2, loss_vals[1].item()/2))
            sys.stdout.write(
                "\r" + 'epoch:"{0}"step:batch "{1}:{2}" -> training loss "{3:.6f}" - sim "{4:.6f}"  -reg "{5:.6f}"'.format(
                    epoch, idx, len(train_loader), loss.item(), loss_vals[0].item() / 2, loss_vals[1].item() / 2))
            sys.stdout.flush()

            logging.info("img_name:{}".format(moving[1][0]))
            logging.info("TM, epoch: %d  iter: %d batch: %d  loss: %.4f  sim: %.4f  grad: %.4f" % (
                epoch, idx, len(train_loader), loss.item(), loss_vals[0].item() / 2, loss_vals[1].item() / 2))

        writer.add_scalar('Loss/train', loss_all.avg, epoch)
        print('Train: Epoch {} loss {:.4f}'.format(epoch, loss_all.avg))

        '''
        Validation
        '''

        val_Loss = []
        # eval_DSC = utils.AverageMeter()
        with torch.no_grad():
            # for data in val_loader:
            losses = []
            for batch, (moving, fixed) in enumerate(val_loader):
                model.eval()
                # data = [t.cuda() for t in data]
                # x = data[0]
                # y = data[1]
                x = moving[0].to('cuda').float()
                y = fixed[0].to('cuda').float()

                # x_seg = data[2]
                # y_seg = data[3]
                x_in = torch.cat((x, y), dim=1)
                # grid_img = mk_grid_img(8, 1, config.img_size)
                output = model(x_in)  # [warped,DVF]

                # loss_Jacobian = loss_Jdet(output[1].permute(0, 2, 3, 4, 1), grid)
                loss_Jacobian = criterions[1](output[1], y)
                ncc_loss_ori = image_loss_func_NCC(output[0], y)

                mse_loss = MSE(output[0], y)
                loss_sum = ncc_loss_ori + weights[1] * loss_Jacobian
                losses.append([ncc_loss_ori.item(), mse_loss.item(), loss_Jacobian.item(), loss_sum.item()])

                """
                这段代码使用另一个空间变换模型 reg_model_bilin 对一个网格图像 grid_img 进行空间变换操作，并将变换后的结果赋值给变量 def_grid。
                具体地，代码将 grid_img 和 output[1]（表示光流场信息）分别转移到 GPU 上，并将它们作为输入传递给 reg_model_bilin 模型。reg_model_bilin 模型会根据给定的光流场信息对输入的网格图像进行空间变换操作，从而得到变换后的网格图像。
                需要注意的是，这里使用的是另一个空间变换模型 reg_model_bilin，而不是之前提到的 reg_model。reg_model_bilin 模型与 reg_model 模型的主要区别是，在进行空间变换操作时，reg_model_bilin 模型使用的是双线性插值方法，而 reg_model 模型使用的是最近邻插值方法。双线性插值方法可以产生更平滑的变换结果，但计算代价也更高。
                """
                # def_out = reg_model([x_seg.cuda().float(), output[1].cuda()])
                # def_grid = reg_model_bilin([grid_img.float(), output[1].cuda()])
                # dsc = utils.dice_val(def_out.long(), y_seg.long(), 46)
                # eval_dsc.update(dsc.item(), x.size(0))

                # mse_loss = MSE(output[0], x)

                # ncc_loss_ori=loss_similarity(output[0], x)
                # loss_Jacobian = neg_Jdet_loss(flow.permute(0, 2, 3, 4, 1), grid)
                # val_Loss[0]=torch.mean(torch.mean(val_Loss[0], dim=0))#range (expected to be in range of [-1, 0], but got 1)
                # eval_MSE.update(val_Loss[0].item(), x.size(0))
                #
                # print('Iter {} of {} eval_MSE.avg {:.4f}'.format(batch,len(val_loader),eval_MSE.avg))

            mean_loss = np.mean(losses, 0)
        val_ncc_loss, val_mse_loss, val_jac_loss, val_total_loss = mean_loss
        if val_total_loss <= best_loss:
            best_loss = val_total_loss
            # modelname = model_dir + '/' + model_name + "{:.4f}_stagelvl3_".format(best_loss) + str(step) + '.pth'
            modelname = model_dir + '/' + model_name + '_{:03d}_'.format(epoch) + '{:.4f}best.pth'.format(
                val_total_loss)
            logging.info("save model:{}".format(modelname))
            save_model(modelname, model, stop_criterion.total_loss_list, stop_criterion.ncc_loss_list,
                       stop_criterion.jac_loss_list, stop_criterion.train_loss_list, optimizer)
        else:
            modelname = model_dir + '/' + model_name + '_{:03d}_'.format(epoch) + '{:.4f}.pth'.format(
                val_total_loss)
            logging.info("save model:{}".format(modelname))
            save_model(modelname, model, stop_criterion.total_loss_list, stop_criterion.ncc_loss_list,
                       stop_criterion.jac_loss_list, stop_criterion.train_loss_list, optimizer)

        # mean_loss = np.mean(np.array(loss_total), 0)
        print(
            "\n one epoch pass. train loss %.4f . val ncc loss %.4f . val mse loss %.4f . val_jac_loss %.6f . val_total loss %.4f" % (
                loss_all.avg, val_ncc_loss, val_mse_loss, val_jac_loss, val_total_loss))

        # best_MSE = max(eval_MSE.avg, best_MSE)
        # if eval_MSE.avg <= best_MSE:
        #     eval_MSE.avg = best_MSE
        #     modelname = model_dir + '/' + model_name + '_{:03d}_'.format(epoch+1) + '{:.4f}best.pth'.format(eval_MSE.avg)
        #     logging.info("save model:{}".format(modelname))
        #     save_checkpoint({
        #     'epoch': epoch + 1,
        #     'state_dict': model.state_dict(),
        #     'best_MSE': best_MSE,
        #     'optimizer': optimizer.state_dict(),
        # }, save_dir=model_dir, filename='MSE{:.3f}.pth.tar'.format(eval_MSE.avg))
        #     writer.add_scalar('MSE/validate============', eval_MSE.avg, epoch)
        # else:
        #     modelname = model_dir + '/' + model_name + '_{:03d}_'.format(epoch + 1) + '{:.4f}best.pth'.format(
        #         eval_MSE.avg)
        #     logging.info("save model:{}".format(modelname))
        #     save_checkpoint({
        #         'epoch': epoch + 1,
        #         'state_dict': model.state_dict(),
        #         'best_MSE': best_MSE,
        #         'optimizer': optimizer.state_dict(),
        #     }, save_dir=model_dir, filename='MSE{:.3f}.pth.tar'.format(eval_MSE.avg))
        #     writer.add_scalar('MSE/validate============', eval_MSE.avg, epoch)

        # best_dsc = max(eval_dsc.avg, best_dsc)
        # save_checkpoint({
        #     'epoch': epoch + 1,
        #     'state_dict': model.state_dict(),
        #     'best_dsc': best_dsc,
        #     'optimizer': optimizer.state_dict(),
        # }, save_dir='experiments/' + save_dir, filename='dsc{:.3f}.pth.tar'.format(eval_dsc.avg))
        # writer.add_scalar('DSC/validate', eval_dsc.avg, epoch)
        # plt.switch_backend('agg')
        # pred_fig = comput_fig(def_out)
        # grid_fig = comput_fig(def_grid)
        # x_fig = comput_fig(x_seg)
        # tar_fig = comput_fig(y_seg)
        # writer.add_figure('Grid', grid_fig, epoch)
        # plt.close(grid_fig)
        # writer.add_figure('input', x_fig, epoch)
        # plt.close(x_fig)
        # writer.add_figure('ground truth', tar_fig, epoch)
        # plt.close(tar_fig)
        # writer.add_figure('prediction', pred_fig, epoch)
        # plt.close(pred_fig)
        loss_all.reset()
    writer.close()


def comput_fig(img):
    img = img.detach().cpu().numpy()[0, 0, 48:64, :, :]
    fig = plt.figure(figsize=(12, 12), dpi=180)
    for i in range(img.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.imshow(img[i, :, :], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig


def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power(1 - (epoch) / MAX_EPOCHES, power), 8)


def mk_grid_img(grid_step, line_thickness=1, grid_sz=(160, 192, 224)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j + line_thickness - 1, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i + line_thickness - 1] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img


def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=8):
    torch.save(state, save_dir + filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))


if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()  # GPU_num=1

    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):  # GPU_idx=0
        GPU_name = torch.cuda.get_device_name(GPU_idx)  # GPU_name='NVIDIA GeForce RTX 3060'
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)  # 单卡  使用指定的卡
    GPU_avai = torch.cuda.is_available()  # GPU_avai=true
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))

    device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')

    model_dir = 'model'
    train_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    model_name = "{}_TM_".format(train_time)
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    make_dirs()

    log_index = len([file for file in os.listdir(args.log_dir) if file.endswith('.txt')])
    logging.basicConfig(level=logging.INFO,
                        filename=f'Log/log{log_index}.txt',
                        filemode='a',
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

    main()
