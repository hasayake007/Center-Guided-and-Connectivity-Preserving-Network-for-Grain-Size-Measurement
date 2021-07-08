from torch import optim, nn
import visdom
import logging
import warnings
from torch.utils.data import DataLoader
from grain_dataloader_multioutput import *
from cal_metric import *
from tqdm import *
from cl_dice import soft_dice_cldice, soft_dice
from model.center_guided_network import hrnet18
from pytorch_msssim import ssim, ms_ssim

warnings.filterwarnings('ignore')

batchsz = 4
lr = 1e-4
epochs = 4

device = torch.device('cuda')
torch.manual_seed(1234)
train_db = grain_Loader("dataset/train_patch/", mode='train')
val_db = grain_Loader("dataset/val/", mode='val')
test_db = grain_Loader("dataset/test/", mode='val')
train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True)
val_loader = DataLoader(val_db, batch_size=1)
test_loader = DataLoader(test_db, batch_size=1)
Loss = nn.BCELoss(reduction='mean')
cl_dice_loss = soft_dice_cldice()
viz = visdom.Visdom()


def evaluate(model, loader):
    model.eval()
    cl_dice_sum = 0
    total_mat = np.zeros([2, 2])
    i = 0
    for image, label, skeleton in loader:
        image, label, skeleton = image.to(device), label.to(device), skeleton.to(device)
        with torch.no_grad():
            pred_mask = slide_inference_multioutput(image, model)
            pred = np.array(pred_mask.detach().cpu()[0])
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            label = torch.squeeze(label).detach().cpu().numpy()
            confusion_mat, cl_dice = cal_metric(pred[0], label.astype(np.int32))
            total_mat = total_mat + confusion_mat
            cl_dice_sum = cl_dice_sum + cl_dice
            i = i + 1
    precision = total_mat[1, 1]/(total_mat[1, 1]+total_mat[0, 1])
    recall = total_mat[1, 1]/(total_mat[1, 1]+total_mat[1, 0])
    acc = np.mean(np.diag(total_mat) / total_mat.sum(axis=1))
    iou = np.mean(np.diag(total_mat) / (total_mat.sum(axis=1) + total_mat.sum(axis=0) - np.diag(total_mat)))
    return precision, recall, acc, iou, cl_dice_sum/i


def main(model, model_parameter):
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=1, min_lr=1e-6)
    best_epoch, best_iou, best_precision, best_recall, best_acc, best_cl_dice = 0, 0, 0, 0, 0, 0
    global_step = 0
    viz.line([0], [-1], win='loss_BCE', opts=dict(title='loss_BCE'))
    viz.line([0], [-1], win='loss_skeleton', opts=dict(title='loss_skeleton'))
    viz.line([0], [-1], win='loss_cl_dice', opts=dict(title='loss_cl_dice'))
    viz.line([0], [-1], win='loss_seg_ssim', opts=dict(title='loss_seg_ssim'))

    # 自动权重更新
    avg_cost = np.zeros([epochs, 24], dtype=np.float32)
    lambda_weight = np.ones([6, epochs])
    T = 2

    for epoch in tqdm(range(epochs)):
        cost = np.zeros(24, dtype=np.float32)
        # apply Dynamic Weight Average
        if epoch == 0 or epoch == 1:
            lambda_weight[:, epoch] = 1.0
        else:
            w_1 = avg_cost[epoch - 1, 0] / avg_cost[epoch - 2, 0]
            w_2 = avg_cost[epoch - 1, 3] / avg_cost[epoch - 2, 3]
            w_3 = avg_cost[epoch - 1, 6] / avg_cost[epoch - 2, 6]
            w_4 = avg_cost[epoch - 1, 9] / avg_cost[epoch - 2, 9]
            w_5 = avg_cost[epoch - 1, 12] / avg_cost[epoch - 2, 12]
            w_6 = avg_cost[epoch - 1, 15] / avg_cost[epoch - 2, 15]
            lambda_weight[0, epoch] = 6 * np.exp(w_1 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T) + np.exp(w_3 / T) + np.exp(w_4 / T)+ np.exp(w_5 / T) + np.exp(w_6 / T) )
            lambda_weight[1, epoch] = 6 * np.exp(w_2 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T) + np.exp(w_3 / T) + np.exp(w_4 / T)+ np.exp(w_5 / T) + np.exp(w_6 / T) )
            lambda_weight[2, epoch] = 6 * np.exp(w_3 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T) + np.exp(w_3 / T) + np.exp(w_4 / T)+ np.exp(w_5 / T) + np.exp(w_6 / T) )
            lambda_weight[3, epoch] = 6 * np.exp(w_4 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T) + np.exp(w_3 / T) + np.exp(w_4 / T)+ np.exp(w_5 / T) + np.exp(w_6 / T) )
            lambda_weight[4, epoch] = 6 * np.exp(w_5 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T) + np.exp(w_3 / T) + np.exp(w_4 / T) + np.exp(w_5 / T) + np.exp(w_6 / T))
            lambda_weight[5, epoch] = 6 * np.exp(w_6 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T) + np.exp(w_3 / T) + np.exp(w_4 / T)+ np.exp(w_5 / T) + np.exp(w_6 / T) )
        for step, (image, label, skeleton) in enumerate(train_loader):
            # x: [b, 3, 512, 512], y: [b, 1, 512, 512]
            image, label, skeleton = image.to(device), label.to(device), skeleton.to(device)
            model.train()
            pre_mask, pre_skeleton = model(image)
            loss_BCE = Loss(pre_mask, label)
            loss_cl_dice = cl_dice_loss(label, pre_mask)
            loss_skeleton = Loss(pre_skeleton, skeleton)
            loss_dice = soft_dice(skeleton, pre_skeleton)
            loss_seg_ssim = 1 - ms_ssim(pre_mask, label, data_range=1, size_average=True)
            loss_skl_ssim = 1 - ssim(pre_skeleton, skeleton, data_range=1, size_average=True)
            train_loss = [loss_BCE, loss_cl_dice, loss_skeleton, loss_dice, loss_seg_ssim, loss_skl_ssim]
            loss = lambda_weight[0, epoch] * loss_BCE + lambda_weight[1, epoch] * loss_cl_dice \
                   + lambda_weight[2, epoch] * loss_skeleton + lambda_weight[3, epoch] * loss_dice \
                    + lambda_weight[4, epoch] * loss_seg_ssim + lambda_weight[5, epoch] * loss_skl_ssim
            cost[0] = train_loss[0].item()
            cost[3] = train_loss[1].item()
            cost[6] = train_loss[2].item()
            cost[9] = train_loss[3].item()
            cost[12] = train_loss[4].item()
            cost[15] = train_loss[5].item()
            avg_cost[epoch, :24] += cost[:24]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([loss_BCE.item()], [global_step], win='loss_BCE', update='append')
            viz.line([loss_skeleton.item()], [global_step], win='loss_skeleton', update='append')
            viz.line([loss_cl_dice.item()], [global_step], win='loss_cl_dice', update='append')
            viz.line([loss_seg_ssim.item()], [global_step], win='loss_seg_ssim', update='append')
            global_step += 1

        if epoch % 5 == 0 and epoch > 0:
            precision, recall, acc, iou, cl_dice = evaluate(model, val_loader)
            scheduler.step(-cl_dice)
            print('epoch:', epoch, '精确率:', round(precision, 4), '召回率:', round(recall, 4), '准确率:', round(acc, 4),
                  '交并比:', round(iou, 4), 'cl_dice分数:', round(cl_dice, 4))

            if cl_dice > best_cl_dice:
                best_precision, best_recall, best_acc, best_iou, best_cl_dice = precision, recall, acc, iou, cl_dice
                print('best precision:', round(best_precision, 4), 'best recall:', round(best_recall, 4),
                      'best acc:', round(best_acc, 4), 'best iou:', round(best_iou, 4),
                      'best cl_dice:', round(best_cl_dice, 4))
                torch.save(model.state_dict(), model_parameter + '.pth')
    model.load_state_dict(torch.load(model_parameter + '.pth'))
    test_precision, test_recall, test_acc, test_iou, test_cl_dice = evaluate(model, test_loader)
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler("log.txt")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    info = model_parameter + str('BCE_Loss')+'+++' +\
           'Result:'+'pre:'+str(round(test_precision, 4)*100)+'+++'+'recall:'+str(round(test_recall, 4)*100) +\
           '+++'+'ACC:'+str(round(test_acc, 4)*100)+'+++'+'IOU:'+str(round(test_iou, 4)*100) +\
           '+++'+'cl_dice:'+str(round(test_cl_dice, 4)*100)
    logger.info(str(info))
    print(str(info))


if __name__ == '__main__':
    main(model=hrnet18(pretrained=True).cuda(), model_parameter='ours')