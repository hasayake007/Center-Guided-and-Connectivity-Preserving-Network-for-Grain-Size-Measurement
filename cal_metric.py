from skimage.morphology import skeletonize
import numpy as np
import torch.nn.functional as F



def slide_inference_multioutput(img, model):
    """Inference by sliding-window with overlap.
    If h_crop > h_img or w_crop > w_img, the small patch will be used to
    decode without padding.
    """
    h_stride, w_stride = 256, 256
    h_crop, w_crop = 512, 512
    batch_size, _, h_img, w_img = img.size()
    num_classes = 1
    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
    preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
    count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)
            crop_img = img[:, :, y1:y2, x1:x2]
            crop_seg_logit, crop_seg_logit_128 = model(crop_img)
            crop_seg_logit_128 = F.interpolate(crop_seg_logit_128, size=(512, 512), mode='bilinear', align_corners=False)
            crop_seg_logit = (crop_seg_logit + crop_seg_logit_128)/2
            # crop_seg_logit= model(crop_img)
            preds += F.pad(crop_seg_logit, (int(x1), int(preds.shape[3] - x2), int(y1), int(preds.shape[2] - y2)))
            count_mat[:, :, y1:y2, x1:x2] += 1
    assert (count_mat == 0).sum() == 0
    preds = preds / count_mat
    preds = F.interpolate(preds, size=[h_img, w_img], mode='bilinear', align_corners=False)
    return preds


def cl_score(v, s):
    """[this function computes the skeleton volume overlap]
    Args:
        v ([bool]): [image]
        s ([bool]): [skeleton]
    Returns:
        [float]: [computed skeleton volume intersection]
    """
    return np.sum(v*s)/np.sum(s)




def get_confusion_matrix(pred_label, label, num_classes, ignore_index):
    """Intersection over Union
       Args:
           pred_label (np.ndarray): 2D predict map
           label (np.ndarray): label 2D label map
           num_classes (int): number of categories
           ignore_index (int): index ignore in evaluation
       """
    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    pred_label = pred_label.astype(np.int)
    label = label[mask]
    label = label.astype(np.int)
    n = num_classes
    inds = n * label + pred_label
    mat = np.bincount(inds, minlength=n**2).reshape(n, n)
    return mat


# This func is deprecated since it's not memory efficient
def legacy_mean_iou(results, gt_seg_maps, num_classes, ignore_index):

    return get_confusion_matrix(
        results, gt_seg_maps, num_classes, ignore_index=ignore_index)


def clDice(v_p, v_l):
    """[this function computes the cldice metric]
    Args:
        v_p ([bool]): [predicted image]
        v_l ([bool]): [ground truth image]
    Returns:
        [float]: [cldice metric]
    """
    tprec = cl_score(v_p, skeletonize(v_l))
    tsens = cl_score(v_l, skeletonize(v_p))
    return 2*tprec*tsens/(tprec+tsens)


def cal_metric(pred, gt):
    confusion_mat = legacy_mean_iou(pred, gt, 2, 255)
    cl_dice = clDice(pred, gt)
    return confusion_mat, cl_dice





