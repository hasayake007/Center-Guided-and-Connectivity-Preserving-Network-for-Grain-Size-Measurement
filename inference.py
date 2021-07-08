from PIL import Image
from torchvision import transforms
from cal_metric import *
from model.center_guided_network import hrnet18
import cv2
from torch.autograd import Variable
import os
import torch
from skimage.measure import label, regionprops
import time

as_tensor = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def segment(read_pth, model):
    img = cv2.imread(read_pth, -1)
    img = Image.fromarray(img)
    model.eval()
    with torch.no_grad():
        images = as_tensor(img)
        tensor = Variable(torch.unsqueeze(images, dim=0).float(), requires_grad=False).cuda()
        pred_mask = slide_inference_multioutput(tensor, model)
        pred = np.array(pred_mask.detach().cpu()[0])
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
    return pred[0]


def count_grade(mask, tri_circle, Name, draw=True):
    final_result = tri_circle * mask
    labeled_img, number_inter_point = label(final_result, connectivity=2, background=0, return_num=True)
    grade = np.log10(100 * (number_inter_point / 490)) * 6.643856 - 3.288  # 输出最后的晶粒度等级
    if draw:
        properties = regionprops(labeled_img)
        Image = cv2.imread(Name, cv2.COLOR_GRAY2RGB)
        # alpha 为第一张图片的透明度
        alpha = 1
        # beta 为第二张图片的透明度
        beta = 0.2
        gamma = 0
        seg_result = np.stack([np.zeros_like(mask), np.zeros_like(mask), mask*255], axis=2)
        Image = cv2.addWeighted(Image, alpha, seg_result.astype(np.uint8), beta, gamma)
        uu, vv = Image.shape[:2]
        length_input = Image.shape[0]
        Image = cv2.circle(img=Image, center=(int(vv / 2), int(uu / 2)), radius=int(length_input / 2 * 0.325), color=1,
                           thickness=1)
        Image = cv2.circle(img=Image, center=(int(vv / 2), int(uu / 2)), radius=int(length_input / 2 * 0.650), color=1,
                           thickness=1)
        Image = cv2.circle(img=Image, center=(int(vv / 2), int(uu / 2)), radius=int(length_input / 2 * 0.975), color=1,
                           thickness=1)
        for pro in properties:
            # Image = cv2.drawMarker(img=Image, color=(0, 0, 255), position=(int(pro.centroid[1]), int(pro.centroid[0])),
            #                        markerType=1, markerSize=40)
            Image = cv2.circle(img=Image, color=(144, 238, 144), center=(int(pro.centroid[1]), int(pro.centroid[0])),
                               radius=10, thickness=-1)
        result_name = './dataset/result/'+imageName[0:-4]+'_'+str(number_inter_point)+'_'+str(round(grade, 2)) + '.png'
        cv2.imwrite(result_name, Image)
    return grade


def generate_cicle(mask, thickness):
    length = mask.shape[0]
    tri_circle = np.zeros((length, length))
    u, v = tri_circle.shape[:2]
    tri_circle = cv2.circle(img=tri_circle, center=(int(v / 2), int(u / 2)), radius=int(length / 2 * 0.325), color=1,
                            thickness=thickness)
    tri_circle = cv2.circle(img=tri_circle, center=(int(v / 2), int(u / 2)), radius=int(length / 2 * 0.650), color=1,
                            thickness=thickness)
    tri_circle = cv2.circle(img=tri_circle, center=(int(v / 2), int(u / 2)), radius=int(length / 2 * 0.975), color=1,
                            thickness=thickness)
    return tri_circle


if __name__ == '__main__':
    model = hrnet18(pretrained=False).cuda()
    model.load_state_dict(torch.load('./weight/center_guide_network_C+S.pth'), strict=True)
    img_dir = './dataset/SRIF-GSM-B/'
    img_list = os.listdir(img_dir)
    for imageName in img_list:
        if imageName.endswith('.JPG') or imageName.endswith('.jpg') or imageName.endswith('.png'):
            mask = segment(img_dir+imageName, model)
            tri_circle = generate_cicle(mask, thickness=4)
            grade = count_grade(mask, tri_circle, img_dir+imageName, draw=True)

