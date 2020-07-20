import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from craft import CRAFT

from PIL import Image
import cv2 as cv
from skimage import io
import numpy as np
import craft_utils
import imgproc
import file_utils
import json
import zipfile
from collections import OrderedDict

#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# for the arguments use
def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=384, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=True, action='store_true', help='show processing time')
parser.add_argument('--test_folder', default='./test_data/eng/384/', type=str, help='folder path to input images')
parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')
parser.add_argument('--output_folder', default='./test_results/eng/', type=str, help='output path')

args = parser.parse_args()


""" For test images in a folder """
image_list, _, _ = file_utils.get_files(args.test_folder)

result_folder = args.output_folder
if not os.path.isdir(result_folder):
    os.mkdir(result_folder)

class ReaderData(object):
    def __init__(self, DataDir):
        self.RootDir = DataDir
        self.AllFiles = os.listdir(self.RootDir)
        self.ImgExt = '.jpg'
        self.JsonExt = '.json'
        # the FileNames is the name without the extension
        # a member xxx in FileNames accords two files: xxx.jpg and xxx.json
        self.FileNames = list(set([os.path.splitext(item)[0] for item in self.AllFiles]))
        self.crop_height = 384 # default
        self.crop_width = 384  # default

    def setCropParas(self, w, h):
        self.crop_width = w
        self.crop_height = h

    def loadImage(self, src_path):
        return io.imread(src_path)

    def saveCropImage(self, tgt_path, crop_img):
        cv.imwrite(tgt_path, crop_img)

    def findSize(self, img):
        return img.shape[0], img.shape[1]

    def loadFinger(self, json_dir):
        # input the json file name, return the array of finger's position
        with open (json_dir, "r") as load_f:
            finger_dict = json.load(load_f)
            index = 0
            for idx , item in enumerate(finger_dict['shapes']):
                if "finger" in item['label']:
                    index = idx
                    break
                else:
                    continue
            finger_pos = finger_dict['shapes'][index]['points'][0]
        load_f.close()
        return finger_pos

    def cropImage(self, img_path, json_path):
        t0 = time.time()
        img = self.loadImage(img_path)
        ori_h, ori_w = self.findSize(img)
        print("Loading the Original image:{}".format(time.time() - t0))

        t0 = time.time()
        fin = self.loadFinger(json_path)
        print("Loading the finger position:{}".format(time.time() - t0))

        t0 = time.time()
        margin_h = int(self.crop_height/2)
        margin_w = int(self.crop_width/2)
        crop_finger_pos_w = margin_w
        crop_finger_pos_h = margin_h

        rangeCrop = {"left":0,"right":0,"low":0,"up":0}
        # crop the image with finger at the center
        rangeCrop['left'] = fin[0] - margin_w if (fin[0] - margin_w) > 0 else 0
        rangeCrop['right'] = fin[0] + margin_w if (fin[0] + margin_w) < ori_w else ori_w
        rangeCrop['low'] = fin[1] - margin_h if fin[1] - margin_h > 0 else 0
        rangeCrop['up'] = fin[1] + margin_h if (fin[1] + margin_h) < ori_h else ori_h

        # find the new finger position in the cropped image, most of time (margin_w, margin_h)

        crop = img[rangeCrop['low'] : rangeCrop['up'], rangeCrop['left'] : rangeCrop['right']]
        crop = cv.resize(crop, (self.crop_width, self.crop_height))
        print("Croping the image:{}".format(time.time() - t0))
        return crop, [crop_finger_pos_w, crop_finger_pos_h]

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    #print("Input:{}\nInput shape:{}".format(x, x.shape))
    with torch.no_grad():
        y, feature = net(x)

    #print("Inference output:\n{}\nOutput shape:{}".format(y, y.shape))

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text

def findNearest(finger_pos, bboxes):
    def e2_distance(p1, p2):
        import math
        # p1, p2 are both in list format
        return math.sqrt(math.pow((p1[0]-p2[0]), 2) + math.pow((p1[1]-p2[1]), 2))

    """
    input:  finger's position
            a list of boxes' coordinates
    outputs: only one box's coordinates, which is the nearest one to the finger's position
    """
    min_dis = 543 # this the dialog's length of 384*384 image's which is the longest length in the image
    nearest_id = 0
    #fin_pos = np.array(finger_pos)
    for idx, item in enumerate(bboxes):
        item_center = item.mean(axis = 0)
        #print("++++{}".format(item_center))
        item_center = item_center.tolist()
        dist = e2_distance(finger_pos, item_center)
        #print("Distance:{}".format(dist))
        if dist < min_dis:
            nearest_id = idx
            min_dis = dist
    print("The nearest position id: {}".format(nearest_id))
    return nearest_id

if __name__ == '__main__':
    
    # load net
    net = CRAFT()     # initialize

    print('Loading weights from checkpoint (' + args.trained_model + ')')
    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    # LinkRefiner
    refine_net = None
    if args.refine:
        from refinenet import RefineNet
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
        if args.cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

        refine_net.eval()
        args.poly = True

    #t = time.time()

    # load data
    """
    for k, image_path in enumerate(image_list):
        #image_path = "./test_data/chi/0021_crop.jpg"
        print("Test image{:s}".format(image_path), end='\r')
        image = imgproc.loadImage(image_path)
        #print("The image 's path:{}\ttype:{}".format(image_path, type(image)))

        bboxes, polys, score_text = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, refine_net)

        # save score text
        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        mask_file = result_folder + "/res_" + filename + '_mask.jpg'
        cv.imwrite(mask_file, score_text)

        file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=result_folder)

    print("elapsed time : {}s".format(time.time() - t))
    """
    
    # test the dataset's function
    data = ReaderData('./test_data/eng/')
    #print(test_data.FileNames)
    for item in data.FileNames:
        print("-------------------{}-------------------------".format(item))
        j_path = os.path.join(data.RootDir, (item+data.JsonExt))
        i_path = os.path.join(data.RootDir, (item+data.ImgExt))
        #crop_save_path = os.path.join(data.RootDir, ('crop_'+item+data.ImgExt))
        #t0 = time.time()
        crop, finger_new_pos = data.cropImage(i_path, j_path)
        #print("Cropping:{}".format(time.time() - t0))
        #data.saveCropImage(crop_save_path, crop)
        # do the text detection:
        bboxes, polys, score_text = test_net(net, crop, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, refine_net)
        print("bboxes:\n{}, type:\n{}, \ntotal:\t{}".format(bboxes, type(bboxes), len(bboxes)))

        t0 = time.time()
        nearest_id = findNearest(finger_new_pos, bboxes)
        critical_box = bboxes[nearest_id]
        print("The critical box:{}".format(critical_box))
        min_value, max_value = critical_box.min(axis = 0), critical_box.max(axis = 0)
        print("crop paras:\t{}\t{}\t".format(min_value, max_value))
        h_0 = int(min_value[1])
        h_1 = int(max_value[1])
        w_0 = int(min_value[0])
        w_1 = int(max_value[0])
        text_crop = crop[h_0:h_1, w_0:w_1]
        print("Finding the nearest one and crop it:{}".format(time.time() - t0))
        text_area_path = 'text_area_'+ item + data.ImgExt
        io.imsave(os.path.join(result_folder, text_area_path), text_crop)
        
        mask_file = result_folder + "/res_" + item + '_mask.jpg'
        cv.imwrite(mask_file, score_text)
        file_utils.saveResult((item+data.ImgExt), crop[:,:,::-1], polys, dirname=result_folder)
    
    #print("elapsed time : {}s".format(time.time() - t))