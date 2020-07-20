import torch
from torch.autograd import Variable
import cv2
import imgproc
from craft import CRAFT
from collections import OrderedDict
import copy
import onnxruntime
import numpy as np

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

# load net
net = CRAFT()     # initialize
net = net.cuda()
#net = torch.nn.DataParallel(net)

net.load_state_dict(copyStateDict(torch.load('./weights/craft_mlt_25k.pth')))
net.eval()

# load data
image = imgproc.loadImage('./test_data/chi/0021_crop.jpg')

# resize
img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, 384, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5)
ratio_h = ratio_w = 1 / target_ratio

# preprocessing
x = imgproc.normalizeMeanVariance(img_resized)
x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
onnx_input = x.data.numpy()
x = x.cuda()

# trace export
torch.onnx.export(net,
                  x,
                  './craft_opset10.onnx',
                  export_params=True,
                  verbose=True,
                  opset_version=10)

# test the inference process

if 1:
    session = onnxruntime.InferenceSession("./craft_opset10.onnx")
    input_name = session.get_inputs()[0].name
    print('\t>>input: {}, {}, {}'.format(session.get_inputs()[0].name, session.get_inputs()[0].shape, session.get_inputs()[0].type))
    _outputs = session.get_outputs()
    for kk in range(len(_outputs)):
        _out = _outputs[kk]
        #print('\t>>out-{}: {}, {}, {}'.format(kk, _out.name, _out.shape, _out.type))

    _x = np.array(onnx_input).astype(np.float32)

    p = session.run(None, {input_name: _x})
    out1 = p[0]
    print('============================================================================')
    print('>>summary:')
    print("onnx input:{}".format(_x))
    print('onnx out: {} \n{}'.format(np.shape(out1), out1))

