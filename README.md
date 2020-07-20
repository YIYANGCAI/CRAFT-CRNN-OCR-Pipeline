# OCR End2End System: CRAFT and CRNN
## Introduction
This model use a model called CRAFT to do the text detection, the recognition model is the popular CRNN.
This project is in PyTorch and Thanks for these brilliant contributions!

|  Mudule  |  Model  | Reference  |
|  :----:  | :----:  |:----:|
| Detection  | CRAFT | [CRAFT](https://github.com/clovaai/CRAFT-pytorch) |
| Recognition  | CRNN | [CRNN](https://github.com/meijieru/crnn.pytorch)|

## Requirements
|Package Name|Version|Description|
|:-:|:-:|:-:|
|PyTorch|1.3.1|Deep learning tool|
|pillow|7.1.2|Image processing|
|opencv-python|4.2.0.34|Image Processing|
|torchvision|0.4.2|Package for torch's models, data and tranforms|
If other packages are to installed, please follow the information in cmd.

## Usage
> Note that this OCR system is special because for every image a critical point is given, we detect the nearest text object from this point. In this project the point is given as a json file. The model firstly analysis this json file and get the points' coordinates. Then, the crop based on this points is generated and the CRAFT only process this crop. However, sorry that I can only provide you with the format of the data.

Firstly, put the pth model file under the ./weights/ folder.
Put json files and images files under the folder of ./test_data/ :
```
test_1.jpg
test_1.json
test_2.jpg
test_2.json
···

```
test_data
```
python test.py
```
the results are saved in ./test_results/

The CRNN can be converted to TensorRT, please refer to my another [repo](https://github.com/YIYANGCAI/CRNN-Pytorch2TensorRT-via-ONNX)

Far more things will be done for this repo.
Todo list:
> TensorRT for CRAFT
> ...
