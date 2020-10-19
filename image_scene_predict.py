#coding: utf-8

import torch
from torch.autograd import Variable
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os,sys
import numpy as np
import cv2,glob
from PIL import Image
import matplotlib.pyplot as plt


def load_model(num_classes=20):

    classes = ['bathroom', 'bedroom', 'childs_room', 'closet', 'coffee_shop', 'dining_hall', 'dining_room',
			   'dressing_room', 'formal_garden', 'home_office', 'japanese_garden', 'kitchen', 'living_room',
			   'parking_lot', 'restaurant_kitchen', 'restaurant_patio', 'storage_room', 'tea_room',
			   'television_room', 'television_studio']  ##20-classes
    # model_file = './model/resnet18_zucao_class20.pth.tar'


    current_path = os.path.abspath(__file__)
    father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
    model_file =  os.path.join(father_path,'./model/resnet18_zucao_class20.pth.tar')

    print(model_file)

    if not os.path.exists(model_file):
        print(" NotFound:{}".format(model_file))
        sys.exit(0)

    model = models.resnet18(num_classes=num_classes)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model.to(device)
    model.eval()

    return model,classes


def load_place365():
    num_classes = 365


    # model_file = './model/resnet18_places365.pth.tar'
    # file_name_category = './model/categories_places365.txt'


    current_path = os.path.abspath(__file__)
    father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")

    model_file = os.path.join(father_path, './model/resnet18_places365.pth.tar')
    file_name_category = os.path.join(father_path, './model/categories_places365.txt')


    if not os.path.exists(model_file):
        print(" NotFound:{}".format(model_file))
        sys.exit(0)

    classes = list()
    with open(file_name_category) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
            # print(line)


    model = models.resnet18(num_classes=num_classes)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model.to(device)
    model.eval()

    return model,classes


MODEL, MODEL_CLASSES = load_place365()
# MODEL, MODEL_CLASSES = load_model()

def image_predict(img):

    # load the image transformer
    transformer = trn.Compose([trn.Resize((256, 256)), trn.CenterCrop(224), trn.ToTensor(),
							   trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load the model
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model,classes = load_model()


    # get the softmax weight
    # params = list(MODEL.parameters())
    # weight_softmax = params[-2].data.cpu().numpy()
    # weight_softmax[weight_softmax < 0] = 0


    img = transformer(img)
    # img = img.to(device)
    input_img = img.unsqueeze(0)

    # forward pass
    logit = MODEL.forward(input_img)
    h_x = F.softmax(logit, 1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.numpy()
    idx = idx.numpy()

    ### 挑选 top 5
    top_num = 5 if len(MODEL_CLASSES) > 5 else len(MODEL_CLASSES)
    pred_scores = probs.tolist()[:top_num]
    pred_classes = [MODEL_CLASSES[i] for i in idx.tolist()][:top_num]

    results = {
        "scores": pred_scores,### 预测分值列表，[0.2, ...]
        "pred_classes": pred_classes,### 预测物体编码列表，['kitchen', ...]
    }

    # print(dict([results['pred_classes'][i], score] for i,score in enumerate(results['scores'] ) if i< top_num ))
    # print(results)

    return results

def test():
    test_img_path = "../demo.jpg"
    img = Image.open(test_img_path).convert('RGB')
    results = image_predict(img)

    print("{}:\n {}".format(test_img_path,results))

if __name__ == "__main__":
    test()