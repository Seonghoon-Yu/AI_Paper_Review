from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2 

def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA = True):
    
    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)
    
    prediction = prediction.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)
    
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]
    
    # 중심 x,y 좌표와 object confidence를 SIgmoid 합니다.
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])
    
    # 중심 offset을 추가합니다.
    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)
    
    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)
    
    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()
        
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)
    
    prediction[:,:,:2] += x_y_offset
    
    # 높이와 넓이를 log space 변환합니다.
    anchors = torch.FloatTensor(anchors)
    
    if CUDA:
        anchors = anchors.cuda()
        
    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4]) * anchors
                                    
    # class score에 sigmoid activation을 적용합니다.
    prediction[:,:,5: 5 + num_classes] = torch.sigmoid((prediction[:,:,5 : 5 + num_classes]))
    
    # detection map을 입력 이미지의 크기로 resize 합니다.
    prediction[:,:,:4] *= stride
    
    return prediction


def write_results(prediction, confidence, num_classes, nms_conf = 0.4):
    # threshold 보다 낮은 objectness score를 갖은 bounding box의 속성을 0으로 설정
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    prediction = prediction * conf_mask
    
    # bounding box의 중심 점을 좌측 상단, 우측 하단 모서리 좌표로 변환하기
    box_corner = prediction.new(prediction.shape)
    box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) 
    box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_corner[:,:,:4]
    
    batch_size = prediction.size(0)
    
    write = False
    
    # 한번에 하나의 이미지에 대하여 수행
    for ind in range(batch_size):
        image_pred = prediction[ind]
        
        # 가장 높은 값을 가진 class score를 제외하고 모두 삭제
        max_conf, max_conf_score = torch.max(image_pred[:, 5:5 + num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:,:5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)
        
        # threshold보다 낮은 object confidence를 지닌 bounding box rows를 0으로 설정한 것을 제거
        non_zero_ind =  (torch.nonzero(image_pred[:,4]))
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7)
        except:
            continue
        
        # PyToch 0.4 호환성
        # scalar가 PyTorch 0.4에서 지원되기 때문에 no detection에 대한 
        # not raise exception 코드입니다.
        if image_pred_.shape[0] == 0:
            continue 
            
        # 이미지에서 검출된 다양한 classes를 얻기
        img_classes = unique(image_pred_[:,-1] # -1 index는 class index를 지니고 있습니다.
                         
