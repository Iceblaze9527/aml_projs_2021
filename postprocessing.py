import numpy as np
import cv2
from skimage.filters import threshold_multiotsu

def post_proc(frame, connectivity=4, is_max2 = False):
    thresholds = threshold_multiotsu(frame, classes=3)
    pred_tri = np.digitize(frame, bins=thresholds)
    
    pred_1 = np.where(pred_tri==1,1,0).astype(np.uint8) 
    pred_2 = np.where(pred_tri==2,1,0).astype(np.uint8)
    
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(pred_2, connectivity=connectivity)
    
    labels[pred_1 == 1] = 0
    
    if is_max2:
        label = np.where(labels==0, False, True)
    else:
        areas = stats[:,-1]
        max2_idx = np.argsort(areas)[-3:-1]
        label = np.where(np.isin(labels, max2_idx), True, False)
    
    return label

def iou_per_frame(pred_activated, target):
    ious = []
    
    assert target.shape == pred_activated.shape
    num_frames = target.shape[0]
    
    for idx in range(num_frames):
        pred = post_proc(pred_activated[idx], is_max2 = False)
        
        overlap = pred * target[idx]
        union = pred + target[idx]

        ious.append(overlap.sum()/float(union.sum()))

    return np.array(ious).ravel()