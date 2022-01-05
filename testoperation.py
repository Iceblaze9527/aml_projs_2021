from tqdm import tqdm

import numpy as np
import torch
import cv2

def resize_to_ori(pred_activated, origin_shape):
    height = origin_shape.numpy()[0,0]
    width = origin_shape.numpy()[0,1]
    
    min_shape = min(height, width)
    padding = (int(np.floor(abs(height - width)/2)), int(np.ceil(abs(height - width)/2)))

    pred_resized = cv2.resize(pred_activated, dsize=(min_shape, min_shape), interpolation=cv2.INTER_LINEAR)##
    pred_padded = np.pad(pred_resized, pad_width=((0,0), padding)) if height < width else np.pad(pred_resized, pad_width=(padding, (0,0)))

    return pred_padded

def run(dataloader, model, ckpt_path):
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    epoch = checkpoint['epoch']
    print(f'Best model at epoch {epoch}.')
    
    model.eval()
    predictions = dict()

    with torch.no_grad():
        for idx, (X, name, index, origin_shape, output_shape) in tqdm(enumerate(dataloader), total=len(dataloader), desc='Testing'):
            X = X.float().cuda() # [N=1, IC=1, D, H*, W*], N has to be one!

            pred = model(X) # [N=1, OC=2, H**, W**], N has to be one!
            pred_activated = torch.sigmoid(pred).detach().cpu().numpy()[0,1,:,:]

            slice_idx = (pred_activated.shape[-2:] - output_shape.numpy()[0]) // 2
            if slice_idx.all() > 0:##
                pred_activated = pred_activated[slice_idx[0]:-slice_idx[0],slice_idx[1]:-slice_idx[1]]

            if name[0] not in predictions:
                predictions[name[0]] = np.zeros(*origin_shape.numpy(), dtype=np.float32)
            
            predictions[name[0]][:,:,index.numpy()[0]] = resize_to_ori(pred_activated, origin_shape)

            del X, pred
            torch.cuda.empty_cache()

    return predictions
