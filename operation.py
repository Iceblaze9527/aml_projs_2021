import time
import datetime

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from postprocessing import iou_per_frame

timestamp = lambda: time.asctime(time.localtime(time.time()))
tic = lambda: time.time()
delta_time = lambda start, end: str(datetime.timedelta(seconds=round(end-start,3)))
concat = lambda head, tail: np.concatenate((head, tail), axis=0) if head.size else tail

def train(loader, model, optim, criterion):
    model.train()
    
    losses = np.array([])
    ious = np.array([])
    pred_samples = []
    y_samples = []
    
    for idx, (X, y) in tqdm(enumerate(loader), total=len(loader), desc='Training'):
        optim.zero_grad()
        
        X = X.float().cuda() # [N, IC=1, H*, W*]
        y = y.long().cuda() # [N, H, W] with class indices (0, 1)

        pred = model(X) # [N, OC=2, H**, W**]
        slice_idx = (pred.size(dim=-1) - y.size(dim=-1)) // 2
        if slice_idx > 0:
            pred = pred[:,:,slice_idx:-slice_idx,slice_idx:-slice_idx]
        
        loss = criterion(pred, y) # [N, H, W] with reduction=none
        pred_activated = torch.sigmoid(pred).detach().cpu().numpy()[:,1,:,:]
        ious_frame = iou_per_frame(pred_activated, y.detach().cpu().numpy())
        
        losses = concat(losses, loss.detach().cpu().numpy().ravel())# mean loss
        ious = concat(ious, ious_frame)# median iou per epoch

        if idx % 2 == 0:##samples
            pred_samples.append(pred_activated[0])
            y_samples.append(y.detach().cpu().numpy()[0])

        #with torch.autograd.detect_anomaly():
        loss.mean().backward()
        optim.step()

        del X, y, pred, loss
        torch.cuda.empty_cache()
    
    return losses.ravel().mean(), np.median(ious.ravel()), pred_samples, y_samples


def evaluate(loader, model, criterion):
    model.eval()
    
    losses = np.array([])
    ious = np.array([])
    pred_samples = []
    y_samples = []
    
    with torch.no_grad():
        for idx, (X, y) in tqdm(enumerate(loader), total=len(loader), desc='Validating'):
            X = X.float().cuda() # [N, IC=1, H, W]
            y = y.long().cuda() # [N, H, W] with class indices (0, 1)

            pred = model(X) # [N, OC=2, H*, W*]
            slice_idx = (pred.size(dim=-1) - y.size(dim=-1)) // 2
            if slice_idx > 0:
                pred = pred[:,:,slice_idx:-slice_idx,slice_idx:-slice_idx]

            loss = criterion(pred, y) # [N, H, W] with reduction=none
            pred_activated = torch.sigmoid(pred).detach().cpu().numpy()[:,1,:,:]
            ious_frame = iou_per_frame(pred_activated, y.detach().cpu().numpy())
            
            losses = concat(losses, loss.detach().cpu().numpy().ravel())# mean loss
            ious = concat(ious, ious_frame)# median iou per epoch

            if idx % 2 == 0:## samples
                pred_samples.append(pred_activated[0])
                y_samples.append(y.detach().cpu().numpy()[0])

            del X, y, pred, loss
            torch.cuda.empty_cache()
    
    return losses.ravel().mean(), np.median(ious.ravel()), pred_samples, y_samples


#################


def run(dataloader, model, optim, scheduler, criterion, epochs, log_path, save_path):
    tb_writer = SummaryWriter(log_path)
    
    #TODO(2): logger module
    print('====================')
    print('start running at: ', timestamp())
    start = tic()

    train_loader, val_loader = dataloader

    max_iou = 0
    
    print('====================')
    for epoch in tqdm(range(1, epochs + 1), desc = 'Epoch'): 
        torch.cuda.synchronize()
        
        print(f'Epoch {epoch}:')
        print('start at: ', timestamp())
        epoch_start = tic()
        
        train_loss, train_iou, pred_train_samples, y_train_samples = train(loader=train_loader, model=model, optim=optim, criterion=criterion)
        val_loss, val_iou, pred_val_samples, y_val_samples = evaluate(loader=val_loader, model=model, criterion=criterion)
        
        scheduler.step()
        
        torch.cuda.synchronize()
        
        print('end at: ', timestamp())
        epoch_end = tic()     
        print('epoch runtime: ', delta_time(epoch_start, epoch_end))
        
        # TODO(2) tensorboard logger
        tb_writer.add_scalar('learning_rate', optim.param_groups[0]['lr'], global_step=epoch)
        tb_writer.add_scalars('loss', {'train_loss': train_loss, 'val_loss': val_loss}, global_step=epoch)
        tb_writer.add_scalars('median_iou', {'train_iou': train_iou, 'val_iou': val_iou}, global_step=epoch)
        
        tb_writer.add_images('train_sample_pred', np.expand_dims(np.array(pred_train_samples), axis=-1), global_step=epoch, walltime=None, dataformats='NHWC')
        tb_writer.add_images('train_sample_y', np.expand_dims(np.array(y_train_samples), axis=-1), global_step=epoch, walltime=None, dataformats='NHWC')
        tb_writer.add_images('val_sample_pred', np.expand_dims(np.array(pred_val_samples), axis=-1), global_step=epoch, walltime=None, dataformats='NHWC')
        tb_writer.add_images('val_sample_y', np.expand_dims(np.array(y_val_samples), axis=-1), global_step=epoch, walltime=None, dataformats='NHWC')
    # for name, value in model.named_parameters():
    #         tb_writer.add_histogram(name, value.data.cpu().numpy(), global_step=epoch)
    #         tb_writer.add_histogram(name + '/grad', value.grad.data.cpu().numpy(), global_step=epoch)
        
        if val_iou > max_iou:
            max_iou = val_iou
            torch.save({'epoch': epoch, 
                        'model_state_dict': model.state_dict(), 
                        'optim_state_dict': optim.state_dict(),
                        'scheduler': scheduler,
                        }, save_path)

        print('====================')
    
    print('end running at: ', timestamp())
    end = tic()
    
    print('overall runtime: ', delta_time(start, end))
    print('====================')
    
    tb_writer.close()