import numpy as np

#from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor

import cv2 as cv


_func = None

def worker_init(func):
    global _func
    _func = func
    

def worker(x):
    return _func(x)

def is_cuda_cv(): # True == using cuda, False = not using cuda
    try:
        count = cv.cuda.getCudaEnabledDeviceCount()
        if count > 0:
            return True
        else:
            return False
    except:
        return False


def denoiseVideo(video, isFramesFirst:bool = False, temporalWindowSize:int =-1, searchWindowSize=21, templateSize:int =7, h: float = 3.0):
    """
    Denoises a video using the fast non-local means method.

    The edges of the video are mirrored.

    Args:
        video:
            Array of shape (nFrames, height, width) of dtype uint8 if isFramesFirst, otherwise of shape (height, width, nFrames)
        temporalWindowSize: Number of frames around a given frame used for searching templates for averaging. Use -1 for taking the whole video, 0 for taking only the frame being denoised or any positive uneven number
        searchWindowSize: Size of the window used for computing pixel weights
        templateSize: Odd positive integer. Side-length of the patches used for template matching.
        h: Parameter regulating filter strength

    Returns:
        Video of the same shape and type as video
    """
    assert(templateSize % 2 == 1 and templateSize > 0)

    if not isFramesFirst:
        video = toFramesFirst(video)

    retArr = np.zeros_like(video)
    if temporalWindowSize == 0:
        for i in range(video.shape[0]):
            cv.fastNlMeansDenoising(video[i], dst=retArr[i], h=h, templateWindowSize=templateSize, searchWindowSize=searchWindowSize)
    elif temporalWindowSize < 0 or temporalWindowSize % 2 == 1:
        paddedVideo = np.concatenate([np.flip(video, axis=0), video, np.flip(video, axis=0)], axis=0)
        actualTempWinSize = 2*video.shape[0] - 1 if temporalWindowSize < 0 else min(temporalWindowSize, 2*video.shape[0] - 1)
        if is_cuda_cv():
            for i in range(video.shape[0], 2*video.shape[0]):
                print(f"denoising frame {i - video.shape[0]}")
                cv.fastNlMeansDenoisingMulti(paddedVideo, dst=retArr[i - video.shape[0]], imgToDenoiseIndex=i, temporalWindowSize=actualTempWinSize, searchWindowSize=searchWindowSize, templateWindowSize=templateSize, h=h)
        else:
            with ThreadPoolExecutor(max_workers=8, initializer=worker_init, initargs=(lambda i: cv.fastNlMeansDenoisingMulti(paddedVideo, imgToDenoiseIndex=i, temporalWindowSize=actualTempWinSize, searchWindowSize=searchWindowSize, templateWindowSize=templateSize, h=h), )) as p:
                retArr = np.stack(p.map(worker, range(video.shape[0], 2*video.shape[0])), axis=0 if isFramesFirst else -1)
            #with Pool(processes=None, initializer=worker_init, initargs=(lambda i: cv.fastNlMeansDenoisingMulti(paddedVideo, imgToDenoiseIndex=i, temporalWindowSize=actualTempWinSize, searchWindowSize=searchWindowSize, templateWindowSize=templateSize, h=h), )) as p:
            #    retArr = np.stack(p.map(worker, range(video.shape[0], 2*video.shape[0])), axis=0 if isFramesFirst else -1)
            return retArr
    else:
        raise ArgumentError("Temporal window size must be either -1, 0 or a positive odd integer")
    
    if isFramesFirst:
        return retArr
    else:
        return toFramesLast(retArr)


def toFramesFirst(framesLastVideo):
    return np.stack([framesLastVideo[..., i] for i in range(framesLastVideo.shape[-1])], axis=0)

def toFramesLast(framesFirstVideo):
    return np.stack(list(framesFirstVideo), axis=-1)
