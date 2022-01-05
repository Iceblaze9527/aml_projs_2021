import os
import pickle
import gzip
import time
import itertools

from multiprocessing import Pool, cpu_count

import numpy as np
import matplotlib.pyplot as plt

import cv2 as cv

from numba import njit, jit

import sklearn

import tensorflow as tf


from InvertedMap import *

from ReverseRemapping import *

try:
    from sklearnex import patch_sklearn
    patch_sklearn()
    print("sklearnex is installed. Patched sklearn to use the optimized algorithms.")
except ImportError or ModuleNotFoundError:
    print("sklearnex (intel extension for accelerating sklearn) is not installed, not using it.")


def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = pickle.load(f)
        return loaded_object

def dump_zipped(filename, obj):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f)


def knnClassifyTfSingleBatch(k, trainData, trainLabels, dataToPredict):
    """
    Method for KNN classification using tensorflow (on GPU if possible), which is faster than sklearn.
    Uses L2 norm.

    Args:
        k: number of neighbors
        trainData: tensor or array of shape (numSamples, vectorLen)
        trainLabels: tensor or array of shape (numSamplesToPredict, vectorLen)
    """
    if not isinstance(trainData, tf.Tensor):
        trainData = tf.convert_to_tensor(trainData)
    if not isinstance(trainLabels, tf.Tensor):
        trainLabels = tf.convert_to_tensor(trainLabels)
    distances = tf.stack([tf.norm(trainData - dataVec, axis=-1) for dataVec in dataToPredict], axis=0)

    if k == 1:
        indices = tf.argmin(distances, axis=1)
        return tf.gather(trainLabels, indices)
    
    weightedAvgs = tf.zeros(dataToPredict.shape)
    totalWeight = tf.zeros(dataToPredict.shape)
    BIG_VALUE = 10000
    for neighIdx in range(k):
        indices = tf.argmin(distances, axis=1)
        flatIdxs = tf.range(0, distances.shape[0])*distances.shape[1] + indices
        #indicesNd = tf.stack([tf.range(0, dataToPredict.shape[0]), indices], axis=0)
        #weights = tf.gather_nd(distances, indicesNd)
        weights = tf.gather(distances, indices, axis=1)
        weightedAvgs += weights * tf.gather(trainLabels, indices)
        totalWeight += weights
        
        numpDists = distances.numpy()
        numpDists.flat[flatIdxs] = BIG_VALUE
        distances = tf.convert_to_tensor(numpDists)

    weightedAvgs /= totalWeight
    return (weightedAvgs >= 0.5).numpy()

def knnClassifyTf(k, trainData, trainLabels, dataToPredict, maxBatchSize = 1024):
    retArr = np.zeros(dataToPredict.shape[0], dtype=np.bool)
    for batchStart in range(0, dataToPredict.shape[0], maxBatchSize):
        print(f"processing labels {batchStart} - {batchStart + maxBatchSize} of {dataToPredict.shape[0]}")
        retArr[batchStart:batchStart + maxBatchSize] = knnClassifyTfSingleBatch(k=k, trainData=trainData, trainLabels=trainLabels, dataToPredict=dataToPredict[batchStart:batchStart + maxBatchSize])
    return retArr

def calcOptFlowForAllFrames(video, optFlowClass: cv.DenseOpticalFlow):
    """
    Note: the returned array is of shape (nFrames-1, nrows, ncols)
    """
    outArr = np.zeros((video.shape[-1] - 1, *video.shape[:-1], 2), dtype=np.float32)
    for i in range(video.shape[-1] - 1):
        #outArr[i] = cv.calcOpticalFlowFarneback(video[i], video[i+1], None, 0.5, 3, 15, 3, 5, 1.2, 0) #optFlowClass.calc(video[i], video[i + 1], None)
        outArr[i] = optFlowClass.calc(video[..., i], video[..., i + 1], None)
    return outArr


def calcOptFlowForAllFrames_timestepsFirst(video, optFlowClass: cv.DenseOpticalFlow):
    """
    this function expects the video to be of shape (nFrames, height, width)
    Note: the returned array is of shape (nFrames-1, nrows, ncols)
    """
    outArr = np.zeros((video.shape[0] - 1, *video.shape[1:], 2), dtype=np.float32)
    for i in range(video.shape[0] - 1):
        #outArr[i] = cv.calcOpticalFlowFarneback(video[i], video[i+1], None, 0.5, 3, 15, 3, 5, 1.2, 0) #optFlowClass.calc(video[i], video[i + 1], None)
        outArr[i] = optFlowClass.calc(video[i], video[i + 1], None)
    return outArr



@njit
def optFlowToPointCloud(optFlow):
    pointCloud = np.zeros((optFlow.shape[0]*optFlow.shape[1], 2), dtype=np.float32)

    for i in range(optFlow.shape[0]):
        for j in range(optFlow.shape[1]):
            pointCloud[i*j, 0] = i + optFlow[i, j, 0]
            pointCloud[i*j, 1] = j + optFlow[i, j, 1]
    return pointCloud

@njit
def mappedPointsToPointCloudGrid(optFlow, labels, tileSize=10):
    grid = [[[] for j in range((optFlow.shape[1] + 1)//tileSize)] for i in range((optFlow.shape[0]+1)//tileSize)]

    for i in range(optFlow.shape[0]):
        for j in range(optFlow.shape[1]):
            x = j + optFlow[i, j, 1]
            y = i + optFlow[i, j, 0]
            tileX = x//tileSize
            tileY = y//tileSize
            if tileX >= 0 and tileX < len(grid[0]) and tileY >= 0 and tileY < len(grid):
                grid[tileY, tileX].append([y, x])



@njit
def createCoordinateGridPoints(nrows, ncols, dtype=np.float32):
    gridPoints = np.zeros((nrows * ncols, 2), dtype=dtype)
    for i in range(nrows):
        for j in range(ncols):
            gridPoints[i*nrows + j, 0] = i
            gridPoints[i*nrows + j, 1] = j
    return gridPoints

def createCoordinateGrid(nrows, ncols, dtype=np.float32):
    gridPoints = np.zeros((nrows, ncols, 2), dtype=dtype)
    @njit
    def internal():
        for i in range(nrows):
            for j in range(ncols):
                gridPoints[i, j, 0] = i
                gridPoints[i, j, 1] = j
    return gridPoints

def createMapForRemapping(optFlow):
    inverseMap = createCoordinateGrid(*optFlow.shape) + optFlow
    inverseMapX, inverseMapY = inverseMap[:, :, 1], inverseMap[:, :, 0]

    realMapX, realMapY = invert_map(inverseMapX, inverseMapY, diagnostics=True)
    return realMapX, realMapY

def warpLabelWithOptFlow(optFlow, labelImg, referenceLabelImg = None, newImage=None, enforceSamePositiveSize=True):
    """
    Constructs a label image from labelImg where the pixels are interpolated from the pixels in labelImg that are shifted according to the optical flow optFlow.
    """
    ##mapX, mapY = createMapForRemapping(optFlow)
    ##return cv.remap(labelImg, mapX, mapY, cv.INTER_LINEAR, borderMode=cv.BORDER_TRANSPARENT)
    #pointCloud = optFlowToPointCloud(optFlow)
    #labels = labelImg.flatten()
    ##knnClf = sklearn.neighbors.KNeighborsClassifier(n_neighbors=4, n_jobs=8)
    ##knnClf = knnClf.fit(pointCloud, labels)
    #if coordinateGridPoints is None:
    #    coordinateGridPoints = createCoordinateGridPoints(*optFlow.shape[:-1])
    ##predLabels = knnClf.predict(coordinateGridPoints)
    #predLabels = knnClassifyTf(1, pointCloud, labels, coordinateGridPoints)
    floatLabelImg = 2*labelImg.astype(np.float32) - np.ones(labelImg.shape, dtype=np.float32) #convert labels to float image with True = 1 and False = -1
    floatPredImg = reverseRemap(optFlow, floatLabelImg)
    #return predLabels.reshape(optFlow.shape[:-1])

    #plt.hist(floatPredImg.flatten(), bins=40)
    #plt.show()

    if enforceSamePositiveSize:
        if newImage is not None and referenceLabelImg is not None:
            #scores = (floatPredImg - floatLabelImg.min())*newImage
            scores = scoresFromPredImg(floatPredImg, labelImg, referenceLabelImg, newImage)
        else:
            scores = floatPredImg
        thresh = np.sort(scores, axis=None)[-(labelImg.sum()) - 1]
        print(f"thresh = {thresh}")
        #plt.hist(scores.flatten(), bins=40, range=(10, scores.max()))
        #plt.show()

        return scores > thresh

    return floatPredImg > 0

def inferFollowingLabels(optFlows, labelImg, images=None):
    """
    Iteratively applies optical flow to the labelImg in order to move them in the same way as the origninal image does.
    Note: the returned label array is of shape (nFrames, nImgRows, nImgCols), not of shape (nImgRows, nImgCols, nFrames), as the original labels.

    Args:
        optFlows: A numpy array of shape (nFrames, nImgRows, nImgCols, 2), where optFlows[0] encodes the movements of the pixels in labelImg that lead to the first inferred image; optFlows[1] applied to this inferred image yields the next inferred image and so on.
        labelImg: A numpy array of shape (nImgRows, nImgCols), containing the segmentation at a reference positions
    """
    retArr = np.zeros(optFlows.shape[:-1], dtype=labelImg.dtype)
    last = labelImg
    for i in range(optFlows.shape[0]):
        print(f"start warping frame {i}")
        image = None if images is None else images[i+1]
        retArr[i] = warpLabelWithOptFlow(optFlows[i], last, referenceLabelImg=labelImg, newImage=image)
        print(f"finished warping frame {i}")
        last = retArr[i]
    return retArr

def inferFollowingProbImgs(optFlows, refLabelImg, images, refDistArr = None):
    """
    Infer all labels following refLabelImg.

    Args:
        optFlows: optical flows; one for each frame from the  refLabelImg to the end, excluding the labelImg. Shape: (nFrames-1, nRows, nCols, 2)
        refLabelImg: reference label image to infer the other labels from. Shape: (nRows, nCols)
        images: images from the one corresponding to the refLabelImg to the end, including the one corresponding to the refLabelImg. Shape: (nFrames, nRows, nCols)

        returns:
            Probability images (where each pixel corresponds to the estimated probability that the corresponding pixel is 1). Shape: (nFrames-1, nRows, nCols)
    """
    retArr = np.zeros((optFlows.shape[0], *refLabelImg.shape), dtype=np.float32)
    if refDistArr is None:
        refDistArr = getDistanceMat(refLabelImg, numIterations=100)
    lastLbl = refLabelImg
    numPosPixels = refLabelImg.sum()

    for i in range(optFlows.shape[0]):
        floatLabelImg = 2*lastLbl.astype(np.float32) - np.ones(lastLbl.shape, dtype=np.float32) #convert labels to float image with True = 1 and False = -1
        floatPredImg = reverseRemap(optFlows[i], floatLabelImg)
        scores = scoresFromPredImgAndRefDistArr(floatPredImg, lastLbl, refDistArr, images[i+1])
        sortedScores = np.sort(scores, axis=None)
        thresh = sortedScores[-numPosPixels]
        lastLbl = scores >= thresh
        
        retArr[i] = scores
        #thresh *= 0.75 #set lower threshold for probability map, because positive areas tend to be too small because of averaging probabilities
        retArr[i].flat[scores.flatten() < thresh] /= 2*thresh #such that all pixels below the threshold have probability < 0.5
        retArr[i].flat[scores.flatten() >= thresh] = 0.5 + (scores.flat[scores.flatten() >= thresh] - thresh)/(2*(scores.max() - thresh)) #linearily increasing probability with score

    return retArr

def inferAllProbImgs(optFlows, refLabelIdx, refLabelImg, images, refDistArr = None):
    retArr = np.zeros(images.shape, dtype=np.float32)
    if refLabelIdx > 0:
        retArr[:refLabelIdx] = inferFollowingProbImgs(-optFlows[:refLabelIdx][::-1], refLabelImg, images[:refLabelIdx+1][::-1], refDistArr=refDistArr)[::-1]
    retArr[refLabelIdx] = refLabelImg
    if refLabelIdx < images.shape[0] - 1:
        retArr[refLabelIdx+1:] = inferFollowingProbImgs(optFlows[refLabelIdx:], refLabelImg, images[refLabelIdx:], refDistArr=refDistArr)
    return retArr



def labelImgPostProc(inferredLabels, maxDistForConnecting=15, minDistForRemovingChunk=15, minChunkArea=200, minLargeChunkArea=500, dontTouchFrameIdxs=[]): #don't connect two chunks that are both > minLargeChunkArea
    newLabels = inferredLabels.copy()
    lastFrameDistMap = np.ones(inferredLabels.shape[1:], dtype=np.float32)
    #averagedDistMap =  np.ones(inferredLabels.shape[1:], dtype=np.float32)
    for frameIdx, lblFrame in enumerate(inferredLabels):
        print(f"post-processing label {frameIdx}")
        if frameIdx not in dontTouchFrameIdxs:
            numConnComps, labelledComps, stats, centroids = cv.connectedComponentsWithStats(lblFrame.astype(np.uint8), connectivity=8)
            bigConnComps = [i for i in range(1,numConnComps) if stats[i, cv.CC_STAT_AREA] > minChunkArea]
            bigConnCompsImg = labelledComps == bigConnComps[0]
            for i in bigConnComps[1:]:
                bigConnCompsImg |= labelledComps == i
            bigConnCompsDistMap = getDistanceMat(bigConnCompsImg, numIterations=max(minDistForRemovingChunk, maxDistForConnecting))
            
            remainingConnComps = [i for i in range(1, numConnComps)]
            for i in range(1, numConnComps):
                cX, cY = centroids[i]
                if stats[i, cv.CC_STAT_AREA] < minChunkArea:
                    if bigConnCompsDistMap[int(cY), int(cX)] >= minDistForRemovingChunk:
                        newLabels[frameIdx] &= ~(labelledComps == i) #remove small components that are far away from any big components
                        remainingConnComps.remove(i)
                        continue
                if frameIdx > 0 and lastFrameDistMap[int(cY), int(cX)] > minDistForRemovingChunk:
                    newLabels[frameIdx] &= ~(labelledComps == i) #remove components that appeared out of nowhere
                    remainingConnComps.remove(i)
                    continue
            print(f"connComps sizes: {sorted([stats[i, cv.CC_STAT_AREA] for i in range(1, numConnComps)], reverse=True)}")
            #connect scattered components
            if len(remainingConnComps) > 2:
                distMaps = {connComp: getDistanceMat(labelledComps == connComp, numIterations=maxDistForConnecting) for connComp in remainingConnComps}
                gradsX = {connComp: cv.Sobel(distMaps[connComp], ddepth=-1, dx=1, dy=0, ksize=3) for connComp in remainingConnComps}
                gradsY = {connComp: cv.Sobel(distMaps[connComp], ddepth=-1, dx=0, dy=1, ksize=3) for connComp in remainingConnComps}
                for connComp1, connComp2 in itertools.combinations(remainingConnComps, 2):
                    if stats[connComp1, cv.CC_STAT_AREA] < minLargeChunkArea or stats[connComp2, cv.CC_STAT_AREA] < minLargeChunkArea:
                        distMap = np.empty_like(distMaps[connComp1])
                        distMap.fill(np.inf)
                        grad1X = gradsX[connComp1] #cv.Sobel(distMaps[connComp1], ddepth=-1, dx=1, dy=0, ksize=3)
                        grad1Y = gradsY[connComp1] #cv.Sobel(distMaps[connComp1], ddepth=-1, dx=0, dy=1, ksize=3)
                        grad2X = gradsX[connComp2] #cv.Sobel(distMaps[connComp2], ddepth=-1, dx=1, dy=0, ksize=3)
                        grad2Y = gradsY[connComp2] #cv.Sobel(distMaps[connComp2], ddepth=-1, dx=0, dy=1, ksize=3)
                        newLabels[frameIdx] |= (np.maximum(distMaps[connComp1], distMaps[connComp2], out=distMap, where=((grad1X*grad2X + grad1Y*grad2Y < 0))) < maxDistForConnecting) #a problem with this method is that it sometimes creates little chunks between two connected components; do a second round of removing small stuff
                numConnComps, labelledComps, stats, centroids = cv.connectedComponentsWithStats(lblFrame.astype(np.uint8), connectivity=8)
                for i in range(1, numConnComps):
                    cX, cY = centroids[i]
                    if stats[i, cv.CC_STAT_AREA] < minChunkArea:
                        newLabels[frameIdx] &= ~(labelledComps == i) #remove small components that are far away from any big components

        lastFrameDistMap = getDistanceMat(newLabels[frameIdx])
    return newLabels

def inferAllLabels(sample, returnPreprocessedVideo = False, usePostProcessing=True):
    video = sample["video"]
    preprocessedVideo = np.stack([cv.threshold(cv.blur(video[..., i], ksize=(5, 5)), 5, 255, cv.THRESH_TOZERO)[1] for i in range(video.shape[-1])], axis=0)
    optFlowAlg = cv.DISOpticalFlow_create(preset=cv.DISOPTICAL_FLOW_PRESET_MEDIUM)
    optFlowAlg.setPatchSize(16)
    optFlowAlg.setUseMeanNormalization(False)
    optFlowAlg.setGradientDescentIterations(50)
    optFlows = calcOptFlowForAllFrames_timestepsFirst(preprocessedVideo, optFlowAlg)

    refLabels = {refLblIdx: sample["label"][:, :, refLblIdx] for refLblIdx in sample["frames"]}
    refDistArr = np.ones(preprocessedVideo.shape[1:], dtype=np.float32)
    for refLblImg in refLabels.values():
        refDistArr *= getDistanceMat(refLblImg)
    refDistArr[~sample["box"]] = (refDistArr.shape[0]*refDistArr.shape[1])**2 #set distances outside the region of interest to some high value
    allProbImgs = {refLblIdx: inferAllProbImgs(optFlows, refLblIdx, refLabels[refLblIdx], preprocessedVideo, refDistArr=refDistArr) for refLblIdx in sample["frames"]}

    probImgWeight = lambda lblIdx, refLblIdx: 1 - abs(lblIdx - refLblIdx)/(preprocessedVideo.shape[0] + 1)

    inferredLabels = np.zeros(preprocessedVideo.shape, dtype=np.bool)

    for i in range(preprocessedVideo.shape[0]):
        totWeight = 0
        totProb = np.zeros(preprocessedVideo.shape[1:], dtype=np.float32)
        for refLblIdx, probImgs in allProbImgs.items():
            probImg = probImgs[i]
            if i == refLblIdx:
                totProb = probImg
                totWeight = 1
                break
            else:
                weight = probImgWeight(i, refLblIdx)
                totProb += weight * probImg
                totWeight += weight
        totProb /= totWeight
        #inferredLabels[i] = totProb >= 0.5
        inferredLabels[i] = totProb >= (1/len(refLabels.keys()))

    if usePostProcessing:
        inferredLabels = labelImgPostProc(inferredLabels, dontTouchFrameIdxs=refLabels.keys())

    if returnPreprocessedVideo:
        return inferredLabels, preprocessedVideo

    return inferredLabels

def inferLabelsForWholeDataset(dataset, numThreads=-1):
    if numThreads == -1:
        numThreads = cpu_count()
    
    allLabels = None
    with Pool(processes=numThreads) as pool:
        allLabels = list(pool.map(inferAllLabels, dataset))

    if allLabels == None:
        print(f"warning: could not predict labels", file=sys.stderr)

    return allLabels

def predictLabelsAndStoreNewDataset(dataset, outputFilename, predLabelKey="inferredLabel"):
    predLabels = inferLabelsForWholeDataset(dataset)
    print(predLabels)
    
    for sample, predLabel in zip(dataset, predLabels):
        sample[predLabelKey] = predLabel
    
    dump_zipped(outputFilename, dataset)


def markPoints(img, points):
    mask = np.zeros_like(img)
    for point in points:
        cv.circle(img, point, 8, (0, 0, 255), cv.LINE_8)

if __name__ == "__main__":
    #train_data = load_zipped_pickle("train.pkl")
    #
    #print(f"number of samples in whole dataset: {len(train_data)}")
    #expertDataset = [sample for sample in train_data if sample["dataset"] == "expert"]
    #print(f"number of samples in expert dataset: {len(expertDataset)}")

    #predictLabelsAndStoreNewDataset(expertDataset, outputFilename="expertTrainDsWithInferredLabels.pkl")

    expertDsWithInferredLabels = load_zipped_pickle("expertTrainDsWithInferredLabels.pkl")

#    testSample = expertDataset[1]
#    #dump_zipped("testSample.pkl", testSample)
#    #testSample = load_zipped_pickle("testSample.pkl")
#
#    print(f"annotated frames: {testSample['frames']}")
#
#    labels, preprocVid = inferAllLabels(testSample, returnPreprocessedVideo=True)
#
#    #testVideo = testSample["video"]
#    #testLabelIdx = testSample["frames"][1]
#    #testLabel = testSample["label"][:, :, testLabelIdx]
#
#    ##denoisedTestVideo = np.stack([cv.fastNlMeansDenoising(testVideo[..., i]) for i in range(testVideo.shape[-1])], axis=0)
#    #denoisedTestVideo = np.stack([cv.threshold(cv.blur(testVideo[..., i], ksize=(5, 5)), 40, 255, cv.THRESH_TOZERO)[1] for i in range(testVideo.shape[-1])], axis=0)
#
#    ##optFlow = calcOptFlowForAllFrames(testVideo, cv.DISOpticalFlow_create(preset=cv.DISOpticalFlow_PRESET_MEDIUM))
#    #optFlowAlg = cv.DISOpticalFlow_create(preset=cv.DISOPTICAL_FLOW_PRESET_MEDIUM)
#    #optFlowAlg.setPatchSize(16)
#    #optFlowAlg.setUseMeanNormalization(False)
#    #optFlowAlg.setGradientDescentIterations(50)
#    #print(f"optFlow settings: variationalRefinementAlpha: {optFlowAlg.getVariationalRefinementAlpha()}, useMeanNormalization: {optFlowAlg.getUseMeanNormalization()}, variationalRefinementDelta: {optFlowAlg.getVariationalRefinementDelta()}, variationalRefinementGamma: {optFlowAlg.getVariationalRefinementGamma()}, variationalRefinementIterations: {optFlowAlg.getVariationalRefinementIterations()}, gradientDescentIterations: {optFlowAlg.getGradientDescentIterations()}")
#    #optFlow = calcOptFlowForAllFrames_timestepsFirst(denoisedTestVideo, optFlowAlg)
#
#    #followingLabels = inferFollowingLabels(optFlow[testLabelIdx:], testLabel, images=denoisedTestVideo[testLabelIdx:])
#
#    #labels = np.concatenate((np.expand_dims(testLabel, axis=0), followingLabels), axis=0)
    testVideo = expertDsWithInferredLabels[0]["video"]
    labels = expertDsWithInferredLabels[0]["inferredLabel"]
    for i in range(labels.shape[0]):
        #cv.imshow("label", labels[i].astype(np.uint8)*255)
        #cv.imshow("origImage", testVideo[:, :, testLabelIdx+i])
        img = cv.cvtColor(testVideo[:, :, i], cv.COLOR_GRAY2BGR)
        #img = cv.cvtColor(denoisedTestVideo[testLabelIdx+i], cv.COLOR_GRAY2BGR)
        #img = cv.cvtColor(preprocVid[i], cv.COLOR_GRAY2BGR)
        img[:,:, 1] = labels[i].astype(np.uint8)*255
        #pointsToTrack = cv.goodFeaturesToTrack(denoisedTestVideo[testLabelIdx+i], -1, qualityLevel=0.01, minDistance=10)
        #print(f"pointsToTrack: {pointsToTrack}")
        #markPoints(img, pointsToTrack)
        cv.imshow("overlaid", img)
        #cv.imwrite(f"overlaid_{i}.png", img)
        #cv.imwrite(f"label_{i}.png", labels[i].astype(np.uint8)*255)
        k = cv.waitKey(40) & 0xff
        if k == 27:
            break
#
#    #testVideo = expertDataset[0]["video"]
#    #np.save("testVideo.npy", testVideo)
#
#    #testVideo = np.load("testVideo.npy", allow_pickle=True)
#    #
#    #optFlow = calcOptFlowForAllFrames(testVideo, cv.DISOpticalFlow_create(preset=cv.DISOpticalFlow_PRESET_MEDIUM))
#
#
#    #optFlow = calcOptFlowForAllFrames(testVideo, cv.optflow.createOptFlow_DeepFlow())
#    #optFlowMagnitudes = np.linalg.norm(optFlow, axis=-1)
#    #optFlowMagnitudes -= optFlowMagnitudes.min()
#    #optFlowMagnitudes /= optFlowMagnitudes.max()
#
#    #hsv = np.zeros((*optFlow.shape[1:-1], 3), dtype=np.uint8)
#    #hsv[:, :, 1] = 255
#    #for i in range(len(optFlow)):
#    #    flow = optFlow[i]
#    #    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
#    #    hsv[..., 0] = ang*180/np.pi/2
#    #    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
#    #    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
#    #
#    #    cv.imshow("optFlow", bgr)
#    #    cv.imshow("orig", testVideo[..., i])
#    #    k = cv.waitKey(30) & 0xff
#    #    if k == 27:
#    #        break
#    #    #cv.imwrite(f"optFlow_frame_{i}.png", bgr)
#    #    #cv.imwrite(f"orig_frame_{i}.png", testVideo[i])
#
#    #testVideo = train_data[0]["video"]
#    #print(testVideo[len(testVideo)//2])
#    #
#    #print(testVideo[len(testVideo)//2].sum())
#    #
#    #cv.namedWindow("test", cv.WINDOW_AUTOSIZE)
#    #for i in range(len(testVideo)):
#    #    cv.imwrite(f"frame_{i}.png", testVideo[i])
#    #    #cv.imshow("test", np.expand_dims(testVideo[i], -1))
#    #    #time.sleep(0.05)
#    ##cv.waitKey(0)
#    #cv.destroyAllWindows()
#
