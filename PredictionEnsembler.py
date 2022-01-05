"""
Usage: PredictionEnsembler [output filename] [prediction csvs, ordered by their importance, in descending order]

Aggregates multiple predictions into a single one, by using the majority vote system. If there are multiple top votes, the class label from the file with the least index among the top votes will be taken.
"""
import sys

import pandas as pd
import numpy as np
import numba as nb
from numba import jit, njit

@njit
def getHist(preds):
    hist = np.zeros((len(preds), 4), dtype=np.int32)
    for i in range(len(preds)):
        for j in range(preds.shape[1]):
            hist[i, preds[i, j]] += 1
    return hist

def aggregate(preds):
    out = np.zeros(len(preds), dtype=np.int32)

    for i in range(len(preds)):
        cntsAndFirstIdxs = np.zeros((4, 2), dtype=np.int32)
        maxCnt = 0
        for j in range(preds.shape[1]):
            cntsAndFirstIdxs[preds[i, j]] += 1
            if cntsAndFirstIdxs[preds[i, j], 0] == 1:
                cntsAndFirstIdxs[preds[i, j], 1] = j
            if cntsAndFirstIdxs[preds[i, j], 0] > maxCnt:
                maxCnt += 1
        out[i] = preds[i, np.min(cntsAndFirstIdxs[cntsAndFirstIdxs[:,0] == maxCnt, 1])]

    return out



if __name__ == '__main__':
    inPreds = np.stack([pd.read_csv(filename, index_col='id').to_numpy().ravel().astype(np.int32) for filename in sys.argv[2:]], axis=-1)

    ensembled = aggregate(inPreds)

    print(f"similarities of inputs with output:")
    for i, inputFilename in enumerate(sys.argv[2:]):
        similarity = (inPreds[:, i] == ensembled).sum()/inPreds.shape[0]
        print(f"{inputFilename}: {similarity}")

    outFilename = sys.argv[1]
    outDf = pd.DataFrame({'y': ensembled})
    outDf.to_csv(path_or_buf=outFilename, index_label='id')