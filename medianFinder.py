import pandas as pd
import numpy as np

#filenames = ["y_test_yutong_v1.csv", "y_test_yutong_v2.csv", "y_test_yutong_v3.csv", "y_test_yutong_v4.csv", "y_test_yutong_v5.csv", "y_test_yutong_v6.csv", "y_test_yutong_v7.csv", "y_test_yutong_v8.csv", "y_test_yutong_v9.csv"]
filenames = ["y_test_yutong_v4.csv", "y_test_yutong_v5.csv", "y_test_yutong_v6.csv", "y_test_yutong_v7.csv", "y_test_yutong_v8.csv", "y_test_yutong_v9.csv"]

values = np.stack([pd.read_csv(filename).values[:, 1] for filename in filenames])
print(values.shape)
print(np.median(values, axis=0).shape)

ensembledValues = np.median(values, axis=0)

outDf = pd.DataFrame({'y': ensembledValues})
outDf.to_csv("ensembledValues.csv", index_label="id")
