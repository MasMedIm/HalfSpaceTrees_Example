# Half Space Trees implementation using Scikit-multiflow

- Half-Space-Trees [Paper](https://www.ijcai.org/Proceedings/11/Papers/254.pdf)

# Why Half Space Trees for anomaly detection ?

- Batch learning in anomaly detection (Like isolation forest) is very expensive in memory.
- Real world data is online and continuous, thus we need a viable time series anomaly detection algorithm.
- Half Space Tree has a constant amortised time complexity and constant memory requirement.

# Dataset used

Multi-stage continuous-flow manufacturing process.
This data was taken from an actual production line near Detroit, Michigan. The goal is to predict certain properties of the line's output from the various input data. The line is a high-speed, continuous manufacturing process with parallel and series stages.

## Data Source

[Kaggle](https://www.kaggle.com/supergus/multistage-continuousflow-manufacturing-process)

# Prequisites

```BASH
pip install -U scikit-multiflow pandas sklearn
```

# Implementation

- The jupyter notebook can be found [Here](HSTrees.ipynb)

## Step by step

Imports

```PYTHON
from skmultiflow.data import DataStream
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.anomaly_detection import HalfSpaceTrees
from sklearn import preprocessing
import pandas as pd
import datetime
```

Setup Half-Space Trees estimator with default params


```PYTHON
HSTrees_model = HalfSpaceTrees(n_estimators=25, window_size=250,
                               depth=15, size_limit=50, anomaly_threshold=0.5
                               )
```

Method to format data and deleting NULLs

```PYTHON
def format_dataset(dataset):
    for col in dataset.columns:
        if dataset[col].dtype == 'object':
            le = preprocessing.LabelEncoder()
            dataset[col].fillna("Null", inplace=True)
            dataset[col] = le.fit_transform(dataset[col])
        else:
            dataset[col].fillna(0, inplace=True)
    return dataset
```

Simulating a data stream with pd.read_csv with chunks

```PYTHON
start = datetime.datetime.now()
CHUNK = 10000
detected_anomalies = 0
n_samples = 0
for data in pd.read_csv("continuous_factory_process.csv", chunksize=CHUNK):
    data = format_dataset(data)
    stream = DataStream(data)
    while stream.has_more_samples():
        X, y = stream.next_sample()
        prediction = HSTrees_model.predict(X)
        if prediction[0] == 1:
            detected_anomalies += 1
        HSTrees_model.partial_fit(X,y)
        n_samples += 1
end = datetime.datetime.now()
print(end - start)
print('Half-Space Trees detected {} anomalies out of {} samples'.format(detected_anomalies,n_samples))
```

Half-Space Trees detected 1261 anomalies out of 14088 samples


# Scikit-multiflow documentation quickstart

[Scikit-multiflow](https://scikit-multiflow.readthedocs.io/en/stable/user-guide/quick-start.html)
