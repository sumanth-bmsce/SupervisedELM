# SupervisedELM
Implementation of Supervised Extreme Learning Machine for binary classification

## Binary Classification Examples for Synthetically Generated data

### Example 1: Linear Classification

```python
# Inporting the Supervised_ELM class, numpy and scikit learn libraries
from SELM.supervised_elm import Supervised_ELM
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
```
Create synthetic data of class 1 containing 2D random points from 0 to 50 and class 2 containing 2D random points from 60 to 100

```python
data = np.concatenate((np.random.randint(0, 50, size = (30, 2)), np.random.randint(60, 100, size = (30, 2))), axis = 0)
labels = np.concatenate((np.full(30, 0), np.full(30, 1)), axis = 0)
traindata, testdata, trainlabels, testlabels = train_test_split(data, labels, test_size = 0.1)
scaler = MinMaxScaler()
traindata = scaler.fit_transform(traindata)
testdata = scaler.fit_transform(testdata)
traindata = np.array(traindata)
testdata = np.array(testdata)
```

Create an object of the Supervised_ELM class and pass the required parameters

```python
selm = Supervised_ELM(traindata, trainlabels, "sigmoid", 5)
```
Train the model using the train() function

```python
selm.train()
```
Test the model using the test data
```python
prediction = selm.test(testdata)
# Print the accuracy of the tarined model
print("Accuracy = {}".format(accuracy_score(testlabels, prediction)))
```


