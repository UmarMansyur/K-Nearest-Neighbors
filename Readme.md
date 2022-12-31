# K-Nearest Neighbors

## Introduction

Machine learning algorithms can be divided into three categories: supervised, unsupervised and reinforcement learning. Supervised learning is further divided into classification and regression. In classification, the goal is to predict a discrete-valued output. In regression, the goal is to predict a continuous-valued output. K-Nearest Neighbors is a simple algorithm that stores all available cases and classifies new cases based on a similarity measure (e.g., distance functions). 

KNN has been used in statistical estimation and pattern recognition already in the beginning of 1970's as a non-parametric technique. However, it was not until the machine learning boom in the beginning of 1990's that KNN became a popular algorithm for machine learning.

KNN is a type of instance-based learning, or lazy learning, where the function is only approximated locally and all computation is deferred until function evaluation. The KNN algorithm assumes that similar things exist in close proximity. In other words, similar things are near to each other.



Appropriate distance functions are used to measure the distance between two data points. The distance function is used to find the nearest neighbors of a new data point. The KNN algorithm assumes that similar things exist in close proximity. In other words, similar things are near to each other.



## Theory

### Distance functions

The most common distance functions are the Euclidean distance and the Manhattan distance. The Euclidean distance between two points $x$ and $y$ in $\mathbb{R}^D$ is defined as

$$
d(x, y) = \sqrt{\sum_{i=1}^D (x_i - y_i)^2}.
$$

Explaination of the formula:

* $x$ and $y$ are two points in $\mathbb{R}^D$

* $x_i$ and $y_i$ are the $i$-th coordinates of $x$ and $y$

* $D$ is the dimension of the space


The Manhattan distance between two points $x$ and $y$ in $\mathbb{R}^D$ is defined as

$$
d(x, y) = \sum_{i=1}^D |x_i - y_i|.
$$

## Algorithm

Given a set of training samples $\{x_i, y_i\}_{i=1}^N$, where $x_i \in \mathbb{R}^D$ is a $D$-dimensional input vector and $y_i \in \{1, \dots, K\}$ is the corresponding label (class), KNN predicts the label of a new sample $x$ by

$$
\hat{y} = \arg\max_{k=1,\dots,K} \sum_{i=1}^N I(y_i = k) I(d(x, x_i) < r),
$$

explained as follows:

* $I(\cdot)$ is the indicator function, which is equal to 1 if the statement inside the parentheses is true and 0 otherwise.

* $d(\cdot, \cdot)$ is the distance function, which measures the distance between two data points.

* $r$ is the radius of the neighborhood.

* $\hat{y}$ is the predicted label of the new sample $x$.

* $K$ is the number of classes.

* $N$ is the number of training samples.

* $D$ is the dimension of the space.

The algorithm can be summarized as follows:

1. Compute the distance between the new sample $x$ and all training samples $\{x_i\}_{i=1}^N$.

2. Find the $K$ training samples that are nearest to $x$.

3. Predict the label of $x$ by majority vote.

## Example

Let's consider the following dataset:

| x1 | x2 | Label |
| --- | --- | --- |
| 1 | 1 | 0 |
| 1 | 2 | 1 |
| 2 | 1 | 0 |
| 2 | 2 | 0 |
| 3 | 1 | 0 |
| 3 | 2 | 1 |
| 4 | 1 | 1 |
| 4 | 2 | 1 |

Let's assume that we want to predict the label of the new sample $x = (6, 3)$. The distance between $x$ and all training samples is computed as follows:

$$
d(x, x_1) = \sqrt{(6-1)^2 + (3-1)^2} = \sqrt{25} = 5
$$

$$
d(x, x_2) = \sqrt{(6-1)^2 + (3-2)^2} = \sqrt{26} = 5.1
$$

$$
d(x, x_3) = \sqrt{(6-2)^2 + (3-1)^2} = \sqrt{25} = 5
$$

$$
d(x, x_4) = \sqrt{(6-2)^2 + (3-2)^2} = \sqrt{26} = 5.1
$$

$$
d(x, x_5) = \sqrt{(6-3)^2 + (3-1)^2} = \sqrt{10} = 3.16
$$

$$
d(x, x_6) = \sqrt{(6-3)^2 + (3-2)^2} = \sqrt{13} = 3.61
$$

$$
d(x, x_7) = \sqrt{(6-4)^2 + (3-1)^2} = \sqrt{26} = 5.1
$$

$$
d(x, x_8) = \sqrt{(6-4)^2 + (3-2)^2} = \sqrt{25} = 5
$$

resulting in the following table:

| x1 | x2 | Label | Distance |
| --- | --- | --- | --- |
| 1 | 1 | 0 | 5 |
| 1 | 2 | 1 | 5.1 |
| 2 | 1 | 0 | 5 |
| 2 | 2 | 0 | 5.1 |
| 3 | 1 | 0 | 3.16 |
| 3 | 2 | 1 | 3.61 |
| 4 | 1 | 1 | 5.1 |
| 4 | 2 | 1 | 5 |

Sorting the table by the distance column, we get:

| x1 | x2 | Label | Distance |
| --- | --- | --- | --- |
| 3 | 1 | 0 | 3.16 |
| 3 | 2 | 1 | 3.61 |
| 1 | 1 | 0 | 5 |
| 2 | 1 | 0 | 5 |
| 1 | 2 | 1 | 5.1 |
| 2 | 2 | 0 | 5.1 |
| 4 | 1 | 1 | 5.1 |
| 4 | 2 | 1 | 5 |


The $K$ nearest neighbors of $x$ are $x_5$ and $x_6$. The label of $x$ is predicted by majority vote, which is 0. Therefore, the predicted label of $x$ is 0.


where $d(\cdot, \cdot)$ is the distance function, $r$ is the radius of the neighborhood and $I(\cdot)$ is the indicator function.

## Implementation

The implementation of KNN in Python is straightforward. We only need to define the distance function and the neighborhood size. Here, we use the Euclidean distance and $r=1$. The code is shown below:

```python
import pandas as pd

df = pd.read_csv('./dataset.csv', ';')
df.head()

from sklearn.model_selection import train_test_split

x= df.iloc[:, 1:3]

y = df['Label']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors= 7)
knn.fit(x_train, y_train)

y_pred = knn.predict([[6,3]])
y_pred

knn.score(x_test, y_test)
```

## Pros and Cons

### Pros

* Simple algorithm, easy to implement
* Versatile: Can be used for classification, regression and search (e.g., for image retrieval)
* No assumptions about data. 

### Cons

* Computationally expensive: Prediction stage requires computing the distance of the new sample to all training samples
* High memory requirement: Store all training samples
* Prediction stage might be slow for large $N$
* Sensitive to irrelevant features and the scale of the data
* Does not perform well with high dimensional data

## Applications

* Face recognition
* Handwriting recognition
* Image retrieval


