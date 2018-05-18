---
layout:         post
title:          Tensor 데이터 처리하기 Part.4
subtitle:       manipulate Tensor
card-image:     http://www.aviz.fr/wiki/uploads/Research/CubixMatrixCubeExplanation.png
date:           2018-05-18 15:30:00
tags:           MachineLearning TensorFlow Coding, DataScience
post-card-type: image
---
![](http://www.aviz.fr/wiki/uploads/Research/CubixMatrixCubeExplanation.png)
### Squeeze & Expand
```python
with tf.Session() as sess:
    print(sess.run(tf.squeeze([[0], [1], [2]])))
```
벡터의 형태를 간단하게 바꾸는 더 간단한 방법도 있다. ```squeeze``` 함수를 사용하면 위의 2차원 벡터를 1차원 행렬로 만들 수 있다.
```python
[0 1 2]
```
1차원 행렬이 됐다.
```python
with tf.Session() as sess:
    print(sess.run(tf.expand_dims([0, 1, 2], 1)))
```
거꾸로 차원을 늘릴 수도 있다. ```expand_dims```로 본래 1차원이었던 행렬을 ```1```차원 늘려줄 수 있다.
```python
[[0]
 [1]
 [2]]
```
2차원 행렬이 된 것을 볼 수 있다.
### One hot
데이터를 분석하기 위한 데이터 전처리 과정에서 많이 사용하는 개념이 ```One-hot-encoding```이다. 직접 하나하나 값을 바꿔주는 방법도 있지만 ```TensorFlow```에서 이미 함수로 만들어 놓았기에 우리는 사용만 잘 하면 된다.
```python
with tf.Session() as sess:
    print(sess.run(tf.one_hot([[0], [1], [2], [0]], depth=3)))
```
2차원, ```(4, 1)```형태의 벡터를 ```one-hot```으로 만들어 달라는 코드다. ```depth = 3```의 의미는 최종적으로 나올 수 있는 결과값이 ```0 ~ 2```까지 총 세가지라는 의미다. 당연히 ```(?, 3)```의 형태로 결과값이 나올 것이다.
```python
[[[1. 0. 0.]]

 [[0. 1. 0.]]

 [[0. 0. 1.]]

 [[1. 0. 0.]]]
```
이렇게 결과 행렬이 나왔다. 우리가 원하던 것과 같이 각 해당하는 값이 각 위치에 ```1 = True``` 값으로 표시됐다. 하지만 차원이 하나 더 늘어난 것을 발견할 수 있다. 이렇게 차원이 늘어나면 데이터 처리하는데 복잡성 역시 늘어나므로 줄여주는 것이 편하다.
```python
with tf.Session() as sess:
    t = tf.one_hot([[0], [1], [2], [0]], depth=3)
    print(sess.run(tf.reshape(t, shape=[-1, 3])))
```
앞서 배웠던 ```reshape```를 사용하면 손쉽게 벡터의 형태를 바꿀 수 있다. 가장 안쪽 차원의 축 형태는 그대로 놔둔 채 총 2차원의 행렬로 형태를 조정하라는 코드다.
```python
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]
 [1. 0. 0.]]
```
이제 우리가 보기 편한 형태의 ```one-hot-encoding```으로 변환되었다.
### Casting
```python
with tf.Session() as sess:
    print(sess.run(tf.cast([1.8, 2.2, 3.3, 4.9], tf.int32)))
```
데이터 값의 형태를 바꿔주는 메소드도 있다. ```cast``` 함수를 사용하면 데이터의 형태를 바꿀 수 있다. 현재 ```float```으로 되어있는 숫자 데이터를 ```int32```로 바꿔달라는 코드다.
```python
[1 2 3 4]
```
모두 ```int32```로 데이터 값이 바뀐 것을 볼 수 있다.
```python
with tf.Session() as sess:
    print(sess.run(tf.cast([True, False, 1 == 1, 0 == 1], tf.int32)))
```
반드시 숫자뿐 아니라 ```boolean``` 값들도 사실상 ```0```과 ```1```로 표현되기 때문에 데이터 값을 바꿀 수 있다.
```python
[1 0 1 0]
```
데이터가 ```int32```로 표시되어 있다.
### Stack
```python
x = [1, 4]
y = [2, 5]
z = [3, 6]

# Pack along first dim
with tf.Session() as sess:
    print(sess.run(tf.stack([x, y, z])))
```
여러 변수로 쪼개져있는 벡터를 하나의 행렬로 쌓아주는 함수도 있다. ```stack```을 사용하면 여러 값을 하나의 벡터로 모아준다.
```python
[[1 4]
 [2 5]
 [3 6]]
```
2차원, ```(3, 2)``` 형태의 행렬이 하나 생겼다.
```python
with tf.Session() as sess:
    print(sess.run(tf.stack([x, y, z], axis=1)))
```
여느 함수와 마찬가지로 ```axis```를 조정하면 다른 형태의 행렬을 만들 수 있다.
```python
[[1 2 3]
 [4 5 6]]
```
```axis = 1```이었기 때문에 다른 형태로 만들어진 행렬이 결과물로 나왔다.
### Ones and Zeros like
가끔씩 샘플 행렬 또는 가짜 행렬을 만들어 사용해야 할 때가 온다. 보통 ```0```또는 ```1```로만 구성된 행렬을 만드는데 이때 손쉽게 사용할 수 있는 함수가 ```zeros_like```와 ```ones_like``` 함수다.
```python
x = [[0, 1, 2],
     [2, 1, 0]]

with tf.Session() as sess:
    print(sess.run(tf.ones_like(x)))
```
```(2, 3)``` 형태의 2차원 행렬인 ```x```와 같은 형태의 ```1```로만 구성된 행렬을 만드는 코드다.
```python
[[1 1 1]
 [1 1 1]]
```
```x```와 같은 형태이지만 구성 요소가 모두 ```1```인 행렬이 결과물로 나왔다.
```python
with tf.Session() as sess:
    print(sess.run(tf.zeros_like(x)))
```
```python
[[0 0 0]
 [0 0 0]]
```
마찬가지로 이번에는 ```0```으로만 구성된 행렬을 만들어 볼 수 있다.
### Zip
```zip``` 역시 데이터 전처리 과정에서 심심찮게 사용된다. 여러 요소 중 각 요소의 일치하는 순서끼리 뽑아낼 수 있다. 예시로 살펴보면 더 쉽다.
```python
for x, y in zip([1, 2, 3], [4, 5, 6]):
    print(x, y)
```
```[1, 2, 3]```과 ```[4, 5, 6]```이라는 각각의 리스트에서 각 순서에 맞는 요소들끼리 뽑아낸다.
```python
1 4
2 5
3 6
```
첫번째 요소인 ```1, 4```, 이후 ```2, 5```, ```3, 6``` 순서대로 뽑아내 출력한다.
```python
for x, y, z in zip([1, 2, 3], [4, 5, 6], [7, 8, 9]):
    print(x, y, z)
```
데이터가 더 많아져도 원리는 똑같다.
```python
1 4 7
2 5 8
3 6 9
```
역시나 각 순소에 맞게 요소를 뽑아내 출력한 결과값을 확인할 수 있다.

---