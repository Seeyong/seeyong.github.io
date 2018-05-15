---
layout:         post
title:          Tensor 처리하기 #1
subtitle:       manipulate Tensor
card-image:     http://cfile27.uf.tistory.com/image/221A8C4F527B96CA0AA464
date:           2018-05-15 16:00:00
tags:           ML
post-card-type: image
---
# Tensor manipulation
더욱 깊은 이론으로 들어가기 전에 기본으로 돌아가보자. ```Tensor```를 손쉽게 다룰 수 있어야 이후 과정이 편해진다. 사실 기본이다.
### Simple ID Array and Slicing
![](http://cfile27.uf.tistory.com/image/221A8C4F527B96CA0AA464)
```Array```는 조각으로 자른 김밥과 비슷하다. 순서대로 나열되어 있는 데이터 집합이다. 다만 순서의 시작이 ```1```이 아닌 ```0```부터 시작한다는 점이 차이점이다.
```python
import numpy as np
import pprint as pp

# one dimension array
t = np.array([0., 1., 2., 3., 4., 5., 6.])
pp.pprint(t)
print(t.ndim) # rank
print(t.shape) # shape
```
1차원의 ```Array```를 만든 후 ```ndim```과 ```shape``` 메소드로 각각 ```차원```과 ```형태```를 알아볼 수 있다.
```python
array([0., 1., 2., 3., 4., 5., 6.])
1
(7,)
```
위 ```Array```는 ```1차원```의 ```(7)```이라는 형태를 가지고 있다. 하나의 차원에 7개의 요소를 가지고 있다는 의미다.
```python
# slice
print(t[0], t[1], t[-1])
print(t[2:5], t[4:-1])
print(t[:2], t[3:])
```
더불어 기본적인 파이썬과 같이 ```Slicing```도 가능하다.
```python
0.0 1.0 6.0
[2. 3. 4.] [4. 5.]
[0. 1.] [3. 4. 5. 6.]
```
하나하나의 요소를 가지고 올 수도 있고, 특정 범위를 잘라낼 수도 있다.
### 2D Arrary
```python
t = np.array([[1.,2.,3.], [4.,5.,6.], [7.,8.,9.], [10.,11.,12.]])
pp.pprint(t)
print(t.ndim) # rank
print(t.shape) # shape
```
이번에는 조금 더 복잡해진 ```Array```를 살펴보자. 몇 차원의 어떤 형태를 가지고 있을까.
```python
array([[ 1.,  2.,  3.],
       [ 4.,  5.,  6.],
       [ 7.,  8.,  9.],
       [10., 11., 12.]])
2
(4, 3)
```
먼저 차원은 ```2```다. 꼼수처럼 보일 수 있지만 맨 앞의 ```[```, 즉 대괄호 숫자를 세면 차원을 알아낼 수 있다. 2개의 대괄호를 가지고 있으므로 ```2차원 Array```다. 차원이 ```2```라는 뜻은 형태를 나타내는 숫자도 2개라는 것이다. 형태는 ```(?, ?)```로 표현되는데, ```?```의 개수는 차원의 개수와 같다.

가장 기본적인, 깊숙히 들어가있는 최소 단위의 데이터 집합을 먼저 봐야 한다. ```element```가 ```3개``` 들어있는 리스트를 ```4개``` 발견할 수 있다. ```Array```의 형태를 뒤에서부터 적어가면 된다. 따라서 ```(4, 3)```의 형태로 표현된다.
### Shape, Rank, Axis
```python
import tensorflow as tf

t = tf.constant([1,2,3,4])
print(t.get_shape())
```
이제 실제로 ```Tensor```를 구현해보자. 다행히 ```get.shape()``` 메소드를 사용해 형태를 바로 알아낼 수 있다.
```python
(4,)
```
일단 대괄호가 하나이므로 차원은 ```1```이고, 4개의 요소가 들어있는 데이터 뭉치가 하나만 있으므로 형태는 ```(4)```로 표현할 수 있다.
```python
t = tf.constant([[1,2],
               [3,4]])
print(t.get_shape())
```
조금 더 복잡한 형태라도 마찬가지로 적용해보면,
```python
(2, 2)
```
```(2, 2)```라는 형태를 찾아낼 수 있다.
```python
t = tf.constant([[[[1,2,3,4], [5,6,7,8], [9,10,11,12]],
                [[13,14,15,16], [17,18,19,20], [21,22,23,24]]]])
print(t.get_shape())
```
갑자기 중간 없이 너무 복잡해졌다. 하지만 기본적인 법칙은 똑같다. 일단 너무 다닥다닥 붙어있으니 풀어서 보자.
```python
[ 
    [ 
        [ 
            [1,2,3,4], 
            [5,6,7,8],
            [9,10,11,12]
        ],
        [
            [13,14,15,16],
            [17,18,19,20],
            [21,22,23,24]
        ]
    ]
]
```
일단 앞에 있는 대괄호의 숫자는 총 4개이므로 ```4차원 Array```다. 따라서 ```Array```의 형태는 ```(?, ?, ?, ?)```로 표현된다. 가장 깊숙한 데이터 집합을 보자. 총 4개의 숫자가 하나의 묶음으로 묶여있다. 따라서 형태의 가장 마지막 물음표는 ```4```가 되고, 형태는 ```(?, ?, ?, 4)```로 표현된다.

4개의 데이터가 묶여있는 또 하나의 집합은 3개씩 묶여있다. 살펴보니 그 뭉치가 총 2개다. 따라서 형태는 ```(?, 2, 3, 4)```인 것을 알 수 있다. 마지막으로 가장 큰 데이터 뭉치가 하나이므로 ```Array```의 최종적인 형태는 ```(1, 2, 3, 4)```다.
```python
(1, 2, 3, 4)
```
물론 ```get_shape()``` 메소드를 사용하면 단 번에 알 수 있다.
### Axis
```python
[ # axis = 0
    [ # axis = 1
        [ # axis = 2
            [1,2,3,4], # axis = 3 = -1
            [5,6,7,8],
            [9,10,11,12]
        ],
        [
            [13,14,15,16],
            [17,18,19,20],
            [21,22,23,24]
        ]
    ]
]
```
또 한가지 중요한 개념이 있다. ```Axis```, 즉 축이다. 가장 큰 범주의 차원부터 ```axis = 0```이고 총 차원의 개수(n) - 1만큼 축이 생성된다. 위 ```Array```는 4차원이므로 ```axis = 3```까지 있는 것이다.

그리고 가장 안쪽의 축은 ```axis = -1```로 표현하기도 한다. 실무적으로 많이 사용되는 개념이다. 큰 범위의 차원을 바꾸는 경우는 있어도 가장 기본 데이터가 속해있는 마지막 축은 바꾸는 경우가 극히 드물기 때문이다.
### Matmul vs Multiply
중학교 수학 시간에 우리는 행렬의 곱셈을 배웠으므로 ```TensorFlow```를 사용해 같은 연산을 할 수 있다.
```python
# multiply for matrix

matrix1 = tf.constant([[1.,2.], [3.,4.]])
matrix2 = tf.constant([[1.], [2.]])
print('Matrix 1 shape', matrix1.get_shape())
print('Matrix 2 shape', matrix2.get_shape())
```
먼저 ```matrix1```과 ```matrix2```의 형태를 알아보자.
```python
Matrix 1 shape (2, 2)
Matrix 2 shape (2, 1)
```
```matrix1```은 ```(2, 2)```, ```matrix2```는 ```(2, 1)```의 형태를 가지고 있다. 중학교 수학 시간에 배웠듯이 앞 행렬 형태의 끝자리 숫자와 뒷 행렬 행텨의 앞자리 숫자가 일치해야 행렬의 곱셈이 가능하다.
```python
with tf.Session() as sess:
    print(sess.run(tf.matmul(matrix1, matrix2)))
```
```matmul```로 행렬의 곱을 해보면,
```python
[[1. 2.]
 [6. 8.]]
```
이제 우리가 행렬의 곱셈을 손으로 계산하지 않아도 된다는 사실에 기뻐할 수 있다.
### Broadcasting
하지만 언제나 그렇듯 주의해야 할 사항이 있다. 행렬의 형태를 맞추지 않은 상태에서 여러 문제가 발생한다.
```python
# Operations between the same shapes
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2., 2.]])
with tf.Session() as sess:
    print(sess.run(matrix1 + matrix2))
```
행렬의 덧셈은 어렵지 않다. 둘 모두 ```(1, 2)```라는 같은 형태를 가지고 있는 행렬이므로 같은 자리에 있는 숫자끼리 더하면 된다.
```python
[[5. 5.]]
```
마찬가지로 ```(1, 2)```라는 같은 형태의 행렬이 도출된다.
```python
matrix1 = tf.constant([[1., 2.]])
matrix2 = tf.constant(3.)
with tf.Session() as sess:
    print(sess.run(matrix1 + matrix2))
```
한 행렬에 ```상수```를 더하는 것은 어떨까. 사실 엄격한 의미에서 행렬의 덧셈이 아니지만 ```TensorFlow```는 개념을 임의적으로 확장해서 오류가 아닌 결과를 나타낸다.
```python
[[4. 5.]]
```
행렬 각 자리의 숫자에 상수를 더한 결과값을 나타낸다.
```python
matrix1 = tf.constant([[1., 2.]])
matrix2 = tf.constant([3., 4.])
with tf.Session() as sess:
    print(sess.run(matrix1 + matrix2))
```
행렬의 차원과 형태가 달라도 덧셈을 진행한다. ```matrix1```의 형태는 ```(1, 2)```이고 ```matrix2```는 ```(2)```의 형태를 가지고 있는 상태에서 덧셈을 하면,
```python
[[4. 6.]]
```
어떻게든 결과값을 도출해낸다.
```python
matrix1 = tf.constant([[1., 2.]])
matrix2 = tf.constant([[3.], [4.]])
with tf.Session() as sess:
    print(sess.run(matrix1 + matrix2))
```
```matrix1```는 ```(1, 2)```의 형태를, ```matrix2```는 ```(2, 1)```를 가지고 있다. 곱셈은 가능하지만 덧셈은 불가능하다. 하지만 ```TensorFlow```는 ```Brodcasting```이라는 개념을 사용해 결과를 보여준다.
```python
[[4. 5.]
 [5. 6.]]
```
```(2, 2)```라는 새로운 형태의 행렬을 만들어냈다. ```Brodcasting``` 개념을 잘 알고 활용할 수 있다면 강력한 처리 도구로 사용할 수 있겠지마, 개념을 잘 모르는 상태에서 실수로 곱셈이나 덧셈을 할 경우에는 의도와 전혀 다른 결과가 나올 수 있으니 항상 ```Tensor```의 차원과 형태를 확인한 후 학습을 진행하는 습관을 가져야 한다.

---