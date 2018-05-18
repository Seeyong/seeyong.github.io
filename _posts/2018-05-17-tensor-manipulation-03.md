---
layout:         post
title:          Tensor 데이터 처리하기 Part.3
subtitle:       manipulate Tensor
card-image:     https://imagecdn.coggle.it/58dbf7f08d69190001e820e3-20cac254-10f6-4dcb-938b-df9bbc102111.jpg
date:           2018-05-17 21:30:00
tags:           MachineLearning TensorFlow Coding DataScience
post-card-type: image
---
![](https://imagecdn.coggle.it/58dbf7f08d69190001e820e3-20cac254-10f6-4dcb-938b-df9bbc102111.jpg)

### Reduce mean
평균을 구하는 방법도 있다. 그런데 함수의 이름이 그냥 ```mean```이 아니라 ```reduce_mean```이다. 사실 차원을 감소하며 평균을 구한다는 의미다.
```python
with tf.Session() as sess:
    print(sess.run(tf.reduce_mean([1, 2], axis=0)))
```
```python
1
```
```1```과 ```2```의 평균을 구하니 ```1```이 나왔다. 숫자의 형태가 ```float```이 아닌 ```integer```이기 때문이다. 
```python
x = [[1., 2.],
     [3., 4.]]

with tf.Session() as sess:
    print(sess.run(tf.reduce_mean(x)))
```
```python
2.5
```
데이터의 형태를 ```float```으로 바꿔주면 결과값도 ```float```으로 나오게 된다.
```python
with tf.Session() as sess:
    print(sess.run(tf.reduce_mean(x, axis=0)))
```
```axis```, 즉 축을 바꿔서 계산해보자. ```reduce_mean```에서 ```axis```의 기본값은 지정되어 있지 않고 모든 요소를 하나의 결과값으로 계산한다. 하지만 ```axis = 0```으로 지정해주면 아래와 같은 결과가 나온다.
```python
[2. 3.]
```
이 경우, 결과적으로 행렬의 세로축을 기준으로 계산을 하게 된다. 즉, ```1```과 ```3```, ```2```와 ```4```끼리 계산을 하는 것이다. 
```python
with tf.Session() as sess:
    print(sess.run(tf.reduce_mean(x, axis=1)))
```
```axis = 1```일 경우는 어떨가.
```python
[1.5 3.5]
```
행렬의 가로축을 기준으로 계산을 한다. 숫자 ```1```과 ```2```, ```3```과 ```4```끼리 계산을 한다. 결과적으로 ```axis = 0 = 세로```, ```axis = 1 =가로```라고 외울 수도 있겠지만, 앞서 배운 개념에 따라 가장 안쪽 차원이 될 수록 축의 숫자가 올라가게 된다. 더 복잡한 차원의 행렬을 위해 개념을 익히는 것이 가장 좋다.
```python
with tf.Session() as sess:
    print(sess.run(tf.reduce_mean(x, axis=-1)))
```
역시 앞에서 배운 내용과 같이, 가장 안쪽 차원의 축을 ```axis = -1```이라고 하므로,
```python
[1.5 3.5]
```
가장 안쪽 축인 ```axis = 1```과 같은 결과값이 나올 것으로 기대할 수 있다.
### Reduce sum
```reduce_mean```의 이름과 마찬가지로 ```reduce_sum```도 차원을 줄여나가며 합을 구한다.
```python
with tf.Session() as sess:
    print(sess.run(tf.reduce_sum(x)))
```
```python
10.0
```
```axis```를 지정하지 않으면 모든 값을 더하는 것도 똑같다.
```python
with tf.Session() as sess:
    print(sess.run(tf.reduce_sum(x, axis=0)))
```
```python
[4. 6.]
```
마찬가지로 ```axis = 0```을 지정하면 가장 바깥쪽 차원을 기준으로 계산을 한다. 결과적으로는 세로축을 기준으로.
```python
with tf.Session() as sess:
    print(sess.run(tf.reduce_sum(x, axis=1)))
```
```axis = 1```을 기준으로 하면,
```python
[3. 7.]
```
가장 안쪽 차원을 기준으로 계산을 한다. 현재로는 가로축 기준이다.
```python
with tf.Session() as sess:
    print(sess.run(tf.reduce_sum(x, axis=-1)))
```
```python
[3. 7.]
```
```axis = -1```로 지정해서 가장 안쪽 차원을 계산하는 개념도 똑같다.
```python
with tf.Session() as sess:
    print(sess.run(tf.reduce_mean(tf.reduce_sum(x, axis=-1))))
```
```python
5.0
```
그동안 ```Session```을 실행시킬 때 먼저 합을 구한 후 평균을 구했던 이유도 이런 이유에서다. 가장 안쪽 차원 단위의 숫자들끼리 합을 구한 후, 해당 값을 모아서 평균을 구한다.
### Argmax
```Tensor``` 안에서 가장 큰 값의 위치를 구하는 방법도 사용했었다. ```Argmax```가 그 역할을 해준다.
```python
x = [[0, 1, 2],
     [2, 1, 0]]

with tf.Session() as sess:
    print(sess.run(tf.argmax(x, axis=0)))
```
```0```과 ```2``` 사이에서, ```1```과 ```1``` 사이에서, ```2```와 ```0``` 사이에서 어느 값이 큰지 구해주는 코드다. ```axis = 0```이기 때문에 결과적으로 세로축끼리 비교를 한 것이다.
```python
[1 0 0]
```
기본적으로 ```0```이라는 순서부터 큰 값의 위치를 알려주고, 같은 값일 경우 먼저 나오는 큰 값의 위치를 알려준다.
```python
with tf.Session() as sess:
    print(sess.run(tf.argmax(x, axis=1)))
```
```python
[2 0]
```
```axis = 1```일 경우 가장 안쪽의 숫자들끼리, 즉 행렬의 가로축을 기준으로 계산을 하기 때문에 위와 같은 결과값이 도출된다. 각각 ```2```번째, ```0```번째 숫자가 가장 큰 값이라는 의미다.
```python
with tf.Session() as sess:
    print(sess.run(tf.argmax(x, axis=-1)))
```
```python
[2 0]
```
가장 안쪽 차원을 찾는 방법이기에 앞선 결과값과 똑같은 결과가 나온다.
### Reshape
우리가 가장 많이 사용할 함수가 아닌가 싶다. 어떤 복잡한 차원의 벡터가 들어와도 우리가 학습을 시키기 위해서는 차원 정리가 필요하다. 이때 사용하는 함수가 ```reshape```다.
```python
t = np.array([[[0, 1, 2],
               [3, 4, 5]],
              
              [[6, 7, 8],
               [9, 10, 11]]])

t.shape
```
```python
(2, 2, 3)
```
위와 같은 차원의 벡터가 있을 경우 우리가 한 눈에 파악하기 어렵다. 차원을 하나 낮춰서 2차원으로 만들어보자.
```python
with tf.Session() as sess:
    print(sess.run(tf.reshape(t, shape=[-1, 3])))
```
함수 안의 파라미터 ```shape``` 자체로 차원을 설정할 수 있다. 2개의 숫자가 들어가므로 2차원으로 만들 수 있는 것이다. 뒤의 숫자부터 살펴보면, ```3```은 기존 벡터의 가장 안쪽 차원 숫자 개수와 같다. 가장 근본이 되는 숫자 데이터는 그대로 놔두는 것이다.

그 앞의 ```-1```은 ```TensorFlow```가 나름대로 차원을 구성하라는 의미다. 정확한 의미로는 가장 마지막 차원 하나만 더 만들고 끝내라는 의미이므로 마지막 차원을 구성한 후 결과물을 나타내게 된다.
```python
[[ 0  1  2]
 [ 3  4  5]
 [ 6  7  8]
 [ 9 10 11]]
```
앞선 벡터보다 훨씬 보기에 편한 모습으로 ```reshape```된 결과물을 볼 수 있다.
```python
with tf.Session() as sess:
    print(sess.run(tf.reshape(t, shape=[-1, 1, 3])))
```
차원을 그대로 유지한 채 다른 모양으로도 바꿀 수 있다. 3차원 상태는 유지한 채로 중간중간 형태를 바꾸도록 했다.
```python
[[[ 0  1  2]]

 [[ 3  4  5]]

 [[ 6  7  8]]

 [[ 9 10 11]]]
```
같은 3차원의 벡터이지만 다른 형태로 표현된 것을 확인할 수 있다.

---