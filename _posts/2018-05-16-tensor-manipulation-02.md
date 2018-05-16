---
layout:         post
title:          Tensor 데이터 처리하기 Part.2
subtitle:       manipulate Tensor
card-image:     https://www.mathsisfun.com/algebra/images/matrix-multiply-a.svg
date:           2018-05-16 23:00:00
tags:           ML
post-card-type: image
---

### Matmul vs Multiply
중학교 수학 시간에 우리는 행렬의 곱셈을 배웠으므로 ```TensorFlow```를 사용해 같은 연산을 할 수 있다.
![](https://www.mathsisfun.com/algebra/images/matrix-multiply-a.svg)
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