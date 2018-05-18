---
layout:         post
title:          Softmax Classification
subtitle:   
card-image:     https://lh3.googleusercontent.com/QD5YSR1GV--CWs1_2ILhslxfIe1hc0lDj0qVhZ2l7ZiZlTq3PxYnqDvx3JlPtBS1tkB8SHG1wG6o4vfMb1CZIy1XKppa7GoaM1JetkGjRdUus1EduZRmvozna2ShOEj34U_Y9AV2KoXfC0pyopg4hMO_0dhr9PKhK8KZfFLNColK9Et0mjTuLBX0fh-6iT98EgDZW5VzDg-IXoGL38fAMQ9nzC1KCi3tVMkdN96-YowXEtoNo-_wRkYfyn1ROYyjjF9_z4xXmdjdrPw6mh-srbvkeizv232_AZMww7XsNwOrXoyKi-8Uo5Htv_b8Ah1-80UyIOb7BvEo0HzKabbbLKZql4H5jGLa87e_eojq531e92DSOzHFHRFNhw124I3MqYQm60YGt3hdPGejzKymyqQT1J-nDAWkVPVs13UgvsCZx8vXR3nJh6hjOHlOf8MOi8h-k7w3rINqPmGbVZWlJffqvbBkfHs_S3-OdRsVjP1cIvsU4ugbEjzFAiPyZWdZPsMc76jLCXtYQx4VAiMi5qSLJkIALkkkEwZJ3MgBr5E6K_XH-SvV4nVgz1S4cXa_zYnfVOazSAfxQOrX2_KnqqS_vJVSCnbNMSUXFLiK=w518-h274-no
date:           2018-4-30 23:00:00
tags:           MachineLearning TensorFlow Coding DataScience
post-card-type: image
---

그동안 우리는 특정 숫자를 도출해내거나, 0 또는 1 중의 한 숫자를 예측하는 방법들을 살펴봤다. 사실 현실에서는 그 중간에 위치한 변수들이 너무나도 많다. 대학에서 평점을 매길 때 A부터 F까지 한정된 개수의 점수를 나타내는 것이 대표적이다. 0 또는 1만으로는 나타내기 힘들고, 100점 만점의 점수만으로 평기하기에도 애매하다. 이제 우리는 Softmax Classifiaction이라는 방법을 사용해 이 문제를 해결할 수 있다.
```python
import tensorflow as tf

x_data = [[1, 2, 1, 1],
         [2, 1, 3, 2],
         [3, 1, 3, 4],
         [4, 1, 5, 5],
         [1, 7, 5, 5],
         [1, 2, 5, 6],
         [1, 6, 6, 6],
         [1, 7, 7, 7]]

# One-hot-encoding
y_data = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]]

X = tf.placeholder('float', [None, 4])
Y = tf.placeholder('float', [None, 3])
nb_classes = 3

W = tf.Variable(tf.random_normal([4, nb_classes]), name = 'weight')
b = tf.Variable(tf.random_normal([nb_classes]), name = 'bias')
```
위의 ```x_data```와 같은 변수와 ```y_data```라는 결과물을 토대로 학습을 진행하려고 한다. 먼저 알아야 할 기본 개념은 ```One-hot-encoding```이다. 일종의 약속을 코드로 구현했다고 보면 된다. 예를 들어 ```A```라는 점수는 ```[1, 0, 0]```으로, ```B```는 ```[0, 1, 0]```으로, ```C```는 ```[0, 0, 1]```이라는 코드로 치환해서 사용한다는 약속이다. 단 하나의 숫자만 ```1```로 표현한다고 해서 ```One-hot-encoding```이라고 부른다. 위의 ```y_data```는 이와 같은 개념을 사용한 것이다.
![](https://lh3.googleusercontent.com/QD5YSR1GV--CWs1_2ILhslxfIe1hc0lDj0qVhZ2l7ZiZlTq3PxYnqDvx3JlPtBS1tkB8SHG1wG6o4vfMb1CZIy1XKppa7GoaM1JetkGjRdUus1EduZRmvozna2ShOEj34U_Y9AV2KoXfC0pyopg4hMO_0dhr9PKhK8KZfFLNColK9Et0mjTuLBX0fh-6iT98EgDZW5VzDg-IXoGL38fAMQ9nzC1KCi3tVMkdN96-YowXEtoNo-_wRkYfyn1ROYyjjF9_z4xXmdjdrPw6mh-srbvkeizv232_AZMww7XsNwOrXoyKi-8Uo5Htv_b8Ah1-80UyIOb7BvEo0HzKabbbLKZql4H5jGLa87e_eojq531e92DSOzHFHRFNhw124I3MqYQm60YGt3hdPGejzKymyqQT1J-nDAWkVPVs13UgvsCZx8vXR3nJh6hjOHlOf8MOi8h-k7w3rINqPmGbVZWlJffqvbBkfHs_S3-OdRsVjP1cIvsU4ugbEjzFAiPyZWdZPsMc76jLCXtYQx4VAiMi5qSLJkIALkkkEwZJ3MgBr5E6K_XH-SvV4nVgz1S4cXa_zYnfVOazSAfxQOrX2_KnqqS_vJVSCnbNMSUXFLiK=w518-h274-no)
```hypothesis```를 정의하는 기본 논리는 같다. ```X```라는 데이터에 ```W```라는 변수를 곱해 ```Y```라는 결과물을 예측하는 것이다. 다만 이제는 ```0``` 또는 ```1```의 단 두 결과만 나오거나 무한한 임의의 숫자를 도출해내는 것이 아니라 한정된 결과물만 나타내야 하기 때문에 더욱더 ```차원```에 대한 고려가 필요하다.

위 그림에서 결과물은 세 개만 나올 수 있다. 그리고 만약 세 개의 결과물에 대한 예측 점수가 각각 ```2.0```, ```1.0```, ```0.1```으로 나올 경우, 전체 합이 ```1```인 확률로 표시하는 것이 ```Softmax Classification```의 골자다.

그리고 그림에 나타나 있는 식은 물론 이해하면 좋지만, 몰라도 사용할 수 있다. 구글에서 이미 ```TensorFlow```안에 구현해두었기 때문에.
![](https://lh3.googleusercontent.com/rWr4lGp-oZYHACPbiALy2OlF_9ds899U6Yys-M_IOEx1KS5_J6UvYU4skZIecX2d_IaOUub5qUS7X8J4LJ7-qUFnzLv9Bkc69kfp86YD6pTnxWlPaxlmMtueda7Hrx9MHzd5-3rMt9vfYaXOhaxFskNWw2HVAiQ0bf9uukZC9I4IyzGg2_v6IiSL2jroDscc7RjsdeCS4w-dqtiUULYDwtXTyHeOSW2nsjXHlDUw9MTyV788ugPvyeDDdK81Cdp1bKOcJiDes-aA-UQT7ZkDmT_B6KTyaqRszgYoKgYBMxqKk_KP8PBHKoeogWY7SXRIaosY2wKK3l-mhRMwG1vdK-jkm_ofBNMab33n7FiUT2vuImckLSS_W_mgbv1f6UriQoIAx0qHNw2b3IuIic-TGEA8CJ5QGt2qnDfB6cLB5wTy83QLQhvbcGvhaHBvL8WGltKORSvo1dma1ruGIEX5THK40mZ5l16lpgRXD9J9En_9aAZN7_OcmU1NIxAEaAcYZOG02P9gfDUz22_RtyE3gKDo8eOD4z1XTqcd1ycl1GGHvXhTxvJRcu27WzGB_eWixyCTLID3knkll30nVpZZ3cxTqQ_ti6omzjFpS9Z-=w1398-h672-no)
```python
# tf.nn.softmax computes softmax activations
# exp(logits) / reduce_sum(exp(logits), dim)
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

# cross entropy cost/loss
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis = 1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
```
```tf.nn.softmax```라는 메소드를 사용해서 한 줄의 코드로 저 위의 복잡한 수식을 코드로 구현할 수 있다. ```cost```의 경우 기존의 방식과는 달라진 것을 발견할 수 있다. 실제 결과물인 ```Y```와 ```hypothesis```의 결과물과의 차이가 아닌 곱으로 나타내고 있다. 이또한 수학적인 부분이기 때문에 자세한 공식은 사용하지 않겠지만, 결과적으로 예측이 틀릴 경우 ```cost```가 증가하고 반대의 경우 ```cost```가 ```0```이 되는 수식으로 이해하면 된다.

```python
# Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(2001):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}))
```
학습을 하는 방식은 동일하다. 전체 ```Graph```를 그리고 변수를 초기화한 후에 ```optimizer```를 실행하기 위해 ```x_data```와 ```y_data```를 넣는다.
```python
0 2.2636468
200 0.6693883
400 0.56024593
600 0.4671715
800 0.3768742
1000 0.28712988
1200 0.23116957
1400 0.20998713
1600 0.19221275
1800 0.17709729
2000 0.16409844
```
위와 같이 출력이 되는 것을 볼 수 있다. 학습을 진행할수록 ```cost```가 ```0```을 향해 줄어드는 것을 확인할 수 있다.
```python
    # Testing & One-hot-encoding
    a = sess.run(hypothesis, feed_dict={X: [[1, 11, 7, 9],
                                           [1, 3, 4, 3],
                                           [1, 1, 0, 1]]})
    print(a, sess.run(tf.argmax(a, 1)))
```
실제로 학습 결과가 잘 나오는지 테스트를 해본다. 여기서 ```argmax```라는 새로운 메소드를 사용한다. 첫번째 그림에서 보았듯이 가장 확률이 높은 결과물을 예측치로 설정하기 위해, 여러 확률 중에서 ```max``` 값을 찾아내는 방법이다. 그 결과는 아래와 같이 나온다.
```python
[[8.8530891e-03 9.9113691e-01 1.0071689e-05]
 [8.2685965e-01 1.5776965e-01 1.5370723e-02]
 [2.3587551e-08 4.0142154e-04 9.9959856e-01]]
 
 [1 0 2]
```
첫번째 리스트에 담긴 결과물들은 각 결과물에 대한 ```score```를 나타낸다. 이를 확률로 변환해서 가장 높은 확률의 결과물을 찾아낸 것이 두번째 리스트에 담긴 숫자들이다. 학습한 내용들을 토대로 다른 임의의 ```x_data```들에 대해 새로운 예측치를 나타내는 것이다.

---