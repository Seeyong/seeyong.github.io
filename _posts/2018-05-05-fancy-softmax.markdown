---
layout:    	 	post
title:      	Fancy한 Softmax
subtitle:   	
card-image: 	
date:       	2018-05-05 21:00:00
tags:       	ML
post-card-type: article
---

여러 예측 결과물을 도출해내기 위한 방법으로 ```Softmax Classification```을 사용한다. 저번 글에서는 기초적인 여러 공식들을 우리가 직접 구현해냈지만, 사실 구글이 이미 다 한 줄의 코드로 만들어놨다. 공부를 하는 입장에서 작동 원리를 아는 것이 중요하기 때문에 기초부터 배우는 것이 중요하지만, 실무에서는 간단하게 사용할 수 있다는 의미다. 이번에는 더 ```Fancy```하게 ```Softmax Classification```을 구현해보자.
```python
logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)

# Cross entropy cost/loss
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
```
이전 글에서 구현했던 기본적인 코드는 위와 같다. ```logits```라는 ```node```를 정의한 후 이를 활용해 ```hypothesis```를 ```Softmax```로 만들었다. ```cost```역시 수학 공식을 하나하나 코드로 구현해 만들어 두었다.
```python
# Cross entropy cost/loss
cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot)

cost = tf.reduce_mean(cost_i)
```
하지만 ```softmax_cross_entropy_with_logits```라는 한 줄의 코드를 구글이 이미 만들어놨다. ```logits```과 ```one-hot-encoding```으로 표현한 ```Y``` 값을 지정하면 된다. 실제 데이터를 분석해보자.
```python
import numpy as np

# Predicting animal type based on various features
xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)
x = xy[:, 0:-1]
y = xy[:, [-1]]
```
총 16가지의 특징을 활용해 각 동물이 어떤 종으로 분류되는지 정리한 데이터를 분석해보자. ```numpy```의 ```loadtxt```를 활용해 데이터를 읽어오고 각각 ```feature```인 ```x```와 결과값인 ```y```로 나눈다.
```python
nb_classes = 7 # 0 ~ 6

X = tf.placeholder(tf.float32, [None, 16])
Y = tf.placeholder(tf.int32, [None, 1]) # 0 ~ 6, shape=(?, 1)

Y_one_hot = tf.one_hot(Y, nb_classes) # one-hot shape =(?, 1, 7)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes]) # shape=(?, 7)

W = tf.Variable(tf.random_normal([16, nb_classes]), name = 'weight')
b = tf.Variable(tf.random_normal([nb_classes]), name = 'bias')
```
총 결과값은 ```0 ~ 6```의 7가지로 표현된다. ```X```와 ```Y``` 각각의 형태에 맞게 ```placeholder```를 지정해주고, 특히 ```Y```는 ```one-hot-encoding```으로 바꿔준다. 이때 사용하는 메소드가 ```tf.one_hot```이다. ```Y```라는 데이터를 총 ```7```개로 표현되는 ```one-hot-encoding```으로 만들라는 의미다. 그리고 ```double-brackets```로 도출되는 결과값을 다시 하나의 리스트 형태로 바꿔주기 위해 ```tf.reshape``` 메소드를 적용한다. ```W```와 ```b``` 역시 각각의 형태에 맞게 ```Variable```을 지정한다.
```python
# tf.nn.softmax computes softmax activation
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)

# Cross entropy cost/loss
cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot)

cost = tf.reduce_mean(cost_i)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
```
이제 다시 처음의 내용으로 돌아와 구글에서 만들어둔 ```Softmax Classification```을 적용하면 된다. 지금까지 짰던 코드의 흐름과 크게 다르지 않다.
```python
prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
```
```one-hot-encoding```으로 표현된 ```Y``` 값 중에서 우리가 알아보기 쉽도록 ```tf.argmax```를 사용해 다시 ```0 ~ 6``` 사이의 특정한 하나의 값으로 바꿔준다. 우리가 예측한 값과 실제 ```Y```의 값이 일치하는지 ```tf.equal``` 메소드로 판별을 하고 ```accuracy```도 계산할 수 있다.
```python
# Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(2000):
        sess.run(optimizer, feed_dict={X: x, Y: y})
        if step % 200 == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={X: x, Y: y})
            print('Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}'.format(step, loss, acc))
            
    # Let's see if we can predict
    pred = sess.run(prediction, feed_dict={X: x})
    
    # y_data: (N, 1) = flatten => (N, ) matches pred.shape
    for p, y in zip(pred, y.flatten()):
        print('[{}] Prediction: {} True Y: {}'.format(p == int(y), p, int(y)))
```
이제 그래프를 그려서 학습을 시켜본다. 우리가 최종적으로 알고 싶은 것은 학습 회차가 진행될 수록 ```cost```와 ```accurcy```가 어떻게 개선되는지다. 이후 각 개체당 실제 결과값이 일치하는지도 확인할 수 있다.
```python
Step:     0	Loss: 5.991	Acc: 5.94%
Step:   200	Loss: 0.533	Acc: 85.15%
Step:   400	Loss: 0.334	Acc: 91.09%
Step:   600	Loss: 0.237	Acc: 93.07%
Step:   800	Loss: 0.180	Acc: 94.06%
Step:  1000	Loss: 0.143	Acc: 95.05%
Step:  1200	Loss: 0.117	Acc: 100.00%
Step:  1400	Loss: 0.099	Acc: 100.00%
Step:  1600	Loss: 0.085	Acc: 100.00%
Step:  1800	Loss: 0.075	Acc: 100.00%
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
[True] Prediction: 3 True Y: 3
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
...
[True] Prediction: 1 True Y: 1
[True] Prediction: 0 True Y: 0
[True] Prediction: 5 True Y: 5
[True] Prediction: 0 True Y: 0
[True] Prediction: 6 True Y: 6
[True] Prediction: 1 True Y: 1
```
학습 회차가 진행될수록 ```cost```는 줄어들고 ```accuracy```는 ```100%```에 도달하는 것을 확인할 수 있다. 각 개체별로 나눠봤을 때도 실제 결과값과 우리가 예측한 값이 일치한다. 이제 우리는 ```Fancy```하게 ```Softmax Classification```을 사용할 수 있다.

---