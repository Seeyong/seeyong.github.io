---
layout:     	post
title:      	가설 세우기 & 비용 함수 만들기
subtitle:   	Minimize It!
card-image: 	https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Linear_regression.svg/400px-Linear_regression.svg.png
date:       	2018-04-09 20:30:00
tags:       	Machine Learning, TensorFlow, Coding, Data Science
post-card-type: image
---

# Hypothesis and Cost Function

![](https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Linear_regression.svg/400px-Linear_regression.svg.png)

우리가 가지고 있는 ```X``` 값과 ```Y```라는 결과값을 가지고서 최적의 ```기울기(W)```와 ```절편(b)```을 구할 수 있다면, 우리가 가지고 있지 않은 미지의 새로운 ```X``` 값에 따른 ```Y```를 추정해볼 수 있다.

X|Y
--|--
1|1
2|2
3|3

우리가 가지고 있는 데이터가 위 표와 같다면 우리는 ```독립변수 X```와 ```종속변수 Y```가 어떤 관계를 가지고 있는지 좌표평면에 찍어보며 추정해볼 수 있다. 물론 위의 사례는 너무 명확해서 직관적으로 판별할 수 있지만 앞으로 우리가 분석할 데이터는 전혀 그런 친절함이 없다.

## Hypothesis and Cost Function
$$H(x) = Wx + b$$
$$cost(W, b) = \frac{1}{m}\sum_{i=1}^m (H(x^i) - y^i)^2$$

말그대로 ```가설(Hypothesis)```을 찾아보고 검증하여 오차를 줄이려는 것이 우리의 목표다. 앞서 언급했듯이 ```기울기(W)```와 ```절편(b)```을 추정해보려 한다.
그리고 각 가설마다 실제 ```Y```와의 차이를 구한 후, 제곱의 평균을 해서 ```cost / loss```를 구할 수 있다.

```
제곱을 하는 이유는 두 가지.
1. 오차가 음수일 경우 양수로 통일
2. 오차가 클 수록 더 많은 페널티 부여
```

### 1 Build Graph(tensor) using TensorFlow operations
```python
# H(x) = Wx + b
# X and Y data
x_train = [1, 2, 3]
y_train = [1, 2, 3]

# variable in TF (= trainable variable)
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

#Our Hypothesis WX + b(node)
hypothesis = W * x_train + b
```
먼저 ```x_train```과 ```y_train```을 각각 설정한다. 그리고 우리의 ```node```를 하나씩 만들어 전체 그래프(```tensor```)를 완성해보자.

```기울기(W)```는 1차원의 무작위 숫자로, 이름은 ```weight(가중치)```이고 ```tf.Variable```로 ```node```를 지정한다. ```TensorFlow```에서의  ```variable```은 일반적인 프로그래밍의 변수가 아니라 ```train 가능한 변수``` 정도로 이해하면 된다. ```절편(b)``` 역시 같은 방식으로 ```node```를 만든다.

이후 우리의 가설식에 두 변수를 대입한다. 
$$H(x) = Wx + b$$
```python
hypothesis = W * x_train + b
```
이제 ```hypothesis```라는 새로운 가설 ```node```가 생성됐다.

```python
# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

#reduce_mean
'''
t = [1., 2., 3., 4.]
tf.reduce_mean(t) == 2.5
'''
```
이제 ```cost```를 측정하는 ```node```를 작성할 차례다. ```hypothesis```와 마찬가지로 수식을 코드로 옮기면 된다. 
$$cost(W, b) = \frac{1}{m}\sum_{i=1}^m (H(x^i) - y^i)^2$$
```python
cost = tf.reduce_mean(tf.square(hypothesis - y_train))
```
```reduce_mean```은 일단 일반적인 평균을 구하는 방식으로 이해하면 된다.

### GradientDescent
```python
#Minimize cost
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)
```
현재 시점에서 이 부분에 대한 자세한 설명은 건너뛰자. 다음 포스팅에서 다룰 내용인데, 우리가 설정한 ```cost```를 최소화해주는 마법 공식이라는 정로로 이해하자. 최종적으로 ```train```이라는 ```node```를 만들어 사용할 것이다.

### 2,3 Run/Update Graph and get Result
```python
# Launch the graph in a session
sess = tf.Session()

# Initializes global variables in the graph
sess.run(tf.global_variables_initializer())
```
전체 그래프(```tensor```)를 만들기 위해 ```session```을 지정하고서, ```global_variables_initializer```를 실행한다. 앞서 우리가 만들었던 ```W```와 ```b```의 ```tf.Variable(tf.random_normal())```을 확정적으로 선언하기 위한 작업이다.

```python
# Fit the line
for step in range(2001):
    sess.run(train)
    if step % 100 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))
```
이제 ```train```을 2,000번 실행시켜 가장 ```cost / loss```가 적은 ```W```와 ```b```를 찾아낼 것이다. 모든 스텝마다 출력할 경우 부담스러우니 100번 마다 출력할 것이다.
```python
0 0.18124902 [0.54359144] [0.70696163]
100 0.051277865 [0.736996] [0.59786755]
200 0.0316866 [0.79325557] [0.46997872]
300 0.01958041 [0.83747995] [0.3694462]
400 0.012099485 [0.8722445] [0.2904183]
500 0.0074767545 [0.89957255] [0.22829522]
600 0.0046201865 [0.92105484] [0.17946094]
700 0.0028549908 [0.93794185] [0.1410727]
800 0.0017642155 [0.95121664] [0.11089599]
900 0.001090174 [0.96165186] [0.08717432]
1000 0.00067366054 [0.96985495] [0.06852694]
1100 0.00041627904 [0.9763032] [0.05386832]
1200 0.0002572355 [0.9813722] [0.04234548]
1300 0.00015895742 [0.9853568] [0.03328738]
1400 9.8226825e-05 [0.9884891] [0.02616697]
1500 6.069679e-05 [0.9909515] [0.02056947]
1600 3.7507194e-05 [0.992887] [0.01616953]
1700 2.3177143e-05 [0.99440855] [0.01271076]
1800 1.4321881e-05 [0.9956046] [0.00999186]
1900 8.850558e-06 [0.9965448] [0.00785452]
2000 5.469174e-06 [0.9972839] [0.00617441]
```
결과는 다음과 같다. 우리가 직관적으로 예상한 바와 같이 ```W```(세번째 column)는 ```1.0```에  수렴하고, ```b```(네번째 column)는 ```0.0```에 수렴한다.

```python
# Now we can use X and Y in place of x_data and y_data
# placeholders for a tensor that will be always fed using feed_dict
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

for step in range(2001):
    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train],
                feed_dict={x: [1, 2, 3], y: [1, 2, 3]})
    if step % 100 == 0:
        print(step, cost_val, W_val, b_val)
```
```placeholder```를 사용해 ```node```에 나중에 값을 넣어줄 수도 있다. ```tf.placeholder```로 ```x```와 ```y```라는 ```node```를 생성한 후, 다시 2,000번 반복해서 ```train```을 실행한다.
```python
0 3.0695446e-12 [0.99999785] [4.350787e-06]
100 3.0695446e-12 [0.99999785] [4.350787e-06]
200 3.0695446e-12 [0.99999785] [4.350787e-06]
300 3.0695446e-12 [0.99999785] [4.350787e-06]
400 3.0695446e-12 [0.99999785] [4.350787e-06]
500 3.0695446e-12 [0.99999785] [4.350787e-06]
600 3.0695446e-12 [0.99999785] [4.350787e-06]
700 3.0695446e-12 [0.99999785] [4.350787e-06]
800 3.0695446e-12 [0.99999785] [4.350787e-06]
900 3.0695446e-12 [0.99999785] [4.350787e-06]
1000 3.0695446e-12 [0.99999785] [4.350787e-06]
1100 3.0695446e-12 [0.99999785] [4.350787e-06]
1200 3.0695446e-12 [0.99999785] [4.350787e-06]
1300 3.0695446e-12 [0.99999785] [4.350787e-06]
1400 3.0695446e-12 [0.99999785] [4.350787e-06]
1500 3.0695446e-12 [0.99999785] [4.350787e-06]
1600 3.0695446e-12 [0.99999785] [4.350787e-06]
1700 3.0695446e-12 [0.99999785] [4.350787e-06]
1800 3.0695446e-12 [0.99999785] [4.350787e-06]
1900 3.0695446e-12 [0.99999785] [4.350787e-06]
2000 3.0695446e-12 [0.99999785] [4.350787e-06]
```
역시나 ```W```(세번째 column)는 ```1.0```에  수렴하고, ```b```(네번째 column)는 ```0.0```에 수렴한다. 이런 과정을 거쳐 우리는 하나의 ```선형회귀분석모델(Linear Regression Model)```을 ```TensorFlow```로 만들어냈다.

---