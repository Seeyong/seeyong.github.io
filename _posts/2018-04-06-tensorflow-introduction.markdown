---
layout:         post
title:          TensorFlow 기초
subtitle:       a powerful ML tool
card-image:     https://upload.wikimedia.org/wikipedia/commons/thumb/1/11/TensorFlowLogo.svg/2000px-TensorFlowLogo.svg.png
date:           2018-04-06 23:20:00
tags:           dev
post-card-type: image
---

# Hello Tensorflow
![tensorflow stucture](http://nlpx.net/wp/wp-content/uploads/2015/11/TensorFlow-graph1.jpg)

```TensorFlow```의 구조...아직 몰라도 됨. 나도 모름.
**```이론 말고 실전부터```** 일단 코드부터 바로 쳐보기.

```python
import tensorflow as tf

# Create a constant op
# This op is added as a node to the default graph
hello = tf.constant("Hello, Tensorflow!")

#assert a TF session
sess = tf.Session()

# run the OP and get result
print(sess.run(hello))
```
```tf.constant```로 ```node```를 생성한 후 ```session```을 실행하면 다음과 같이 출력
```python
b'Hello, Tensorflow!' # b is short for binary
```
# Computational Graph
### 1 Build Graph(tensor) using TensorFlow operations
### 2 feed Data and Run Graph(operation)
### 3 Update variables in the graph
위 순서대로 ```Tensor```가 ```Flow```함
```python
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
node3 = tf.add(node1, node2)
```
```node1```과 ```node2```를 생성한 후 두 ```node```를 합하는 ```node3```를 생성
```python
print("node1 :",node1, "node2 :",node2)
print("node3 :",node3)
```
```node```를 출력하면 바로 출력되지 않고 다음과 같이 ```Tensor```의 형태를 알려줌
```python
node1 : Tensor("Const_1:0", shape=(), dtype=float32)
node2 : Tensor("Const_2:0", shape=(), dtype=float32)
node3 : Tensor("Add:0", shape=(), dtype=float32)
```
실제 결과를 출력하기 위해서는 다음과 같은 코드가 필요
```python
sess = tf.Session()
print("sess.run(node1, node2) :", sess.run([node1, node2]))
print("sess.run(node3) :", sess.run(node3))
```
```session```을 지정한 다음에 ```run```으로 출력
```python
sess.run(node1, node2) : [3.0, 4.0]
sess.run(node3) : 7.0
```
# Placeholder
특정 ```constant```를 지정하지 않고서도 ```placeholder```를 사용해 ```node```를 만들어 활용할 수 있음
```python
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node  = a + b # + provides a shortcut for tf.add(a, b)

print(sess.run(adder_node, feed_dict={a:3, b:4.5}))
print(sess.run(adder_node, feed_dict={a: [1,3], b: [2,4]}))
```
바로 ```run```을 실행할 때 ```feed_dict``` 딕셔너리로 값을 지정할 수 있음
* **```tf.Session```을 따로 지정하지 않음**

```python
7.5
[3. 7.]
```
---