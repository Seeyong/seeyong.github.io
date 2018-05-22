---
layout:         post
title:          XOR 문제 해결하기 Part.1
subtitle:       Neural Net for XOR
card-image:     http://ecee.colorado.edu/~ecen4831/lectures/xor2.gif
date:           2018-05-19 14:30:00
tags:           MachineLearning TensorFlow Coding DataScience
post-card-type: image
---
```and``` 조건과 ```or``` 조건으로 ```False = 0```과 ```True = 1```을 구분하며 최적의 변수를 찾아 학습하는 것이 머신러닝의 기본이다. 그런데 과거 머신러닝에 대한 연구가 시작되고 얼마 되지 않았던 1969년, 중대한 문제가 발생했다. 
![](https://images-na.ssl-images-amazon.com/images/I/71QXEPrw3YL.jpg)
```Marvin L. Minsky``` 교수가 위 기본 공식으로 해결할 수 없는 난제를 제시한 것이다. 먼저 다음의 표를 살펴보자.
x1|x2|y
---|---|---
0|0|0
1|0|0
0|1|0
1|1|1
위 표는 ```and``` 조건일 때 독립변수(```x1, x2```)와 종속변수(```y```)의 관계를 나타낸 표다.
x1|x2|y
---|---|---
0|0|0
1|0|1
0|1|1
1|1|1
위 표는 ```or``` 조건일 때 독립변수(```x1, x2```)와 종속변수(```y```)의 관계를 나타낸 표다. 그리고 두 표를 좌표평면으로 나타내면 아래 그림과 같다.
![](https://cdn-images-1.medium.com/max/1600/1*CyGlr8VjwtQGeNsuTUq3HA.jpeg)
두 경우 모두 하나의 선(```linear```)를 그어 ```True``` 영역과 ```False``` 영역을 구분할 수 있다. 그럼 다음의 경우를 살펴보자.
x1|x2|y
---|---|---
0|0|0
1|0|1
0|1|1
1|1|0
만약 ```x1```과 ```x2```가 같을 경우 무조건 ```0 = False```이고 다를 경우는 무조건 ```1 = True```인 ```XOR``` 조건을 정리한 표다. 이경우 다시 위 그림으로 돌아가보면 단 하나의 선으로 ```True```와 ```False``` 영역을 구분할 수 없다. 그동안 해왔던 방식으로 문제를 풀 수 없는 것이다. ```Marvin L. Minsky``` 교수는 이 부분을 지적하면서, 변수와 관계가 더 복잡해질 경우 아무도 이 문제를 풀 수 없다고 주장했다.

정말 풀 수 없는 문제일까. 어느 시대나 덕후들은 세상을 바꿔왔다. 발상의 전환을 통해 해결할 수 있게 됐다.
## Backpropagation
![](http://hmkcode.github.io/images/ai/backpropagation.png)
진짜로 거꾸로 밟아 올라가보자는 아이디어다. 만약 우리가 학습을 하기 위한 테스트 데이터의 독립변수와 종속변수 모두 갖고 있다면 결과물을 도출해내기 위해 독립변수가 어떻게 영향을 미치는지 거꾸로 추론해볼 수 있다.

그리고 여기서 한 가지 방법이 더 추가된다. ```chain-rule```을 사용하는 것. 중간 중간 특정 변수들이 최종 결과물을 도출해 내는데 여러 영행을 줄텐데, 그 요소들을 잘게 쪼개서 서로 어떻게 연결되어 있는지 나눠보자는 아이디어다.
![](https://cdn-images-1.medium.com/max/1600/1*q1M7LGiDTirwU-4LcFq7_Q.png)
결과적으로 최종 결과값을 ```미분(derivative)```하면 요소를 쪼갤 수 있다. 특별히 하나의 변수만 알아내기 위해 ```편미분(Partial derivative)```을 한다.

![](https://i2.wp.com/python3.codes/wp-content/uploads/2017/01/XIHOY.jpg?fit=640%2C417)
```backpropagation```과 ```chain-rule```을 사용해 이제 우리는 ```XOR``` 문제를 풀 수 있는 실마리를 발견했다. 그리고 당연하게도 ```TensorFlow```로 구현해 실제 ```XOR``` 문제를 해결할 수도 있다.

---