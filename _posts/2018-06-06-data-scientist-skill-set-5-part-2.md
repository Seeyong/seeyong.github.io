---
layout:         post
title:          데이터 사이언티스트가 갖춰야 할 5가지 스킬셋 Part.2
subtitle:       
card-image:     https://images.unsplash.com/photo-1509869175650-a1d97972541a?ixlib=rb-0.3.5&ixid=eyJhcHBfaWQiOjEyMDd9&s=26cbb818d3b2f1ccd7ceb03a75026f3d&auto=format&fit=crop&w=1650&q=80
date:           2018-06-06 22:30:00
tags:           MachineLearning Coding DataScience Study
post-card-type: image
---

**본 글은 Pabii 이경환 대표님의 [블로그 글](https://pabii.co/5-skillsets-for-data-scientists/)이 너무나도 유익해 공유한 내용입니다.**

> [저번글](https://seeyong.github.io/2018/06/data-scientist-skill-set-5-part-1/1)에서 이어지는 내용입니다.

---

# 3. 수리통계학
![](https://images.unsplash.com/photo-1509869175650-a1d97972541a?ixlib=rb-0.3.5&ixid=eyJhcHBfaWQiOjEyMDd9&s=26cbb818d3b2f1ccd7ceb03a75026f3d&auto=format&fit=crop&w=1650&q=80)
개발자들이 하는 가장 큰 착각이 바로 머신러닝은 공학이지 수학이 아니라는 거다. 여기에 필자는 정면으로 반박하고 싶다.

머신러닝은 수학이지 공학이 아니다. 머신러닝 입문자들이 듣는 Coursera의 Andrew Ng 아저씨 수업을 평가한 [Quora 글](https://www.quora.com/Machine-Learning-Did-Andrew-Ngs-Coursera-class-convert-a-lot-of-ML-people-into-using-Matlab-Octave)을 보라.

> *“What most people missed was that they approach machine learning as some kind of programming project, while in reality it is actually more of an applied math project with computational approach, and they really needed to focus on learning the math, which, in this case, is linear algebra.”*

머신러닝은 한 단어로 요약하면 응용수학이다. 선형대수학과 회귀분석을 섞고, 발전하다보니 더 고급 수학과 더 고급 통계학을 쓰는 그런 응용수학이다. 필자가 데이터 사이언티스트라고 소개하니 머신러닝을 공학이라고 생각해서 “어, 공대 출신이 아닌데 어떻게 데이터 사이언티스트에요?” 라고 질문하는 벤쳐 캐피탈리스트도 있더라.

정리하면, **확률론 + 선형대수 + 회귀분석**이 머신러닝의 근간이고, 이걸 정말 잘하고 싶으면 대학원 수준의 수학과 통계학을 공부해야한다. 수업 왔던 개발자들 일부가 커리어를 접고 대학원으로 방향을 튼 이유도 같은 맥락일 것이다. 그냥 갖다 쓰는 방식의 개발과는 근본적으로 다른 학문이라는 걸 이해했기 때문일 것이고, 데이터 사이언스 영역에서 인정받고 싶다면 수리통계학 깊이가 깊어야한다는 사실을 깨달으셨기 때문일듯.

![](https://pabii.co/wp-content/uploads/2017/06/math_formulas_400.jpg)

당장 빅데이터로 작업할려고 전처리할 때 Dimensionality reduction에 쓰이는 PCA나 FA 분석이 어떻게 작동하는지 머신러닝한다는 개발자들에게 물어봐라. 둘 중 어떤 방법을 써야하는지도 모르고, 프로그램에서 디폴트로 되어 있는 PCA만 쓰고 있을 것이다. (애시당초 “그냥 데이터 크기 줄여주는거 아냐?” 수준을 너무 많이 봤다.) PCA는 Total variation을 볼 때, FA는 변수들간 C0-variation을 볼 때 쓰는 테크닉들이다. 이 차이를 몰라서 학계에서 유명한 통계 패키지 하나가 계산을 거꾸로 하던 바람에 1990년대에 쓴 모든 논문을 다시 써야하는 사건들도 발생했다.

왜? variance & co-variance matrix하나 그려놓고, 모든 숫자를 다 쓰는지, off-the-diagonal term만 쓰는지에 따라 연구 결과는 당연히 다르게 나올 수 밖에 없다. 심지어는 자기 논문을 스스로 비판하고 저널에서 내려달라고 하던 연구자도 있을 정도였다.

> 무슨 말인지 잘 모르겠다고? 근데 이거 사실 자연계열 1, 2학년 선형대수학 수업에서 배운 지식이다. (그리고 필자가 페이X북 Data Scientist 면접볼 때 나왔던 질문이기도 하다.)

# 4. 데이터 시각화 패키지 지식

빅데이터 분석을 해서 결과값을 보여주는데 최적화된 수 많은 툴들이 있다. 당장 R로도 예쁜 그래프들 참 많이 그릴 수 있는데, 필자는 회사 생활하면서 Tableau로 작업하는 걸 배웠다. 좋더라. 비싸서 그렇지. 그리고 가격경쟁력이 좀 더 있는 툴로 Qilk도 써 본적이 있다. 왜 기업들이 Tableau 쓰는지 알겠다는 생각도 들었고, 이것도 참 쓸만하다는 생각도 했다.

![](https://pabii.co/wp-content/uploads/2017/06/Sankey.png)

사실 R로 작업하기로 맘 먹으면 Java Script로 그래프 만드는 걸 처리해주는 패키지도 있기 때문에, 왠만한 일들은 다 R로 해결할 수 있다. 당장 Sankey, Sunburst 같은 그래프들 한번 그릴려고 시도해봐라. 아마 아직도 R말고 다른 패키지로 저런 복잡한 그래프를 쉽게 Customizing 하기 힘들 것이다.

# 5. Data Modeling 센스
마지막으로 가장 중요한 스킬 셋이다. 위에 4가지 지식은 사실 없으면 배우면 된다. 그런데 Data Modeling 센스 없으면 데이터로 분석을 요하는 그 어떤 종류의 직업도 할 수가 없다. (아마 데이터 서버 관리하는 엔지니어가 이쪽 관련 직군에서 유일한 대안이 아닐까 싶다.)

가장 단순하게는 엑셀로 그래프를 잘 그려서 보는 사람들이 한번에 쉽게 이해할 수 있도록 해 주는 것도 포함될 것이고, 복잡하게는 어떤 통계 모델로 가설들을 검증할 수 있을지에 대한 아이디어들이 포함된다.

![](https://pabii.co/wp-content/uploads/2017/06/Modeling_Overview.png)

필자가 석사 처음들어가서 논문을 쓸 때였다. 논문 주제로 뭐 하나를 가져갔다가 입이 좀 걸걸한 교수님께 거의 모욕 수준의 평가를 듣고 나왔는데, 지금 생각해보면 그 교수님 그래도 착했다는 생각이 든다. 이런 말도 안 되는거 하지말고 주제 제대로 잡으라고 리스트를 쭈욱 뽑아주셨는데, 그 예시들 보면서 정말 눈물이 찔끔 나더라. 주제라고 뽑아주시는 리스트에 있는 내용들을 어떻게 테스트하는지 다 수업시간에 배웠는데, 실제로 이런 방식의 테스트를 하겠다는 생각을 그전까지 한번도 해 본 적이 없었다.

비싼 돈 내고 공부하러와서 여태까지 답답한 공부만 하고 있었다는 쪽팔림에 속으로 울었다.

그 분수령같은 사건 이후로, 어떤 통계 모델을 써서 테스트해야되겠다는 그림을 머릿속으로 그릴 수 있게 되었으니 그 교수님께 참 감사해야된다 싶다. (물론 그 때 같이 수업 들었던  친구들 대다수가 아직도 그 교수님의 독설에 많은 반감을 갖고 있기는 하다 ㅋㅋ)

> 사실 이 중에 3개만 잘 갖추고 있어도 데이터 사이언티스트로 먹고사는데 별 지장없을 것 같기는 하다 ㅋㅋ

##### * 출처 : [데이터 과학자가 갖춰야 할 5가지 스킬셋](https://pabii.co/5-skillsets-for-data-scientists/)

---