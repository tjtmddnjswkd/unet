-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## retinal vessel 데이터를 unet을 사용하여 segmentation해본 repository입니다.
### 의료영상의 자동분할은 의사의 진단에 도움이 될 수 있는 유용한 정보를 추출하는 중요한 단계다. 예를 들어, 망막 혈관을 분할하여 우리가 망막 혈관의 구조를 나타내고 폭을 측정할 수 있도록 할 수 있으며, 이는 망막 질환을 진단하는 데 도움이 될 수 있다. 이 포스트에서는 망막 혈관 영상에 적용되는 영상 분할을 수행하는 Neural baseline을 구현한다.
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### Dataset

![image](https://user-images.githubusercontent.com/69955858/96858840-bdb25700-149b-11eb-80e1-f206e111e4b7.png)
![image](https://user-images.githubusercontent.com/69955858/96858857-c1de7480-149b-11eb-8b25-7f521722bf03.png)

| Size of Image |  Num of Data  |  Channels of Input Image  |Channel of Mask| Num of Train | Num of Val|
|:------------:|:---:|:---:|:---:||:--:||:--:|
| 584 x 565|20|3|1|16|4|
  
### Model

##### UNET 사용
<img src='https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FUcMLK%2FbtqDQupfFnY%2F8aCl8icgtwVKERCWfzfK41%2Fimg.png' width=700 height=600>

### Train and Test
| id | Resize Shape | Learning Rate| Optimizer  |  Loss Function  | Mean of dice score | Min of dice score | Max of dice score | Augment |
|:--:|:------------:|:---:|:---:|:--:|:--:|:--:|:--:|:--:|
|  1 | 512 |0.001|Adam|BCEWithLogitsLoss|0.7744|0.6374(2)|0.8217|None|
|  2 | 592 |0.001|Adam|BCEWithLogitsLoss|0.7789|0.6904(2)|0.8211|None|
|  3 | 592 |0.001|Adam|BCEWithLogitsLoss|0.7805|0.7433(2)|0.8261|gray|
|  4 | 592 |0.001|Adam|BCEWithLogitsLoss|0.7931|0.7650(2)|0.8295|gray+flip+rotate(45)|
|  5 | 592 |0.001|Adam|BCEWithLogitsLoss|0.7892|0.7565(16)|0.8264|gray+flip+rotate(45)+dropout|

##### train loss, validation loss
<img src='https://user-images.githubusercontent.com/69955858/99534124-47e1d280-29ea-11eb-95bb-7348e15518b7.png' width=900 height=350>

### Test result
id - 4

##### input and output of image 0
<img src='https://user-images.githubusercontent.com/69955858/97461875-83582680-1981-11eb-9425-8b24348c23aa.png' width='300' height='300'> 
<img src='https://user-images.githubusercontent.com/69955858/99538409-47e4d100-29f0-11eb-8fb9-edc062787ac3.png' width='300' height='300'>

result in base U-Net

<img src='https://user-images.githubusercontent.com/69955858/99538692-ab6efe80-29f0-11eb-8346-6151366c983b.png' width='300' height='300'>

result in SA-Unet

<img src='https://user-images.githubusercontent.com/69955858/99538852-e3764180-29f0-11eb-86ca-e619914d7bfd.png' width='300' height='300'>

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### 현재 상황

지금까지의 실험 세팅이 완전히 잘못되어 전부 삭제 후 다시 진행하였다.

정확하게 어떤 것이 성능이 나아지는지 알기 위해 하나씩 하나씩 추가하면서 천천히 정리해보았다.

현재까지는 gray+flip+rotate(45) 를 적용하였을 때의 결과가 가장 좋았다.

왜 RGB보다 gray가 더 좋은 지는 잘 모르겠다.
