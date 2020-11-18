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

### Train
| id | Resize Shape | Learning Rate| Optimizer  |  Loss Function  | Mean of dice score | Min of dice score | Max of dice score | Augment |
|:--:|:------------:|:---:|:---:|:--:|:--:|:--:|:--:|:--:|
|  1 | 512 x 512 |0.001|Adam|BCEWithLogitsLoss|0.7744|0.6374(2)|0.8217|None|
|  2 | 592 x 592 |0.001|Adam|BCEWithLogitsLoss|0.7789|0.6904(2)|0.8211|None|
|  3 | 592 x 592 |0.001|Adam|BCEWithLogitsLoss|0.7805|0.7433(2)|0.8261|gray|
|  4 | 592 x 592 |0.001|Adam|BCEWithLogitsLoss||||gray+flip+rotate(45)|
|  5 | 592 x 592 |0.001|Adam|BCEWithLogitsLoss|16|4|||

##### train loss, validation loss
<img src='https://user-images.githubusercontent.com/69955858/99534124-47e1d280-29ea-11eb-95bb-7348e15518b7.png' width=900 height=350>

### Test
train 64 val 16
##### input, output of image 0
<img src='https://user-images.githubusercontent.com/69955858/97461875-83582680-1981-11eb-9425-8b24348c23aa.png' width='300' height='300'> <img src='https://user-images.githubusercontent.com/69955858/97461758-61f73a80-1981-11eb-8226-9ded145721f2.png' width='300' height='300'>

##### input, output of image 2
<img src='https://user-images.githubusercontent.com/69955858/97463727-67558480-1983-11eb-8d24-22cac46a4148.png' width='300' height='300'> <img src='https://user-images.githubusercontent.com/69955858/97463746-69b7de80-1983-11eb-8cab-d0c743472c69.png' width='300' height='300'>


| image number |  0  |  1  |  2  |  3  |  4  |  5  |  6  |  7  |  8  |  9  |
|:------------:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|  dice score  |0.788|0.827|0.628|0.808|0.793|0.775|0.772|0.742|0.762|0.782|
      
| image number |  10 |  11 |  12 |  13 |  14 |  15 |  16 |  17 |  18 |  19 |
|:------------:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|  dice score  |0.770|0.797|0.789|0.792|0.768|0.799|0.757|0.784|0.804|0.781|
          
| dice_max |  dice_min  |  dice_mean  |
|:------------:|:---:|:---:|
|  0.827|0.628|0.776|

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### 현재 상황

차라리 input 이미지의 크기를 584 584로 키우고 output의 이미지를 565 584로 줄이면 나을까 싶었는데 역시 결과를 아주 비슷하다.

너무 많은 실험에 로그데이터들과 결과 이미지들이 쌓여 뭐가 뭔지 못알아보겠어서 어차피 성능이 대부분 비슷하여

flip으로 불린 80개의 데이터를 사용했을 때의 결과들은 모두 삭제했다.

트레이닝 데이터 16개로 대표 몇 가지 실험만 해서 저장해야겠다.

첫 시도는 contrastive loss를 적용해보는 것이다. 

그리고도 안되면,

https://github.com/clguo/SA-UNet 여기서 엄청난 성능을 보여주었기 때문에

하나하나씩 공들여서 봐야겠다.

슬쩍 본 바로는, 원래 데이터는 10개 였는데 augmentation으로 260개까지 늘려서 training을 진행했다.

일반 unet과 다른 점은 중간에 over fitting을 줄이기 위해 drop box를 추가하고, conv 레이어를 줄이고, spatial attention module을 추가한 것인데 베이스라인과 많은 성능차이가 난다.(심지어 피쳐맵 채널수도 기본 유넷보다 한참 작다.)

다른 점이 좀 있지만 아마도 내 생각엔 augmentation 절반정도 영향을 끼치는 것 같은데

내가 augmentation을 했을 때 성능이 비슷한거 보면,

내가 무언가 단단히 잘못하고 있는 것 같다. augmentation 하이퍼 파라미터 조절이던지 괴상한 코드던지.

다시 눈 부릅뜨고 정리해보자.