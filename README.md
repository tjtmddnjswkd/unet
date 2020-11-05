-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## retinal vessel 데이터를 unet을 사용하여 segmentation해본 repository입니다.
### 의료영상의 자동분할은 의사의 진단에 도움이 될 수 있는 유용한 정보를 추출하는 중요한 단계다. 예를 들어, 망막 혈관을 분할하여 우리가 망막 혈관의 구조를 나타내고 폭을 측정할 수 있도록 할 수 있으며, 이는 망막 질환을 진단하는 데 도움이 될 수 있다. 이 포스트에서는 망막 혈관 영상에 적용되는 영상 분할을 수행하는 Neural baseline을 구현한다.
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### Dataset

![image](https://user-images.githubusercontent.com/69955858/96858840-bdb25700-149b-11eb-80e1-f206e111e4b7.png)
![image](https://user-images.githubusercontent.com/69955858/96858857-c1de7480-149b-11eb-8b25-7f521722bf03.png)

| Size of Image |  Num of Data  |  Channels of Input Image  |Channel of Mask|
|:------------:|:---:|:---:|:---:|
| 584 x 565|20|3|1|
  
### Model

##### UNET 사용
<img src='https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FUcMLK%2FbtqDQupfFnY%2F8aCl8icgtwVKERCWfzfK41%2Fimg.png' width=700 height=600>

### Train
| Resize Shape | Learning Rate| Optimizer  |  Loss Function  |Num of Train| Num of Val|
|:------------:|:---:|:---:|:--:|:--:|:--:|
| 512 x 512|0.001|Adam|BCEWithLogitsLoss|16|4|

##### train loss, validation loss
<img src='https://user-images.githubusercontent.com/69955858/97574479-a42a8580-1a2e-11eb-95a5-c573dcfeab80.png' width=500 height=330> <img src ='https://user-images.githubusercontent.com/69955858/97573974-e0111b00-1a2d-11eb-9e3e-7616d10cf515.png' width=500 height=330>

### Test

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

다양한 augmentation에도 불구하고 iou 스코어의 변화가 작고,

2번 이미지에 대한 스코어가 상대적으로 작다고 판단하여 overfitting이 발생한다고 판단.

이를 방지하기 위해 dropout을 적용해봄.

그러나 애초에 training image가 적어서인지 큰 변화가 없다.

그리고 gray sclae로 변형 후 돌려보았을 때 괜찮은 성능을 보였으나,

이상하게 flip을 통해 데이터를 불렸을때는 엄청나게 underfitting이 됐다.

이제 무엇을 적용해야 할지 잘 모르겠다.

<img src='https://user-images.githubusercontent.com/69955858/98187173-abccbb80-1f53-11eb-9358-9d93eb5a0353.png' width=500 height=350>
