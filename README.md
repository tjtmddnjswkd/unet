-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## vessel 데이터를 unet을 사용하여 segmentation해본 repository입니다.
### 의료영상의 자동분할은 의사의 진단에 도움이 될 수 있는 유용한 정보를 추출하는 중요한 단계다. 예를 들어, 망막 혈관을 분할하여 우리가 망막 혈관의 구조를 나타내고 폭을 측정할 수 있도록 할 수 있으며, 이는 망막 질환을 진단하는 데 도움이 될 수 있다. 이 포스트에서는 망막 혈관 영상에 적용되는 영상 분할을 수행하는 Neural baseline을 구현한다.
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### Dataset

Image :

![image](https://user-images.githubusercontent.com/69955858/96858840-bdb25700-149b-11eb-80e1-f206e111e4b7.png)
![image](https://user-images.githubusercontent.com/69955858/96858857-c1de7480-149b-11eb-8b25-7f521722bf03.png)

Size :

    584 x 565
  
Num of data :

    20
  
Channels of input image :
  
    3
  
Channel of mask :
  
    1
  
### Model

#### - UNET 사용
<img src='https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FUcMLK%2FbtqDQupfFnY%2F8aCl8icgtwVKERCWfzfK41%2Fimg.png' width=700 height=600>

### Train

Resize :

    512 x 512

Optimizer :

    Adam

Loss function :

    BCEWithLogitsLoss
        
Loss :

#### train loss, validation loss
<img src='https://user-images.githubusercontent.com/69955858/97574479-a42a8580-1a2e-11eb-95a5-c573dcfeab80.png' width=500 height=330> <img src ='https://user-images.githubusercontent.com/69955858/97573974-e0111b00-1a2d-11eb-9e3e-7616d10cf515.png' width=500 height=330>

### Test

Image : 

#### image 0
<img src='https://user-images.githubusercontent.com/69955858/97461875-83582680-1981-11eb-9425-8b24348c23aa.png' width='300' height='300'> <img src='https://user-images.githubusercontent.com/69955858/97461758-61f73a80-1981-11eb-8226-9ded145721f2.png' width='300' height='300'>

#### image 2
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
### 문제점

dropout, Randaugment 등 다양한 방법을 사용했을 때

loss는 줄어드는 것을 볼 수 있었는데

output image가 아래와 같이 괴상하게 나올때가 존재한다.

<img src = 'https://user-images.githubusercontent.com/69955858/97519749-14102000-19dd-11eb-8b55-b1e3bda408a4.png' width='300' height='300'>

그리고 augmentation을 진행해도 iou 스코어가 거의 변화가 없다.

따라서 유넷에 조정이 필요해 보인다.
