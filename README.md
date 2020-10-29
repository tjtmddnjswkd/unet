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
<img src='https://user-images.githubusercontent.com/69955858/97574479-a42a8580-1a2e-11eb-95a5-c573dcfeab80.png' width=500 height=400> <img src ='https://user-images.githubusercontent.com/69955858/97573974-e0111b00-1a2d-11eb-9e3e-7616d10cf515.png' width=550 height=400>

### Test

Image : 

#### 0 image
<img src='https://user-images.githubusercontent.com/69955858/97461875-83582680-1981-11eb-9425-8b24348c23aa.png' width='300' height='300'> <img src='https://user-images.githubusercontent.com/69955858/97461758-61f73a80-1981-11eb-8226-9ded145721f2.png' width='300' height='300'>

#### 2 image
<img src='https://user-images.githubusercontent.com/69955858/97463727-67558480-1983-11eb-8d24-22cac46a4148.png' width='300' height='300'> <img src='https://user-images.githubusercontent.com/69955858/97463746-69b7de80-1983-11eb-8cab-d0c743472c69.png' width='300' height='300'>

    
dice : 
      
      "0": 0.7881735933959723,
      "1": 0.8267991248299923,      
      "2": 0.6284016012292265,      
      "3": 0.8080079090459713,      
      "4": 0.7929549086361426,      
      "5": 0.7754141986436603,     
      "6": 0.7724830316742082,     
      "7": 0.7421080997844164,      
      "8": 0.7623100055233926,     
      "9": 0.7817273688431952,      
      "10": 0.7704645180814041,      
      "11": 0.7966175725492252,     
      "12": 0.7891303666070812,   
      "13": 0.7923801214236377,      
      "14": 0.7676248405768665,     
      "15": 0.7987369898257514,      
      "16": 0.756875309913415,      
      "17": 0.7841720086440155,      
      "18": 0.8037374017870269,      
      "19": 0.7808719782689414
      
aggregates : 
    
    "dice_max": 0.8267991248299923,    
    "dice_min": 0.6284016012292265,   
    "dice_mean": 0.7759495474641771
      
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### 문제점

dropout, Randaugment 등 다양한 방법을 사용했을 때

loss는 줄어드는 것을 볼 수 있었는데

output image가 아래와 같이 괴상하게 나올때가 존재한다.

<img src = 'https://user-images.githubusercontent.com/69955858/97519749-14102000-19dd-11eb-8b55-b1e3bda408a4.png' width='300' height='300'>

그리고 augmentation을 진행해도 iou 스코어가 거의 변화가 없다.

따라서 유넷에 조정이 필요해 보인다.
