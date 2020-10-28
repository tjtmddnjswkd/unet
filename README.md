# *vessel 데이터를 unet을 사용하여 segmentation해본 repository입니다.*

### Dataset

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
![image](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FUcMLK%2FbtqDQupfFnY%2F8aCl8icgtwVKERCWfzfK41%2Fimg.png)

### Train

Optimizer :

    Adam

Loss function :

    BCEWithLogitsLoss
    
Resize :

    512 x 512
    
Loss :

### Test

Image : 

    
![image](https://user-images.githubusercontent.com/69955858/97461875-83582680-1981-11eb-9425-8b24348c23aa.png)
![image](https://user-images.githubusercontent.com/69955858/97461758-61f73a80-1981-11eb-8226-9ded145721f2.png)

    
"dice": {
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
      }
"aggregates": {
    "dice_max": 0.8267991248299923,
    "dice_min": 0.6284016012292265,
    "dice_mean": 0.7759495474641771
      }
