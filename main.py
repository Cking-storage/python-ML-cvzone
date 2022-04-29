import tensorflow as tf
import keras
import numpy as np
import cv2

model = keras.models.load_model('keras_model.h5')   # 모델 로드
# cap = cv2.VideoCapture("rtsp://admin:12131415@172.30.1.25:10554/tcp/av0_0")
cap = cv2.VideoCapture(0)

classes = ['Scissors', 'Rock', 'Paper']     # 가위, 바위, 보 문자열 리스트

while True:
    ret, frame = cap.read()     # frame은 numpy array 형태
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # 좌우반전없는 거울영상

    h, w, c = frame.shape
    frame = frame[:, 160:160+h] # 슬라이싱으로 티처블 머신의 영상 크기에 맞게 정사각형으로 만들다
    # h, w, c = frame.shape
    # print(h, w)

    img_input = cv2.resize(frame, (224, 224))   # 224 x 224 크기로 변경
    img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)  # BGR -> RGB 컬러 시스템으로 변경
    img_input = (img_input.astype(np.float32) / 127.0) - 1.0    # 0 ~ 255 컬러 범위를 -1.0 ~ 1.0 범위로 변경
    img_input = np.expand_dims(img_input, axis = 0) # 차원 늘리기, 0번 축 추가(1, 224, 224, 3)

    prediction = model.predict(img_input)   # data 예측결과를 리턴한다.
    print(prediction)
    idx = np.argmax(prediction) # 확률이 가장 높은 인덱스를 반환한다.(labels.txt 파일의 인덱스 참조)

    cv2.putText(frame, text = classes[idx], org = (10, 30), fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
    fontScale = 1, color = (255, 255, 255), thickness = 2)  # 그림위에 그림을 그리는 putText()

    cv2.imshow('cam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()