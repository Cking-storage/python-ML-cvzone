import cv2
import mediapipe as mp
import numpy as np

gesture = {
    0:'fist', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five',
    6:'six', 7:'rock', 8:'spiderman', 9:'yeah', 10:'ok',
}

rps_gesture ={ 0:'rock', 5:'paper', 9:'scissors' }  # 가위, 바위, 보

cap = cv2.VideoCapture(0)

# MediaPipe Hands
mp_hands = mp.solutions.hands   # mediapipe의 솔루션중 hands를 가져온다. 
mp_drawing = mp.solutions.drawing_utils   # 그리기위한 매서드를 가져온다.

with mp_hands.Hands(     # hands 객체 생성및 초기화
    max_num_hands = 1,       # 탐지할 손의 수
    min_detection_confidence = 0.5,     # 최소 탐지 신뢰도
    min_tracking_confidence = 0.5) as hands:     # 최소 추적 신뢰도

    file = np.genfromtxt('../Rock-Paper-Scissors-Machine-main/data/gesture_train.csv', delimiter = ',')
    #print(file)
    angle = file[:, :-1].astype(np.float32)     # column의 마지막 인덱스 앞까지(각도)
    label = file[:, -1].astype(np.float32)      # column의 마지막 인덱스(라벨)

    knn = cv2.ml.KNearest_create()  # OpenCV에서 제공하는 KNN 모델 사용
    knn.train(angle, cv2.ml.ROW_SAMPLE, label)  # 학습

    while cap.isOpened():   # 카메라가 열려있으면
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)  # 거울영상으로 반전
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)  # imgRGB로부터 손과 관절의 위치 결과를 받아온다.
        #print(results.multi_hand_landmarks)  # 손의 좌표를 표시한다. 손이 없으면 NONE
        
        imgBGR = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks is not None:    # 손이 인식되면
            for res in results.multi_hand_landmarks:    # 인식된 손만큼 반복한다.
                joint = np.zeros((21, 3))               # 21x3의 영행렬을 저장할 joint 객체
                for i, lm in enumerate(res.landmark):   # 손 한개당 21개의 좌표(랜드마크)를 반환한다.
                    #print(lm)
                    joint[i] = [lm.x, lm.y, lm.z]
                #print(joint)
                #좌표사이의 거리를 구한다.(벡터길이 구하기(Norm 사용))
                v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
                v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
                v = v2 - v1 # [20,3], 3차원으로 표현된 좌표간 차이값 20개를 저장(vector로 전환)
                #print('shape: ', v.shape) # [20, 3] y=20, x=3   
                
                # Normalize v
                #v = v / np.linalg.norm(v, axis=1)  # (20, 3)/(20, ) 차원 불일치로 에러.
                #print(v)
                v = v / np.expand_dims(np.linalg.norm(v, axis=1), axis=-1) # (20, 3)/(20, 1) 축 추가하여 계산실행
            #    print(v)

                # Get angle using arcos of dot product
                radi = np.arccos(np.einsum('ij,ij->i',              
                    v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                    v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]
    
            #    print('radi shape: ', radi.shape)
                angle = np.degrees(radi)    # 라디안 -> 각도 변환
                angle = np.expand_dims(angle.astype(np.float32), axis=0)    # 머신러닝모델은 float32 지원
            #   print(angle)

                # 제스처 추론하기
                _, results, _, _ = knn.findNearest(angle, 3)
            #    print(results)
                idx = int(results[0][0])    # results에 []를 없애고 int형으로 변환

                if idx in rps_gesture.keys():
                    gesture_name = rps_gesture[idx]
            #    print(gesture_name)

                    cv2.putText(imgBGR, text=gesture_name, org=(10, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2,
                        color=(255, 0, 0), thickness=2)

                mp_drawing.draw_landmarks(imgBGR, res, mp_hands.HAND_CONNECTIONS)   # 손에 좌표를 드로우잉 한다.

        cv2.imshow('MediaPipe Hands', imgBGR)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
