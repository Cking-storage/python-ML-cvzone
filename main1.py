from cvzone.HandTrackingModule import HandDetector
import cv2

detector = HandDetector(detectionCon=0.8, maxHands=2) # 손두개까지 인식

cap = cv2.VideoCapture(0)
capVideo = cv2.VideoCapture('Sunset.mp4')

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))      # 프레임의 길이를 구해서 정수형으로 변환한다.
totalFrames = int(capVideo.get(cv2.CAP_PROP_FRAME_COUNT))       # 전체 프레임의 갯수를 구한다.
#print(totalFrames)     # frame 241
_, videoImg = capVideo.read()       # 정지된 화면을 띄운다.

def draw_timeLine(videoImg, rel_x):
    img_h, img_w, img_c = videoImg.shape
    timeLine_w = int(img_w * rel_x)
    cv2.rectangle(videoImg, pt1=(0, img_h-50), pt2=(timeLine_w, img_h - 48), color=(0,0,255), thickness=-1)

rel_x = 0
draw_timeLine(videoImg, rel_x)

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)      # 거울반전
    hands, img = detector.findHands(img)        # with Draw
    #print(hands)

    if hands:
        lmList1 = hands[0]['lmList']        # 21개의 랜드마크 리스트        
        fingers1 = detector.fingersUp(hands[0]) # 손가락의 상태를 확인(0: 구부렸을 때, 1: 폈을 때)
        #cv2.putText(img, text=str(fingers1), org=(10, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
        #        fontScale=1, color=(255,255,255), thickness=2)

        #length, info, img = detector.findDistance(lmList1[4], lmList1[8], img)
        length, info, img = detector.findDistance(lmList1[4][:2], lmList1[8][:2], img)  # 엄지와 검지 거리
    #    print(length)

        if fingers1 == [0, 0, 0, 0, 0]: # 주먹일때 stop
            pass
        else:                           # 탐색 또는 플레이 모드
            if length < 50:     # Navigate 탐색모드
                rel_x = lmList1[4][0] / w       # 엄지의 x 성분을 구하여 창의 상대좌표로 변환한다.(0 ~ 1)

                frameIdx = int(rel_x * totalFrames) # 동영상프레임 갯수를 출력 img 의 길이와 매칭시킨다.
                if frameIdx < 0:
                    frameIdx = 0
                elif frameIdx > totalFrames:
                    frameIdx =totalFrames

                capVideo.set(1, frameIdx)       # 읽어온 동영상 프레임 설정 (propID 1은 프레임 설정 상수) 
            #    _, videoImg = capVideo.read()
            #    cv2.imshow('video', videoImg)

                cv2.putText(img, text='Navigate %.2f, %d' % (rel_x, frameIdx), org=(10, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=1, color=(255,255,255), thickness=2)
            else:               # Play 재생모드
                frameIdx = frameIdx + 1
                rel_x = frameIdx / totalFrames
                
            _, videoImg = capVideo.read()   # 손이 발견되면 동영상을 읽어온다.
            draw_timeLine(videoImg, rel_x)
    
    cv2.imshow('video', videoImg)
    cv2.imshow('cam', img)
    if cv2.waitKey(1) == ord('q'):
        break
