from cvzone.HandTrackingModule import HandDetector
import cv2

detector = HandDetector(detectionCon=0.8, maxHands=2) # 손한개

cap_cam = cv2.VideoCapture(0)
#cap_video = cv2.VideoCapture('Sunset.mp4')
while cap_cam.isOpened():
    ret1, cam_img = cap_cam.read()
    #ret2, video_img = cap_video.read()
    if not ret1:
        break

    cam_img = cv2.flip(cam_img, 1)      # 거울반전
    
    #hands = detector.findHands(cam_img, draw=False)    # No Draw
    #print(len(hands))                                  # 손의 갯수
    hands, cam_img = detector.findHands(cam_img)        # with Draw

    if hands:
        lmList1 = hands[0]['lmList']        # 21개의 랜드마크 리스트
        bbox1 = hands[0]['bbox']            # 바운딩박스 x,y,w,h
        centerPoint1 = hands[0]['center']   # 손중앙 cx,cy
        handType1 = hands[0]['type']        # 오른손/왼손
        #print(len(lmList1), lmList1)
        print(bbox1)
        #print(centerPoint1)

        if len(hands) == 2:
            lmList2 = hands[1]['lmList']        # 21개의 랜드마크 리스트
            bbox2 = hands[1]['bbox']            # 바운딩박스 x,y,w,h
            centerPoint2 = hands[1]['center']   # 손중앙 cx,cy
            handType2 = hands[1]['type']        # 오른손/왼손
            #print(len(lmList2), lmList2)
            #print(bbox2)
            #print(centerPoint2)
            #print(handType1, handType2)

    cv2.imshow('cam', cam_img)
    #cv2.imshow('video', video_img)
    if cv2.waitKey(1) == ord('q'):
        break
