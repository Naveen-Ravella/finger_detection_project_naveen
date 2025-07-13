import cv2
import mediapipe as mp
mp_hands=mp.solutions.hands
hands=mp_hands.Hands(static_image_mode=False,max_num_hands=2,min_detection_confidence=0.5,min_tracking_confidence=0.5)
mp_draw=mp.solutions.drawing_utils
def find_hands(img,draw=True):
    img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    res=hands.process(img_rgb)
    if res.multi_hand_landmarks:
        for hand_landmark in res.multi_hand_landmarks:
            if draw:
                mp_draw.draw_landmarks(img,hand_landmark,mp_hands.HAND_CONNECTIONS)
    return img,res
def find_position(img,res,hand_no=0,draw=True):
  lm_list=[]
  if res.multi_hand_landmarks:
    my_hand=res.multi_hand_landmarks[hand_no]
    for id,lm in enumerate(my_hand.landmark):
      h,w,c=img.shape
      cx,cy=(lm.x*w),(lm.y*h)
      lm_list.append([id,cx,cy])
      if draw:
        cv2.circle(img,(int(cx),int(cy)),15,(255,0,0),cv2.FILLED)
  return lm_list
cp=cv2.VideoCapture(0)
while True:
  success,frame=cp.read()
  if not success:
    break
  frame=cv2.flip(frame,1)
  frame,res=find_hands(frame)
  lms=find_position(frame,res,draw=False)
  finger_tips=[4,8,12,16,20]
  finger_status=[]
  if len(lms)!=0:
    if(lms[finger_tips[1]][0]<lms[finger_tips[1]-2][0]):
        print('left')
    else:
        print('right')
  if cv2.waitKey(1) & 0xFF == ord('q'):
      break
cp.release()
cv2.destroyAllWindows()
