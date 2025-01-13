import mediapipe as mp
import cv2
import numpy as np
import video_model as vm
import tensorflow as tf
from tensorflow.keras.models import load_model
from video_model import load_data as ld
from PIL import Image, ImageDraw, ImageFont
from test import reshape_data
#각도 정확도 계산
def calculate_accuracy(actual,predicted_coordinates):
    total_angle = 0
    count = 0
    actual=np.squeeze(actual)#차원크기 줄이기 1인축만 제거 가능

    actual_1=np.array(actual[0])#어깨(x1,y1,z1)
    actual_2=np.array(actual[1])#엉덩이(x2,y2,z2)
    actual_3=np.array(actual[2])#무릎(x3,y3,z3)
    # predicted_array = np.array(list(predicted.values()), dtype=float)
    
    #각도 계산(x1,y1,z1)(x2,y2,z2)(x3,y3,z3) actual[0]=left shoulder actual[1]=left hip actual[2]=left knee
    # 벡터 계산
    shoulder_hip = actual_1 - actual_2
    
    knee_hip = actual_3 - actual_2

    # 내적 계산
    shoulder_hip_dot = shoulder_hip**2  # sholder_hip와 자기 자신 내적
    knee_hip_dot = knee_hip**2  # knee_hip와 자기 자신 내적
    #전치 계산(둘중 하나 계산)
    shoulder_hip=np.transpose(shoulder_hip, (1, 0))
    
    # 내적의 합 계산
    knee_hip_simple_dot_sum_shoulder_hip = np.dot(knee_hip, shoulder_hip)
    
    # 벡터의 제곱 계산
    knee_hip_square = knee_hip_dot
    shoulder_hip_square = shoulder_hip_dot

    # 두 벡터의 제곱합 계산
    knee_hip_sum_shoulder_hip = knee_hip_square + shoulder_hip_square

    # 벡터 크기 (노름) 계산
    knee_hip_sum_shoulder_hip_sqrt = np.sqrt(knee_hip_sum_shoulder_hip)
    
    # 코사인 유사도 계산
    angle_cosine = knee_hip_simple_dot_sum_shoulder_hip[:,0] / knee_hip_sum_shoulder_hip_sqrt[:,0]
    
    # 각도 계산 (rad 단위)
    angle = np.arccos(angle_cosine)
    
    # 결과 출력 (각도를 degree로 변환하려면 np.degrees() 사용)
    angle_test = np.degrees(angle)  
    nan_indices_angle = np.where(np.isnan(angle_test))

    return nan_indices_angle
    
def encoding(text_encoding):
    # 빈 이미지 생성 (흰색 배경)
    image = np.ones((500, 500, 3), dtype=np.uint8) * 255

    # 이미지를 Pillow 이미지로 변환
    pil_image = Image.fromarray(image)

    # 한글 텍스트를 그리기 위한 폰트 로드 (폰트 파일 경로 지정 필요)
    font_path = "C:/Windows/Fonts/malgun.ttf"  # 시스템에 맞는 폰트 경로 설정 필요
    font = ImageFont.truetype(font_path, 40)

    # Pillow에서 텍스트 그리기
    draw = ImageDraw.Draw(pil_image)
    text = text_encoding
    draw.text((50, 150), text, font=font, fill=(0, 0, 0))

    # 다시 OpenCV 형식으로 변환
    image = np.array(pil_image)
    return image

def video_pose_with_landmarks():

    # 한글을 출력할 폰트 경로 설정 (시스템에 맞는 경로로 설정)
    font_path = "C:/Windows/Fonts/malgun.ttf"  # Windows 예시, 다른 OS는 경로 달라질 수 있음
    font = ImageFont.truetype(font_path, 40)
    data = ld('project_me/squat_front.csv')  # 데이터를 한번만 로드

    model=load_model("model_1.keras")
    
    # Mediapipe 포즈 모듈 초기화
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

    # 웹캠 열기
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 웹캠 영상을 RGB로 변환
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Mediapipe로 포즈 랜드마크 추출
        results = pose.process(frame_rgb)

        # 포즈 랜드마크 그리기 및 관절 값 출력
        if results.pose_landmarks:

                # 랜드마크 데이터 추출
            landmarks = results.pose_landmarks.landmark
         
            # 랜드마크 좌표를 화면에 표시
            for idx,landmark in enumerate(landmarks):
                
               
                # 실제 좌표와 추적된 좌표 비교
                predicted_coordinates = {
                    'shoulder_left': [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].z],
                    # 'shoulder_right': [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].z],
                    'hip_left': [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y, landmarks[mp_pose.PoseLandmark.LEFT_HIP].z],
                    # 'hip_right': [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].z]
                    'knee_left': [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].z],
                    # 'knee_right': (landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].z)
                }


                actual_coordinates = ld("project_me/squat_front.csv")

                
 
                accuracy=1-(90-np.array(calculate_accuracy(actual_coordinates,0)))/90
                
                accuracy=np.array(accuracy, dtype=np.float32)
                     
                #정확도 계산
                # accuracy=ac("project_me/squat_front.csv",45/180)

                #랜드마크 좌표 출력
                height, width, _ = frame.shape
                cx, cy = int(landmark.x * width), int(landmark.y * height)  # 좌표 변환
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)  # 랜드마크 표시
                
                accuracy=np.squeeze(accuracy)
                accuracy_value = accuracy[0].item()
                frame = cv2.putText(frame, f'Accuracy : {accuracy_value:.2f}%', (10, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                
                # 랜드마크 그리기
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 웹캠 영상 출력
        cv2.imshow('Pose Estimation with Landmarks', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 웹캠 종료
    cap.release()
    cv2.destroyAllWindows()

def main():
    video_pose_with_landmarks()
    # data = ld('project_me/squat_front.csv') 
    # actual_coordinates = np.split(data, 3, axis=0)
    # calculate_accuracy(actual_coordinates,0)

if __name__ == "__main__":
    main()