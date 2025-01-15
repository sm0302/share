import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import math
import time
from PIL import Image, ImageDraw, ImageFont

class SquatEvaluation:
    def __init__(self, user_coordinates):
        self.df_my_side = user_coordinates
        self.lowest_frame_side = self.find_lowest_frame(self.df_my_side)
        self.lowest_frame_my_side = self.find_lowest_frame(self.df_my_side)
        self.highest_frame_side = self.find_highest_frame(self.df_my_side)
        self.highest_frame_my_side = self.find_highest_frame(self.df_my_side)
        self.side = self.left_or_right(self.df_my_side)

    def find_lowest_frame(self, df):
        return max(df['Left Hip'])

    def find_highest_frame(self, df):
        return min(df['Left Hip'])

    def left_or_right(self, df):
        return 'Left' if df['Left Hip'] > df['Left Knee'] else 'Right'

    def calculate_angle(self, df):
        df_lowest_shoulder = df['Left Shoulder']
        df_lowest_hip = df['Left Hip']

        x1, y1, z1 = df_lowest_shoulder[0], df_lowest_shoulder[1], df_lowest_shoulder[2]
        x2, y2, z2 = df_lowest_hip[0], df_lowest_hip[1], df_lowest_hip[2]

        vx = x2 - x1
        vy = y2 - y1
        vz = z2 - z1

        v_magnitude = math.sqrt(vx**2 + vy**2 + vz**2)
        cos_theta = vy / v_magnitude
        theta_radian = math.acos(cos_theta)
        return abs(90 - math.degrees(theta_radian))

    def calculate_hip_knee(self, df):
        hip_y = max(df['Left Hip'])
        knee_y = max(df['Left Knee'])

        hip_knee_diff = abs(hip_y - knee_y)
        hip_compare_knee = '아래' if hip_y > knee_y else '위'
        return hip_compare_knee, hip_knee_diff

    def evaluate(self):
        decent_score = 0

        angle = self.calculate_angle(self.df_side)
        decent_score += abs(round(angle - 55, 4)) if angle < 55 else abs(round(angle - 70, 4)) if angle > 70 else 0

        _, knee_foot_diff = self.calculate_knee_foot(self.df_side, self.lowest_frame_side, self.side)
        decent_score += abs(knee_foot_diff + 0.03) * 100 if knee_foot_diff < -0.03 else 0

        hip_compare_knee, hip_knee_diff = self.calculate_hip_knee(self.df_side, self.lowest_frame_side, self.side)
        decent_score += abs(hip_knee_diff + 0.05) * 100 if hip_knee_diff < -0.05 else 0

        _, distance_diff = self.compare_knee_foot_distance(self.df_front, self.lowest_frame_front)
        decent_score += abs(distance_diff + 0.01) * 100 if distance_diff < -0.01 else 0

        return round((100 - decent_score), 2)


# Mediapipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils
# 유클리드 거리 계산
def euclidean_distance(p1, p2):
    p2=np.array(p2)
    p1=np.array(p1)

    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 + (p2[2] - p1[2])**2)

# 정확도 계산
def calculate_accuracy(target_pose, user_pose):
    total_accuracy = 0
    total_joints = len(target_pose)

    

    for joint in target_pose:
        target_coords = target_pose[joint]
        user_coords = user_pose[joint]


        # 유클리드 거리 계산
        distance = euclidean_distance(target_coords, user_coords)
        distance=np.array(distance)
        zero=np.zeros(3)
        hundred=np.array([100,100,100])
        # 정확도 계산 (거리 기반)
        accuracy = np.maximum(zero, hundred - distance * 100).max()  # 예시로 거리 * 100을 곱하여 정확도를 0~100 범위로 정규화
        
        total_accuracy += accuracy

    # 평균 정확도
    avg_accuracy = total_accuracy / total_joints
    return avg_accuracy
def feedback(a):
        
        feedback = ""
        evaluator = SquatEvaluation(a)
        angle = evaluator.calculate_angle(a)
        if angle < 55:
            feedback = '허리가 너무 굽혀졌습니다.'
        elif angle > 70:
            feedback = '허리가 너무 펴졌습니다.'

        # _, knee_foot_diff = self.calculate_knee_foot(self.df_side, self.lowest_frame_side, self.side)
        # if knee_foot_diff < -0.03:
        #     feedback.append('무릎이 너무 앞으로 나와 있어요')

        hip_compare_knee, hip_knee_diff = evaluator.calculate_hip_knee(a)
        if hip_knee_diff < -0.05 and hip_compare_knee == '위':
            feedback = '더 앉아 주세요'

        # _, distance_diff = self.compare_knee_foot_distance(self.df_front, self.lowest_frame_front)
        # if distance_diff < -0.01:
        #     feedback.append('무릎을 너무 모았어요')

        if not feedback:
            feedback = '정확한 자세로 스쿼트를 진행하셨습니다.'

        return feedback
# 피드백을 화면에 출력하는 부분
def show_feedback(frame, feedback_messages):
    
    message_encode=feedback_messages.encode('utf-8')
    message_decode=message_encode.decode('utf-8')
    # 피드백 메시지를 화면에 표시
    font_path = "C:/Windows/Fonts/malgun.ttf"
    # PIL에서 한글을 지원하는 폰트로 텍스트 그리기
    pil_img = Image.fromarray(frame)  # OpenCV 이미지를 PIL 이미지로 변환
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.truetype(font_path, 30)  # 폰트 크기 조정
    # 텍스트 추가
    draw.text((10, 200), message_decode, font=font, fill=(0, 255, 0))  # 텍스트 색상 (녹색)
    #화면 보이기
    frame = np.array(pil_img)
    return frame

# 데이터 불러오기 함수
def load_data(data):
    # CSV 데이터 불러오기
    dataFrame = pd.read_csv(data)
    
    # 좌표 데이터 추출
    shoulder = dataFrame.loc[:, ['Left Shoulder x', 'Left Shoulder y', 'Left Shoulder z']]
    shoulder2 = dataFrame.loc[:, ['Right Shoulder x', 'Right Shoulder y', 'Right Shoulder z']]
    hip = dataFrame.loc[:, ['Left Hip x', 'Left Hip y', 'Left Hip z']]
    hip2 = dataFrame.loc[:, ['Right Hip x', 'Right Hip y', 'Right Hip z']]
    knee = dataFrame.loc[:, ['Left Knee x', 'Left Knee y', 'Left Knee z']]
    knee2 = dataFrame.loc[:, ['Right Knee x', 'Right Knee y', 'Right Knee z']]
    foot = dataFrame.loc[:, ['Left Foot Index x', 'Left Foot Index y', 'Left Foot Index z']]
    foot2 = dataFrame.loc[:, ['Right Foot Index x', 'Right Foot Index y', 'Right Foot Index z']]


    # 데이터를 딕셔너리로 변환 (리스트 형태로 변환됨)
    shoulder = {key: list(value.values()) for key, value in shoulder.to_dict().items()}
    shoulder2 = {key: list(value.values()) for key, value in shoulder2.to_dict().items()}
    hip = {key: list(value.values()) for key, value in hip.to_dict().items()}
    hip2 = {key: list(value.values()) for key, value in hip2.to_dict().items()}
    knee = {key: list(value.values()) for key, value in knee.to_dict().items()}
    knee2 = {key: list(value.values()) for key, value in knee2.to_dict().items()}
    foot = {key: list(value.values()) for key, value in foot.to_dict().items()}
    foot2 = {key: list(value.values()) for key, value in foot2.to_dict().items()}


    return shoulder, shoulder2, hip, hip2, knee, knee2, foot, foot2

def evaluate_squat(frame, landmarks, width, height, data):
    # 현재 자세의 y 좌표값 계산
    shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
    shoulder2_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
    hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP].y
    hip2_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y
    knee_y = landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y
    knee2_y = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y
    data=load_data('project_me/squat_Lside.csv')
    # shoulder_y 값을 기준으로 자세 평가 구분
    if shoulder_y > data[0]['Left Shoulder y'][0]:  # 스쿼트 중간 자세로 분류되는 높이 기준
        # 스쿼트 중간 자세 평가
        if hip_y < max(data[2]['Left Hip y']) + 0.15 and \
           hip_y > max(data[2]['Left Hip y']) - 0.15 and \
           hip2_y < max(data[3]['Right Hip y']) + 0.15 and \
           hip2_y > max(data[3]['Right Hip y']) - 0.15 and \
           knee_y < max(data[4]['Left Knee y']) + 0.15 and \
           knee_y > max(data[4]['Left Knee y']) - 0.15 and \
           knee2_y < max(data[5]['Right Knee y']) + 0.15 and \
           knee2_y > max(data[5]['Right Knee y']) - 0.15:
            cv2.putText(frame, 'GOOD (Mid)', (width//2, height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        elif hip_y <= max(data[2]['Left Hip y']) + 0.08 and \
             hip_y >= max(data[2]['Left Hip y']) - 0.08 and \
             hip2_y <= max(data[3]['Right Hip y']) + 0.08 and \
             hip2_y >= max(data[3]['Right Hip y']) - 0.08 and \
             knee_y <= max(data[4]['Left Knee y']) + 0.08 and \
             knee_y >= max(data[4]['Left Knee y']) - 0.08 and \
             knee2_y <= max(data[5]['Right Knee y']) + 0.08 and \
             knee2_y >= max(data[5]['Right Knee y']) - 0.08:
            cv2.putText(frame, 'PERFECT (Mid)', (width//2, height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, 'BAD (Mid)', (width//2, height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    else:  # shoulder_y <= 0.4: 스쿼트 처음 자세로 분류되는 높이 기준
        # 스쿼트 처음 자세 평가
        if shoulder_y < data[0]['Left Shoulder y'][0] + 0.15 and \
           shoulder_y > data[0]['Left Shoulder y'][0] - 0.15 and \
           shoulder2_y < data[1]['Right Shoulder y'][0] + 0.15 and \
           shoulder2_y > data[1]['Right Shoulder y'][0] - 0.15 and \
           knee_y < data[4]['Left Knee y'][0] + 0.15 and \
           knee_y > data[4]['Left Knee y'][0] - 0.15 and \
           knee2_y < data[5]['Right Knee y'][0] + 0.15 and \
           knee2_y > data[5]['Right Knee y'][0] - 0.15:
            cv2.putText(frame, 'GOOD (Start)', (width//2, height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        elif shoulder_y <= max(data[0]['Left Shoulder y']) + 0.08 and \
             shoulder_y >= data[0]['Left Shoulder y'][0] - 0.08 and \
             shoulder2_y <= data[1]['Right Shoulder y'][0] + 0.08 and \
             shoulder2_y >= data[1]['Right Shoulder y'][0] - 0.08 and \
             knee_y <= data[4]['Left Knee y'][0] + 0.08 and \
             knee_y >= data[4]['Left Knee y'][0] - 0.08 and \
             knee2_y <= data[5]['Right Knee y'][0] + 0.08 and \
             knee2_y >= data[5]['Right Knee y'][0] - 0.08:
            cv2.putText(frame, 'PERFECT (Start)', (width//2, height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, 'BAD (Start)', (width//2, height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    return frame

def evaluate(angle,hip_knee_diff):
        decent_score = 0
        if angle < 55:
            decent_score += abs(round(angle - 55, 4))
        elif angle > 70:
            decent_score += abs(round(angle - 70, 4))
        # if knee_foot_diff[1] < -0.03:
        #     decent_score += abs(knee_foot_diff[1] + 0.03) * 100
        if hip_knee_diff[1] < -0.05:
            decent_score += abs(hip_knee_diff[1] + 0.05) * 100
        # if distance_diff[1] < -0.01:
        #     decent_score += abs(distance_diff[1] + 0.01) * 100
        print(f'점수: {round((100 - decent_score), 2)}')
        return round((100 - decent_score), 2)

# 데이터 불러오기
def data_load(data_path):
    data = load_data(data_path)

    # 관절값 딕셔너리화(각각의 키에 x, y, z 값을 각각 저장)

    shoulder_target = {
        'Left Shoulder x': data[0]['Left Shoulder x'],
        'Left Shoulder y': data[0]['Left Shoulder y'],
        'Left Shoulder z': data[0]['Left Shoulder z'],
        'Right Shoulder x': data[1]['Right Shoulder x'],
        'Right Shoulder y': data[1]['Right Shoulder y'],
        'Right Shoulder z': data[1]['Right Shoulder z']
    }
    hip_target = {
        'Left Hip x': data[2]['Left Hip x'],
        'Left Hip y': data[2]['Left Hip y'],
        'Left Hip z': data[2]['Left Hip z'],
        'Right Hip x': data[3]['Right Hip x'],
        'Right Hip y': data[3]['Right Hip y'],
        'Right Hip z': data[3]['Right Hip z']
    }
    knee_target = {
        'Left Knee x': data[4]['Left Knee x'],
        'Left Knee y': data[4]['Left Knee y'],
        'Left Knee z': data[4]['Left Knee z'],
        'Right Knee x': data[5]['Right Knee x'],
        'Right Knee y': data[5]['Right Knee y'],
        'Right Knee z': data[5]['Right Knee z']
    }
    foot_target = {
        'Left Foot Index x': data[6]['Left Foot Index x'],
        'Left Foot Index y': data[6]['Left Foot Index y'],
        'Left Foot Index z': data[6]['Left Foot Index z'],
        'Right Foot Index x': data[7]['Right Foot Index x'],
        'Right Foot Index y': data[7]['Right Foot Index y'],
        'Right Foot Index z': data[7]['Right Foot Index z']
    }
    #튜플 리스트로 변환
    target_coordinates = {
        'Left Shoulder': [(shoulder_target['Left Shoulder x'][i], 
                        shoulder_target['Left Shoulder y'][i], 
                        shoulder_target['Left Shoulder z'][i]) for i in range(len(shoulder_target['Left Shoulder x']))],
        'Right Shoulder': [(shoulder_target['Right Shoulder x'][i], 
                        shoulder_target['Right Shoulder y'][i], 
                        shoulder_target['Right Shoulder z'][i]) for i in range(len(shoulder_target['Right Shoulder x']))],

        'Left Hip': [(hip_target['Left Hip x'][i], 
                    hip_target['Left Hip y'][i], 
                    hip_target['Left Hip z'][i]) for i in range(len(hip_target['Left Hip x']))],
        'Right Hip': [(hip_target['Right Hip x'][i], 
                    hip_target['Right Hip y'][i], 
                    hip_target['Right Hip z'][i]) for i in range(len(hip_target['Right Hip x']))],

        'Left Knee': [(knee_target['Left Knee x'][i], 
                    knee_target['Left Knee y'][i], 
                    knee_target['Left Knee z'][i]) for i in range(len(knee_target['Left Knee x']))],

        'Right Knee': [(knee_target['Right Knee x'][i], 
                    knee_target['Right Knee y'][i], 
                    knee_target['Right Knee z'][i]) for i in range(len(knee_target['Right Knee x']))],
        
        'Left Foot': [(foot_target['Left Foot Index x'][i], 
                    foot_target['Left Foot Index y'][i], 
                    foot_target['Left Foot Index z'][i]) for i in range(len(foot_target['Left Foot Index x']))],
        
        'Right Foot': [(foot_target['Right Foot Index x'][i], 
                    foot_target['Right Foot Index y'][i], 
                    foot_target['Right Foot Index z'][i]) for i in range(len(foot_target['Right Foot Index x']))]
    }

    #튜플 리스트를 1차원 리스트로 변환
    transformed_coordinates = {
        'Left Shoulder': [coord for tup in target_coordinates['Left Shoulder'] for coord in tup],
        'Left Hip': [coord for tup in target_coordinates['Left Hip'] for coord in tup],
        'Left Knee': [coord for tup in target_coordinates['Left Knee'] for coord in tup],
        'Right Shoulder': [coord for tup in target_coordinates['Right Shoulder'] for coord in tup],
        'Right Hip': [coord for tup in target_coordinates['Right Hip'] for coord in tup],
        'Right Knee': [coord for tup in target_coordinates['Right Knee'] for coord in tup],
        'Left Foot': [coord for tup in target_coordinates['Left Foot'] for coord in tup],
        'Right Foot': [coord for tup in target_coordinates['Right Foot'] for coord in tup]
    }
    return transformed_coordinates
# 웹캠 열기
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 영상 RGB 변환
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    
    # Mediapipe로 관절값 추출
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # 현재 관절값 추출
        user_coordinates = {
            'Left Shoulder': [
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y,
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].z,
            ],
            'Left Hip': [
                landmarks[mp_pose.PoseLandmark.LEFT_HIP].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP].y,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP].z,
            ],
            'Left Knee': [
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x,
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y,
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE].z,
            ],
            'Right Shoulder': [
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y,
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].z,
            ],
            'Right Hip': [
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y,
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP].z,
            ],
            'Right Knee': [
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y,
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].z,
            ],
            'Left Foot': [
                landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x,
                landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y,
                landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].z,
            ],
            'Right Foot': [
                landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y,
                landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].z,
            ],
        }

        # 화면 크기 가져오기
        height, width, _ = frame.shape
        
        # 정확도 계산
        # accuracy = calculate_accuracy(transformed_coordinates, user_coordinates)

        # 피드백 메시지 출력
        feedback_messages=[]
        feedback_messages = feedback(user_coordinates)
        frame = show_feedback(frame, feedback_messages)
        
        # 평가 점수 계산
        evaluator = SquatEvaluation(user_coordinates)
    
        squat_angle = evaluator.calculate_angle(user_coordinates)
        # squat_knee_foot = evaluator.calculate_knee_foot(evaluator.df_my_side)
        squat_hip_knee = evaluator.calculate_hip_knee(user_coordinates)
        # squat_distance = evaluator.compare_knee_foot_distance(evaluator.df_my_side, evaluator.lowest_frame_front)

        squat_score = evaluate(squat_angle, squat_hip_knee)

        #자세 분석
        frame = evaluate_squat(frame, landmarks, width, height, data_load('project_me/squat_Lside.csv'))
        # 평가 점수를 화면에 표시
        cv2.putText(frame, f'Accuracy: {squat_score:.2f}%', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        # Mediapipe 랜드마크 그리기
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # 화면 출력
    cv2.imshow("Posture Feedback", frame)

    # 'q'를 눌러 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()