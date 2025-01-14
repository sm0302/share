import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import math
import time
from PIL import Image, ImageDraw, ImageFont

class SquatEvaluation:
    def __init__(self, side_data_path, front_data_path):
        self.df_side = pd.read_csv(side_data_path)
        self.df_front = pd.read_csv(front_data_path)

        # 가장 아래로 내려갔을 때의 프레임 값 추출
        self.lowest_frame_side = self.find_lowest_frame(self.df_side)
        self.lowest_frame_front = self.find_lowest_frame(self.df_front)

        # 가장 위로 올라갔을 때의 프레임 값 추출
        self.highest_frame_side = self.find_highest_frame(self.df_side)
        self.highest_frame_front = self.find_highest_frame(self.df_front)

        # 측면 영상이 왼쪽인지 오른쪽인지 판별
        self.side = self.left_or_right(self.df_side, self.lowest_frame_side)

        # 축척 계산
        self.scale_side, self.scale_front = self.scale(self.df_side, self.df_front, self.highest_frame_side, self.highest_frame_front)

    def find_lowest_frame(self, df):
        max_value = df[['Left Hip y', 'Right Hip y']].max().max()
        max_row_index = df[(df['Left Hip y'] == max_value) | (df['Right Hip y'] == max_value)].index[0]
        return df.loc[max_row_index, 'Frame']

    def find_highest_frame(self, df):
        min_value = df[['Left Hip y', 'Right Hip y']].min().min()
        min_row_index = df[(df['Left Hip y'] == min_value) | (df['Right Hip y'] == min_value)].index[0]
        return df.loc[min_row_index, 'Frame']

    def left_or_right(self, df, lowest_frame_side):
        if df.loc[lowest_frame_side, 'Left Hip x'] > df.loc[lowest_frame_side, 'Left Knee x']:
            return 'Left'
        else:
            return 'Right'

    def calculate_angle(self, df, lowest_frame_side, side):
        df_lowest_shoulder = df[df['Frame'] == lowest_frame_side][[f'{side} Shoulder x', f'{side} Shoulder y', f'{side} Shoulder z']]
        df_lowest_hip = df[df['Frame'] == lowest_frame_side][[f'{side} Hip x', f'{side} Hip y', f'{side} Hip z']]

        x1, y1, z1 = df_lowest_shoulder.iloc[0]
        x2, y2, z2 = df_lowest_hip.iloc[0]

        vx = x2 - x1
        vy = y2 - y1
        vz = z2 - z1

        v_magnitude = math.sqrt(vx**2 + vy**2 + vz**2)
        dot_product = vy  # 내적은 y축 방향 성분

        cos_theta = dot_product / v_magnitude
        theta_radian = math.acos(cos_theta)
        theta_degree = abs(90 - math.degrees(theta_radian))
        return theta_degree

    def calculate_knee_foot(self, df, lowest_frame_side, side):
        df_lowest_knee = df[df['Frame'] == lowest_frame_side][[f'{side} Knee x', f'{side} Knee y', f'{side} Knee z']]
        df_lowest_foot = df[df['Frame'] == lowest_frame_side][[f'{side} Foot Index x', f'{side} Foot Index y', f'{side} Foot Index z']]

        knee_x = df_lowest_knee.iloc[0][f'{side} Knee x']
        foot_x = df_lowest_foot.iloc[0][f'{side} Foot Index x']

        if side == 'Left':
            if knee_x > foot_x:
                knee_compare_foot = '뒤'
            else:
                knee_compare_foot = '앞'
        else:
            if knee_x < foot_x:
                knee_compare_foot = '뒤'
            else:
                knee_compare_foot = '앞'

        return knee_compare_foot, abs(knee_x - foot_x) / self.scale_side

    def calculate_hip_knee(self, df, lowest_frame_side, side):
        df_hip_y = df[df['Frame'] == lowest_frame_side][[f'{side} Hip y']]
        df_knee_y = df[df['Frame'] == lowest_frame_side][[f'{side} Knee y']]

        hip_y = df_hip_y.iloc[0][f'{side} Hip y']
        knee_y = df_knee_y.iloc[0][f'{side} Knee y'] 

        if hip_y > knee_y:
            hip_compare_knee = '아래'
        else:
            hip_compare_knee = '위'

        return hip_compare_knee, abs(hip_y - knee_y) / self.scale_side

    def compare_knee_foot_distance(self, df, lowest_frame_front):
        df_lowest_knee = df[df['Frame'] == lowest_frame_front][['Left Knee x', 'Right Knee x']]
        df_lowest_foot = df[df['Frame'] == lowest_frame_front][['Left Ankle x', 'Right Ankle x']]

        knee_distance = abs(df_lowest_knee.iloc[0]['Left Knee x'] - df_lowest_knee.iloc[0]['Right Knee x'])
        foot_distance = abs(df_lowest_foot.iloc[0]['Left Ankle x'] - df_lowest_foot.iloc[0]['Right Ankle x'])

        if knee_distance > foot_distance:
            knee_foot_distance = '벌어'
        else:
            knee_foot_distance = '좁혀'

        return knee_foot_distance, (knee_distance - foot_distance) / self.scale_front

    def scale(self, df_side, df_front, highest_frame_side, highest_frame_front):
        # 측면 영상 어깨와 발목 평균 계산
        shoulder_side_mean = df_side[df_side['Frame'] == highest_frame_side][['Left Shoulder y', 'Right Shoulder y']].mean(axis=1).iloc[0]
        ankle_side_mean = df_side[df_side['Frame'] == highest_frame_side][['Left Ankle y', 'Right Ankle y']].mean(axis=1).iloc[0]

        # 정면 영상 어깨와 발목 평균 계산
        shoulder_front_mean = df_front[df_front['Frame'] == highest_frame_front][['Left Shoulder y', 'Right Shoulder y']].mean(axis=1).iloc[0]
        ankle_front_mean = df_front[df_front['Frame'] == highest_frame_front][['Left Ankle y', 'Right Ankle y']].mean(axis=1).iloc[0]

        # 어깨-발목 y 좌표 차이 계산
        scale_side = abs(shoulder_side_mean - ankle_side_mean)
        scale_front = abs(shoulder_front_mean - ankle_front_mean)

        return scale_side, scale_front

    def feedback(self):
        # 각도 계산
        feedback = []
        angle = self.calculate_angle(self.df_side, self.lowest_frame_side, self.side)
        if angle < 55:
            feedback.append('허리가 너무 굽혀졌습니다.')
        elif angle > 70:
            feedback.append('허리가 너무 펴졌습니다.')

        # 무릎과 발의 위치 비교
        _, knee_foot_diff = self.calculate_knee_foot(self.df_side, self.lowest_frame_side, self.side)
        if knee_foot_diff > 0.05:
            feedback.append('무릎이 너무 앞으로 나와 있어요')

        # 골반과 무릎의 높이 비교
        hip_compare_knee, hip_knee_diff = self.calculate_hip_knee(self.df_side, self.lowest_frame_side, self.side)
        if hip_knee_diff > 0.1 and hip_compare_knee == '위':
            feedback.append('더 앉아 주세요')

        # 무릎과 발의 거리 비교
        _, distance_diff = self.compare_knee_foot_distance(self.df_front, self.lowest_frame_front)
        if distance_diff < -0.01:
            feedback.append('무릎을 너무 모았어요')

        if not feedback:
            feedback.append('정확한 자세로 스쿼트를 진행하셨습니다.')

        for message in feedback:
            return(message)

    def evaluate(self):
        decent_score = 0
        angle = self.calculate_angle(self.df_side, self.lowest_frame_side, self.side)
        if angle < 55:
            decent_score_1 = abs(round(angle - 55, 4))
        elif angle > 70:
            decent_score_1 = abs(round(angle - 70, 4))
        else :
            decent_score_1 = 0
        decent_score += decent_score_1
    
        _, knee_foot_diff = self.calculate_knee_foot(self.df_side, self.lowest_frame_side, self.side)
        if knee_foot_diff > 0.05:
            decent_score_2 = abs(knee_foot_diff - 0.05) * 100
        else :
            decent_score_2 = 0
        decent_score += decent_score_2
        
        hip_compare_knee, hip_knee_diff = self.calculate_hip_knee(self.df_side, self.lowest_frame_side, self.side)
        if hip_knee_diff > 0.1 and hip_compare_knee == '위':
            decent_score_3 = abs(hip_knee_diff - 0.1) * 100
        else :
            decent_score_3 = 0
        decent_score += decent_score_3   
        
        _, distance_diff = self.compare_knee_foot_distance(self.df_front, self.lowest_frame_front)
        if distance_diff < -0.01:
            decent_score_4 = abs(distance_diff - 0.01) * 100
        else :
            decent_score_4 = 0
        decent_score += decent_score_4

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
    frame = np.array(pil_img)
    return frame

# 데이터 불러오기 함수
def load_data(data):
    # CSV 데이터 불러오기
    dataFrame = pd.read_csv(data)
    
    # 좌표 데이터 추출
    shoulder = dataFrame.loc[:, ['Left Shoulder x', 'Left Shoulder y', 'Left Shoulder z']]
    hip = dataFrame.loc[:, ['Left Hip x', 'Left Hip y', 'Left Hip z']]
    knee = dataFrame.loc[:, ['Left Knee x', 'Left Knee y', 'Left Knee z']]
    
    # 데이터를 딕셔너리로 변환 (리스트 형태로 변환됨)
    shoulder = {key: list(value.values()) for key, value in shoulder.to_dict().items()}
    hip = {key: list(value.values()) for key, value in hip.to_dict().items()}
    knee = {key: list(value.values()) for key, value in knee.to_dict().items()}
    
    return shoulder, hip, knee

# 데이터 불러오기
data = load_data('project_me/squat_Lside.csv')

# 관절값 딕셔너리화(각각의 키에 x, y, z 값을 각각 저장)
shoulder_target = {
    'Left Shoulder x': data[0]['Left Shoulder x'],
    'Left Shoulder y': data[0]['Left Shoulder y'],
    'Left Shoulder z': data[0]['Left Shoulder z']
}
hip_target = {
    'Left Hip x': data[1]['Left Hip x'],
    'Left Hip y': data[1]['Left Hip y'],
    'Left Hip z': data[1]['Left Hip z']
}
knee_target = {
    'Left Knee x': data[2]['Left Knee x'],
    'Left Knee y': data[2]['Left Knee y'],
    'Left Knee z': data[2]['Left Knee z']
}

target_coordinates = {
    'Left Shoulder': [(shoulder_target['Left Shoulder x'][i], 
                       shoulder_target['Left Shoulder y'][i], 
                       shoulder_target['Left Shoulder z'][i]) for i in range(len(shoulder_target['Left Shoulder x']))],
    
    'Left Hip': [(hip_target['Left Hip x'][i], 
                  hip_target['Left Hip y'][i], 
                  hip_target['Left Hip z'][i]) for i in range(len(hip_target['Left Hip x']))],
    
    'Left Knee': [(knee_target['Left Knee x'][i], 
                   knee_target['Left Knee y'][i], 
                   knee_target['Left Knee z'][i]) for i in range(len(knee_target['Left Knee x']))]
}

transformed_coordinates = {
    'Left Shoulder': [coord for tup in target_coordinates['Left Shoulder'] for coord in tup],
    'Left Hip': [coord for tup in target_coordinates['Left Hip'] for coord in tup],
    'Left Knee': [coord for tup in target_coordinates['Left Knee'] for coord in tup]
}


# 웹캠 열기
cap = cv2.VideoCapture(0)

# SquatEvaluation 클래스 인스턴스 생성
side_data_path = 'project_me/squat_Lside.csv'  # 실제 경로로 변경
front_data_path = 'project_me/squat_front.csv'  # 실제 경로로 변경
squat_eval = SquatEvaluation(side_data_path, front_data_path)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 영상 RGB 변환
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    feedback_messages = []
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
        }

        # 화면 크기 가져오기
        height, width, _ = frame.shape
        
        # 정확도 계산
        accuracy = calculate_accuracy(target_coordinates, user_coordinates)

        # 피드백 메시지 출력
        feedback_messages=[]
        feedback_messages = squat_eval.feedback()
        frame = show_feedback(frame, feedback_messages)
            
        # 평가 점수 계산
        squat_score = squat_eval.evaluate()

        # 화면 크기 가져오기
        height, width, _ = frame.shape

        # 각도와 평가 점수를 화면에 표시
        cv2.putText(frame, f'Accuracy: {accuracy:.2f}%', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f'Squat Score: {squat_score}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)


        
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