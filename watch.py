import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

# Mediapipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils
# 유클리드 거리 계산
def euclidean_distance(p1, p2):
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
        
        # 정확도 계산 (거리 기반)
        accuracy = max(0, 100 - distance * 100)  # 예시로 거리 * 100을 곱하여 정확도를 0~100 범위로 정규화
        
        total_accuracy += accuracy

    # 평균 정확도
    avg_accuracy = total_accuracy / total_joints
    return avg_accuracy

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
        }

        # 화면 크기 가져오기
        height, width, _ = frame.shape
        
        # 정확도 계산
        accuracy = calculate_accuracy(transformed_coordinates, user_coordinates)
        cv2.putText(frame, f"Accuracy: {accuracy:.2f}%", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

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