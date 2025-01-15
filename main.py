import cv2
import csv
import mediapipe as mp
from datetime import datetime
import pandas as pd
import math
from watch_score_x import SquatEvaluation as SE
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cProfile
import threading
import time
import multiprocessing

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
        evaluator = SE(a)
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
def show_feedback(frame,feedback_messages):
    
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
    evaluator = SquatDataRecorder()
    
    # 현재 자세의 y 좌표값 계산
    shoulder_y = landmarks[evaluator.mp_pose.PoseLandmark.LEFT_SHOULDER].y
    shoulder2_y = landmarks[evaluator.mp_pose.PoseLandmark.RIGHT_SHOULDER].y
    hip_y = landmarks[evaluator.mp_pose.PoseLandmark.LEFT_HIP].y
    hip2_y = landmarks[evaluator.mp_pose.PoseLandmark.RIGHT_HIP].y
    knee_y = landmarks[evaluator.mp_pose.PoseLandmark.LEFT_KNEE].y
    knee2_y = landmarks[evaluator.mp_pose.PoseLandmark.RIGHT_KNEE].y
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

class SquatDataRecorder:
    def __init__(self):
        # MediaPipe 및 OpenCV 초기화
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.cap = cv2.VideoCapture(0)
        
        # 영상 저장 설정
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = 30
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # 파일명 관련 변수
        self.side_video_filename = None
        self.front_video_filename = None
        self.side_csv_filename = None
        self.front_csv_filename = None
        
        # 녹화 상태 변수
        self.recording = False
        self.recording_count = 0
        self.frame_count = 0
        self.start_time = None
        self.out = None
        
        # 데이터 저장 변수
        self.side_data = []
        self.front_data = []
    
    def main_data(self):
        """메인 녹화 루프"""
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print("웹캠에서 영상을 가져올 수 없습니다.")
                break
            pose = self.mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
            time.sleep(0.01)
            # Mediapipe Pose 처리
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            print("측면 영상 촬영을 위해 준비해주세요.")
            print("Enter를 누르면 측면 영상 촬영이 시작됩니다.")
            # Mediapipe로 관절값 추출
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark


                # 현재 관절값 추출
                user_coordinates = {
                    'Left Shoulder': [
                        landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].z,
                    ],
                    'Left Hip': [
                        landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].y,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].z,
                    ],
                    'Left Knee': [
                        landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE].y,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE].z,
                    ],
                    'Right Shoulder': [
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y,
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].z,
                    ],
                    'Right Hip': [
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].x,
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].y,
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].z,
                    ],
                    'Right Knee': [
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE].x,
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE].y,
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE].z,
                    ],
                    'Left Foot': [
                        landmarks[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX].z,
                    ],
                    'Right Foot': [
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x,
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y,
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].z,
                    ],
                }

                # 화면 크기 가져오기
                height, width, _ = frame.shape
                
                # 정확도 계산
                # accuracy = calculate_accuracy(transformed_coordinates, user_coordinates)

                # 피드백 메시지 출력
                feedback_messages=[]
                feedback_messages = feedback(user_coordinates)
                # frame = show_feedback(frame,feedback_messages)
                
                # 평가 점수 계산
                

                squat_angle = SE.calculate_angle(self,user_coordinates)
                # squat_knee_foot = evaluator.calculate_knee_foot(evaluator.df_my_side)
                squat_hip_knee = SE.calculate_hip_knee(self,user_coordinates)
                # squat_distance = evaluator.compare_knee_foot_distance(evaluator.df_my_side, evaluator.lowest_frame_front)

                squat_score = evaluate(squat_angle, squat_hip_knee)

                #자세 분석
                # frame = evaluate_squat(frame, landmarks, width, height, data_load('project_me/squat_Lside.csv'))
                
                # 평가 점수를 화면에 표시
                cv2.putText(frame, f'Accuracy: {squat_score:.2f}%', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                # 랜드마크 시각화 및 데이터 수집
                self._process_frame(frame, results)
                # 녹화 중일 때 처리
                if self.recording:
                    self._handle_recording(frame)

                # 'q'를 # 화면에 현재 상태 표시
                self._display_status(frame)
            
            cv2.imshow('Pose Estimation', frame)

            # 키보드 입력 처리
            self._handle_keyboard_input()
                    

                   

    def _process_frame(self, frame, results):
        """프레임별 포즈 데이터 처리"""
        self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        
        landmarks = results.pose_landmarks.landmark
        row = [self.frame_count]

        for idx in [11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]:
            landmark = landmarks[idx]
            if landmark.visibility < 0.8:
                row.extend([None, None, None])
            else:
                row.extend([landmark.x, landmark.y, landmark.z])

        if self.recording_count == 0:
            self.side_data.append(row)
        else:
            self.front_data.append(row)
        self.frame_count += 1

    def _handle_recording(self, frame):
        """녹화 처리 및 시간 체크"""
        elapsed_time = (datetime.now() - self.start_time).total_seconds()
        self.out.write(frame)

        if elapsed_time >= 5.0:
            self._save_recording()

    def _save_recording(self):
        """녹화 종료 및 파일 저장"""
        self.recording = False
        self.recording_count += 1
        
        if self.out:
            self.out.release()
            self.out = None

        # CSV 파일 저장
        current_video_filename = self.side_video_filename if self.recording_count == 1 else self.front_video_filename
        csv_filename = current_video_filename.replace('.mp4', '_pose_data.csv')
        current_data = self.side_data if self.recording_count == 1 else self.front_data
        
        self._save_csv_file(csv_filename, current_data)
        
        if self.recording_count == 1:
            self.side_csv_filename = csv_filename
            print("\n정면 영상 촬영을 위해 준비해주세요.")
            print("Enter를 누르면 정면 영상 촬영이 시작됩니다.")
        else:
            self.front_csv_filename = csv_filename
            print("\n촬영이 모두 완료되었습니다.")
            print("'a'를 누르면 분석을 시작합니다.")
            print("'q'를 누르면 종료합니다.")

    def _save_csv_file(self, filename, data):
        """CSV 파일로 데이터 저장"""
        with open(filename, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            header = ['Frame']
            for joint in ['Left Shoulder', 'Right Shoulder', 'Left Hip', 'Right Hip',
                        'Left Knee', 'Right Knee', 'Left Ankle', 'Right Ankle',
                        'Left Heel', 'Right Heel', 'Left Foot Index', 'Right Foot Index']:
                header.extend([f'{joint} x', f'{joint} y', f'{joint} z'])
            csv_writer.writerow(header)
            
            for row in sorted(data, key=lambda x: x[0]):
                csv_writer.writerow(row)
        print(f"CSV 파일 저장 완료: {filename}")

    def _display_status(self, frame):
        """화면에 현재 상태 표시"""
        if self.recording:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            status_text = f"Recording... {5-elapsed:.1f}s"
        else:
            if self.recording_count == 0:
                status_text = "Side view recording standby..."
            elif self.recording_count == 1:
                status_text = "Front view recording standby..."
            else:
                status_text = "Recording complete. Press 'a' to analyze, 'q' to quit"

        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    def _handle_keyboard_input(self):
        """키보드 입력 처리"""
        key = cv2.waitKey(1) & 0xFF
        time.sleep(0.01)
        if key == ord('q'):  # q키는 항상 종료
            self.cap.release()
            if self.out is not None:
                self.out.release()
            cv2.destroyAllWindows()
            return False
            
        elif key == ord('a') and self.recording_count == 2:  # 두 영상 녹화 완료 후 a키로 분석 시작
            if self.out is not None:
                self.out.release()
                self.out = None
            return False
            
        elif key == 13 and not self.recording and self.recording_count < 2:  # Enter 키로 녹화 시작
            self._start_recording()
            
        return True

    def _start_recording(self):
        """녹화 시작 처리"""
        self.recording = True
        self.frame_count = 0
        self.start_time = datetime.now()
        formatted_time = self.start_time.strftime("%Y-%m-%d_%H-%M-%S")
        output_filename = f"{formatted_time}_{'side' if self.recording_count == 0 else 'front'}.mp4"
        
        if self.recording_count == 0:
            self.side_video_filename = output_filename
        else:
            self.front_video_filename = output_filename
        
        self.out = cv2.VideoWriter(output_filename, self.fourcc, self.fps, (self.frame_width, self.frame_height))
        print("녹화를 시작합니다...")

    def record_squat_data(self):
        return self.side_csv_filename, self.front_csv_filename
    
if __name__ == "__main__":
    SquatDataRecorder().main_data()

    
    