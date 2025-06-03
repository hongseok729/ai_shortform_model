from config import *

def calculate_crop_region(tracking_data, shot_boundaries=None, frame_width=1920, frame_height=1080, 
                         target_ratio=9/16):
    """
    calculate_crop_region : 주인공의 얼굴을 Detect한 Bounding Box 정보를 기반으로 9:16 비율의 크롭 영역을 계산하는 함수
    
    Parameters:
        tracking_data : 주인공의 얼굴을 Detect한 Bounding Box 정보
                      [{'frame_idx': int, 'x': int, 'y': int, 'width': int, 'height': int}]
                      (XYWH 형태로 주어짐, CXCY 아님)
                      주의: 사람이 너무 작거나 등장하지 않아 식별 불가인 경우 Bounding Box 정보 없음
        shot_boundaries : 샷 경계를 나타내는 프레임 인덱스 리스트 (없으면 None)
        frame_width : 원본 영상의 가로 길이 (기본값: 1920)
        frame_height : 원본 영상의 세로 길이 (기본값: 1080)
        target_ratio : 숏폼으로 구성할 비디오의 가로 세로 비율 (기본값: 9/16)
        
    Returns:
        crop_regions : list of dict
                     각 프레임별 크롭 영역 정보를 담은 리스트
                     각 dict는 {'frame_idx': int, 'x': int, 'y': int, 'width': int, 'height': int} 형태
                     X, Y, W, H는 (XYWH) 형태임을 유의
    """
    
    # 1. 크롭 영역의 초기 크기 계산
    crop_height = frame_height # 세로 길이는 원본 유지
    crop_width = int(crop_height * target_ratio) # 9:16 비율에 맞는 너비 계산
    
    # shot_boundaries가 None이면 빈 리스트로 초기화
    if shot_boundaries is None:
        shot_boundaries = []
    
    # 2. 각 프레임별 크롭 영역 계산
    crop_regions = []
    previous_crop = None
    previous_face = None # 직전 프레임의 face bounding box 정보 저장
    
    for data in tracking_data:
        frame_idx = data['frame_idx']
        target_x = data['x']
        target_w = data['width']
        target_cx = target_x + target_w // 2 # 타겟 중심 X 좌표
        
        # 2-1. 누락된 프레임 처리 (이전 크롭 정보가 있는 경우)
        if previous_crop is not None:
            prev_idx = previous_crop['frame_idx']
            
            # 프레임이 연속적이지 않은 경우 (누락된 프레임 있음)
            if frame_idx > prev_idx + 1:
                for missing_idx in range(prev_idx + 1, frame_idx):
                    # 누락된 프레임에 대해 이전 크롭 정보 그대로 사용
                    crop_regions.append({
                        'frame_idx': missing_idx,
                        'x': previous_crop['x'],
                        'y': 0,
                        'width': crop_width,
                        'height': crop_height
                    })
        
        # 2-2. 현재 프레임의 크롭 영역 계산
        # 새로운 샷이 시작되는 경우
        if frame_idx in shot_boundaries:
            # 타겟 중심 기준으로 크롭 영역 설정
            ######################################## 이후 15개 frame의 target_cx 평균으로 설정
            target_x = 0
            inbox_count = 0
            for i in range(frame_idx, len(tracking_data)):
                if tracking_data[i]['x'] > 0:
                    target_x += tracking_data[i]['x']
                    inbox_count += 1
                if (inbox_count == 15):
                    break
            crop_x = target_x // 15
            crop_cx = crop_x + tracking_data[i]['width'] // 2
            print(f"샷 시작: {frame_idx}, 타겟 중심 X 좌표: {target_cx}")
            is_moving = True
            
        # 이전 크롭 정보가 있는 경우 (이어지는 샷)
        elif previous_crop is not None:
            prev_crop_cx = previous_crop['crop_cx']
            is_moving = previous_crop['is_moving']
            
            # 타겟과 이전 크롭 중심 간의 거리 (Crop Box와 현재 Face의 거리)
            crop_to_face_distance = target_cx - prev_crop_cx
            abs_crop_to_face_distance = abs(crop_to_face_distance)
            
            # 직전 프레임과 현재 프레임의 face bounding box 중심 간의 거리 계산
            face_movement = 0
            if previous_face is not None:
                prev_face_cx = previous_face['face_cx']
                face_movement = target_cx - prev_face_cx
                abs_face_movement = abs(face_movement)
            
            # 2-2-1. 이동 중인 경우
            if is_moving:
                # 급격한 이동(Jump) 감지
                if abs_crop_to_face_distance > JUMP_THRESHOLD:
                    # Jump 발생 시 크롭 위치 유지
                    crop_cx = prev_crop_cx
                    is_moving = True # 계속 이동 상태 유지
                else:
                    # 타겟 방향으로 부드럽게 이동
                    if crop_to_face_distance > 0: # 오른쪽으로 이동
                        move_amount = min(ANIMATION_SPEED, crop_to_face_distance)
                        crop_cx = prev_crop_cx + move_amount
                    elif crop_to_face_distance < 0: # 왼쪽으로 이동
                        move_amount = min(ANIMATION_SPEED, abs_crop_to_face_distance)
                        crop_cx = prev_crop_cx - move_amount
                    else: # 이동 없음
                        crop_cx = prev_crop_cx
                
                # 타겟에 도달했는지 확인
                if abs(crop_cx - target_cx) < CRITICAL_DISTANCE: # CRITICAL_DISTANCE보다 작으면 이동 끝?
                    is_moving = False
            
            # 2-2-2. 고정된 상태인 경우
            else:
                # 미세한 떨림 감지 (직전 프레임과 현재 프레임의 face 위치 비교)
                if previous_face is not None and abs_face_movement < VIBE_THRESHOLD:
                    crop_cx = prev_crop_cx # 위치 유지
                
                # 급격한 이동(Jump) 감지
                elif abs_crop_to_face_distance > JUMP_THRESHOLD:
                    crop_cx = prev_crop_cx # 위치 유지
                
                # 임계 거리 초과 시 이동 시작 (현재 crop box와 현재 face의 거리)
                elif abs_crop_to_face_distance > CRITICAL_DISTANCE:
                    is_moving = True
                    
                    # 초기 이동
                    if crop_to_face_distance > 0: # 오른쪽으로 이동
                        move_amount = min(ANIMATION_SPEED, crop_to_face_distance)
                        crop_cx = prev_crop_cx + move_amount
                    else: # 왼쪽으로 이동
                        move_amount = min(ANIMATION_SPEED, abs_crop_to_face_distance)
                        crop_cx = prev_crop_cx - move_amount
                else:
                    crop_cx = prev_crop_cx # 위치 유지
        
        # 이전 크롭 정보가 없는 경우 (첫 프레임 또는 예외 상황)
        else:
            crop_cx = target_cx
            is_moving = False
        
        # 2-3. 경계 조건 처리
        # 크롭 영역이 왼쪽 경계를 벗어나는 경우
        if crop_cx < crop_width // 2:
            crop_cx = crop_width // 2
            is_moving = False
        
        # 크롭 영역이 오른쪽 경계를 벗어나는 경우
        elif crop_cx > frame_width - crop_width // 2:
            crop_cx = frame_width - crop_width // 2
            is_moving = False
        
        # 2-4. 중심 좌표에서 왼쪽 상단 좌표로 변환
        crop_x = int(crop_cx - crop_width // 2)
        
        # 2-5. 최종 크롭 영역 정보 저장
        crop_region = {
            'frame_idx': frame_idx,
            'x': crop_x,
            'y': 0, # 세로는 항상 전체 높이 사용
            'width': crop_width,
            'height': crop_height
        }
        
        crop_regions.append(crop_region)
        
        # 2-6. 다음 프레임을 위해 현재 정보 저장
        previous_crop = {
            'frame_idx': frame_idx,
            'crop_cx': crop_cx,
            'x': crop_x,
            'is_moving': is_moving
        }
        
        # 현재 프레임의 face bounding box 정보 저장
        previous_face = {
            'frame_idx': frame_idx,
            'face_cx': target_cx
        }
    
    return crop_regions