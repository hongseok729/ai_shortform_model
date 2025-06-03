import numpy as np
import cv2

# 안전한 크롭 함수 추가
def safe_crop(frame, x, y, width, height, target_width, target_height):
    """
    안전하게 이미지를 크롭하는 함수
    경계를 벗어나는 경우 적절히 조정하고, 오류 처리를 수행
    """
    if frame is None or frame.size == 0:
        return None
        
    # 프레임 크기 확인
    frame_height, frame_width = frame.shape[:2]
    
    # 경계 조건 검사 및 조정
    x = max(0, min(x, frame_width - 1))
    y = max(0, min(y, frame_height - 1))
    
    # 크롭 영역의 너비와 높이가 유효한지 확인
    width = max(1, min(width, frame_width - x))
    height = max(1, min(height, frame_height - y))
    
    try:
        # 안전하게 크롭 수행
        cropped = frame[y:y+height, x:x+width].copy()
        
        # 크기 조정이 필요한 경우
        if cropped.shape[0] != target_height or cropped.shape[1] != target_width:
            cropped = cv2.resize(cropped, (target_width, target_height))
            
        return cropped
    except Exception as e:
        print(f"크롭 중 오류 발생: {e}")
        # 오류 발생 시 중앙 크롭으로 대체
        center_x = frame_width // 2
        center_y = frame_height // 2
        
        half_width = min(target_width // 2, center_x)
        half_height = min(target_height // 2, center_y)
        
        try:
            # 중앙 영역 크롭 시도
            x1 = max(0, center_x - half_width)
            y1 = max(0, center_y - half_height)
            x2 = min(frame_width, center_x + half_width)
            y2 = min(frame_height, center_y + half_height)
            
            cropped = frame[y1:y2, x1:x2].copy()
            
            # 크기 조정
            cropped = cv2.resize(cropped, (target_width, target_height))
            return cropped
        except Exception as e2:
            print(f"중앙 크롭 중 오류 발생: {e2}")
            # 최후의 방법: 검은색 이미지 생성
            return np.zeros((target_height, target_width, 3), dtype=np.uint8)