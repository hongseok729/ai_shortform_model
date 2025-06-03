import cv2

# 프레임 번호를 표시하는 함수
def draw_frame_number(frame, frame_idx): # 영상 Crop 중 화면 Crop Box 및 Face Bounding Box 확인을 위한 함수
    if frame is None or frame.size == 0:
        return frame
        
    text = ''
    # text = f"Frame: {frame_idx}" # Frame 번호 없는 실제 크롭 영상 출력시 이 문장 주석처리할것!!!!! ##################
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 2
    color = (0, 255, 255) # 노란색
    
    # 텍스트 크기 계산
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    
    # 가운데 상단 좌표 계산
    x = (frame.shape[1] - text_width) // 2
    y = text_height + 10 # 상단에서 약간 아래
    
    # 텍스트 테두리(검정색) 먼저 그림
    cv2.putText(frame, text, (x, y), font, font_scale, (0,0,0), thickness+2, cv2.LINE_AA)
    
    # 텍스트 본문(노란색)
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    
    return frame

# 크롭 영역을 Bounding Box로 표시하는 함수
def draw_crop_box(frame, x, y, width, height):
    if frame is None or frame.size == 0:
        return frame
        
    # 초록색 Bounding Box 그리기
    color = (0, 255, 0) # 초록색
    thickness = 3
    cv2.rectangle(frame, (x, y), (x + width, y + height), color, thickness)
    
    # 크롭 영역 정보 표시
    text = f"Crop: ({x}, {y}, {width}, {height})"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    text_thickness = 2
    
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, text_thickness)
    
    # 텍스트 위치 (Bounding Box 아래)
    text_x = x
    text_y = y + height + text_height + 10
    
    # 텍스트가 프레임 밖으로 나가지 않도록 조정
    if text_y >= frame.shape[0]:
        text_y = y - 10 # Bounding Box 위에 표시
    
    # 텍스트 배경 (가독성 향상)
    cv2.rectangle(frame, (text_x, text_y - text_height),
                    (text_x + text_width, text_y + 5), (0, 0, 0), -1)
    
    # 텍스트 그리기
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), text_thickness, cv2.LINE_AA)
    
    return frame

# 얼굴 영역을 Bounding Box로 표시하는 함수
def draw_face_box(frame, x, y, width, height):
    if frame is None or frame.size == 0:
        return frame
        
    # 파란색 Bounding Box 그리기
    color = (255, 0, 0) # 파란색
    thickness = 2
    cv2.rectangle(frame, (x, y), (x + width, y + height), color, thickness)
    
    # 얼굴 영역 정보 표시
    text = f"Face: ({x}, {y}, {width}, {height})"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    text_thickness = 1
    
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, text_thickness)
    
    # 텍스트 위치 (Bounding Box 위)
    text_x = x
    text_y = y - 10
    
    # 텍스트가 프레임 밖으로 나가지 않도록 조정
    if text_y < text_height:
        text_y = y + height + text_height + 5 # Bounding Box 아래에 표시
    
    # 텍스트 배경 (가독성 향상)
    cv2.rectangle(frame, (text_x, text_y - text_height),
                    (text_x + text_width, text_y + 5), (0, 0, 0), -1)
    
    # 텍스트 그리기
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), text_thickness, cv2.LINE_AA)
    
    return frame