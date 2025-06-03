import json

def preprocess_tracking_data(json_file_path):
    """
    JSON 파일에서 트래킹 데이터를 읽어와 계산에 필요한 형식으로 변환

    Parameters:
        json_file_path : str
            트래킹 데이터가 저장된 JSON 파일 경로

    Returns:
        tracking_data : list of dict
            변환된 트래킹 데이터
    """
    # JSON 파일 읽기
    with open(json_file_path, 'r') as f:
        raw_data = json.load(f)

    tracking_data = []

    # 데이터 형식 변환
    for item in raw_data:
        if 'frame' in item and 'main' in item:
            frame_idx = item['frame']
            bbox = item['main']
            
            # bbox는 [x1, y1, x2, y2] 형태
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox

                # XYWH 형태로 변환
                x = int(x1)
                y = int(y1)
                width = int(x2 - x1)
                height = int(y2 - y1)
                
                x = max(0, x)
                x = min(x, 1920 - width) # 1920은 원본 영상의 가로 길이
                y = max(0, y)
                y = min(y, 1080 - height) # 1080은 원본 영상의 세로 길이

                tracking_data.append({
                    'frame_idx': frame_idx,
                    'x': x,
                    'y': y,
                    'width': width,
                    'height': height
                })
            
            # 확률이 포함될 경우? [x1, y1, x2, y2, c]
            elif len(bbox) == 5:
                x1, y1, x2, y2, c = bbox # c는 안쓰니 버립시다.
                
                # XYWH 형태로 변환
                x = int(x1)
                y = int(y1)
                width = int(x2 - x1)
                height = int(y2 - y1)
                
                tracking_data.append({
                    'frame_idx': frame_idx,
                    'x': x,
                    'y': y,
                    'width': width,
                    'height': height
                })

    # 프레임 인덱스 기준으로 정렬
    tracking_data.sort(key=lambda x: x['frame_idx'])
    
    return tracking_data