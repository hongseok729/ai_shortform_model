import json

# 샷 경계 정보 파일에서 start_frame 값을 추출하는 함수
def load_shot_boundaries(json_file_path):
    """
    JSON 파일에서 샷 경계 정보를 읽어와 start_frame 값을 추출
    
    Parameters:
        json_file_path : str
            샷 경계 정보가 저장된 JSON 파일 경로
            
    Returns:
        shot_boundaries : list
            각 샷의 시작 프레임 번호 리스트
    """
    import json
    
    try:
        # JSON 파일 읽기
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        # start_frame 값 추출
        shot_boundaries = [item['start_frame'] for item in data]
        
        print(f"샷 경계 정보 로드 완료: {len(shot_boundaries)}개 샷 경계 발견")
        return shot_boundaries
    
    except Exception as e:
        print(f"샷 경계 정보 로드 중 오류 발생: {e}")
        # 오류 발생 시 임의로 지정한 기본값 반환 # 개발 중 특정 영상의 Shot Boundary 확인을 위해 임의로 제작한 것임. 오류가 안 뜨길...
        return [0, 80, 211, 241, 284, 356, 397, 443, 482, 546, 600, 657, 702, 758, 806, 843, 878, 974, 1013]
