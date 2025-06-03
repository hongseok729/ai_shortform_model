from tracking import preprocess_tracking_data
from cropping import calculate_crop_region
from processor import apply_crop_to_video
from shot_boundary import load_shot_boundaries
from config import *
from datetime import datetime
import os

def main():
    # JSON 파일 경로
    # check_json.py로 보간한 file의 data와 영상의 frame 수가 일치하는지 확인 권장
    # 반드시 change_json.py 먼저 실행하여 소실 frame 없도록 보간 후 file을 불러올 것!!!
    ### 반드시 파일 실행 전 processor.py의 121, 122행 draw_mode 확인할 것. (mode=1 이 실제 최종 결과 / mode=2는 개발 중 확인용)
    ### 실제 결과물로 출력하기 전 draw_utils.py의 10행을 주석처리하여 확인용 Frame 번호가 적용되지 않도록 조치할것!!!
    json_file_path = 'data2.json'
    
    # 비디오 파일 경로
    input_video_path = 'data_2.mp4'
    now_str = datetime.now().strftime("%H%M%S")
    base_name = os.path.splitext(os.path.basename(input_video_path))[0]
    output_video_path = f'{base_name}_{now_str}.mp4' # {원본파일명}_{hhmmss}.mp4
    
    # 샷 경계 정보
    # 샷 경계 정보 파일에서 start_frame 값 추출
    shot_boundary_path = 'shot_boundary_2.json'
    shot_boundaries = load_shot_boundaries(shot_boundary_path)
    
    # 1. JSON 파일에서 트래킹 데이터 전처리
    tracking_data = preprocess_tracking_data(json_file_path)
    
    # 2. 크롭 영역 계산
    crop_regions = calculate_crop_region(
        tracking_data=tracking_data,
        shot_boundaries=shot_boundaries,
        frame_width=1920,
        frame_height=1080,
        target_ratio=9/16
    )
    
    # 3. 크롭 적용하여 비디오 생성 (tracking_data도 함께 전달)
    apply_crop_to_video(input_video_path, output_video_path, crop_regions, tracking_data)
    
    print(f"처리 완료: {output_video_path}")

if __name__ == "__main__":
    main()
