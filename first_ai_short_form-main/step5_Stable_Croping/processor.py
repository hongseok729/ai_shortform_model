from draw_utils import *
from video_utils import *
import os

def process_frames(all_frames, crop_regions, tracking_dict, output_path, fps, output_size, draw_mode=1):
    """
    draw_mode:
        1 = 크롭 + 프레임 번호 표시
        2 = 원본 + 크롭 & 얼굴 박스 표시
    """
    width, height = output_size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, output_size)

    crop_region_dict = {region['frame_idx']: region for region in crop_regions}
    processed_frames = 0

    for frame_idx, frame in all_frames:
        try:
            crop_info = crop_region_dict.get(frame_idx, None)

            if draw_mode == 1:
                # Mode 1: 실제 크롭 수행
                if crop_info:
                    x, y = crop_info['x'], crop_info['y']
                    w, h = crop_info['width'], crop_info['height']
                    if w % 2 != 0: w += 1

                    cropped = safe_crop(frame, x, y, w, h, width, height)
                else:
                    center_x = frame.shape[1] // 2
                    x = max(0, min(center_x - width // 2, frame.shape[1] - width))
                    cropped = safe_crop(frame, x, 0, width, height, width, height)

                result_frame = draw_frame_number(cropped, frame_idx)

            else:
                # Mode 2: 원본 위에 박스만 그림
                result_frame = frame.copy()
                if crop_info:
                    x, y = crop_info['x'], crop_info['y']
                    w, h = crop_info['width'], crop_info['height']
                    result_frame = draw_crop_box(result_frame, x, y, w, h)

                if frame_idx in tracking_dict:
                    f = tracking_dict[frame_idx]
                    result_frame = draw_face_box(result_frame, f['x'], f['y'], f['width'], f['height'])

                result_frame = draw_frame_number(result_frame, frame_idx)

            out.write(result_frame)
            processed_frames += 1

        except Exception as e:
            print(f"[ERROR] frame {frame_idx}: {e}")
            fallback = np.zeros((height, width, 3), dtype=np.uint8)
            fallback = draw_frame_number(fallback, frame_idx)
            out.write(fallback)
            processed_frames += 1

        if frame_idx % 100 == 0 or frame_idx == len(all_frames) - 1:
            print(f"처리 중: {frame_idx+1}/{len(all_frames)} 프레임")

    out.release()
    return processed_frames

# FFmpeg를 추가하여 오디오를 포함한 비디오 생성
def apply_crop_to_video(input_video_path, output_video_path, crop_regions, tracking_data):
    """
    계산된 Crop Box을 적용하여 영상을 세로로 변환하거나
    원본 영상에 Crop Box를 Bounding Box로 표시
    
    Parameters:
    -----------
    input_video_path : str
        입력 영상 파일 경로
    output_video_path : str
        출력 영상 파일 경로
    crop_regions : list of dict
        각 프레임별 크롭 영역 정보
    tracking_data : list of dict
        각 프레임별 얼굴 트래킹 정보
    """
    
    # 임시 디렉토리를 스크립트가 있는 위치에 생성 (C 드라이브 강제 사용 방지)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    temp_dir = os.path.join(script_dir, "temp_video_processing")
    os.makedirs(temp_dir, exist_ok=True)
    
    # 비디오 속성 가져오기
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    temp_cap = cv2.VideoCapture(input_video_path)
    while temp_cap.read()[0]:
        frame_count += 1
    temp_cap.release()
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()  # 자원 해제
    
    # 첫 번째 크롭 영역 정보로 출력 비디오 크기 설정
    output_width = crop_regions[0]['width']
    output_height = crop_regions[0]['height']
    
    # 홀수 너비 처리 (H.264는 짝수 너비 필요)
    if output_width % 2 != 0:
        output_width += 1
    
    print(f"원본 비디오 크기: {width}x{height}, FPS: {fps}")
    print(f"크롭 영역 크기: {output_width}x{output_height}")
    print(f"총 프레임 수: {frame_count}")
    
    # 임시 비디오 파일 경로 (오디오 없는 버전)
    temp_output_path = os.path.join(temp_dir, os.path.basename(output_video_path).replace('.mp4', '_temp.mp4'))
    
    # tracking_data를 프레임 인덱스로 빠르게 접근할 수 있도록 딕셔너리로 변환
    tracking_dict = {item['frame_idx']: item for item in tracking_data}
    
    # 모드 선택 (주석 처리를 통해 모드 변경) ################################################################################
    mode = 1  # 모드 1: 영상을 Crop하는 기능 
    # mode = 2  # 모드 2: Crop할 Box를 나타내는 원본 영상 만들기 #### 개발 중 확인 용도 ##################################
    
    # 모든 프레임을 한 번에 메모리에 로드 (프레임 손실 방지)
    print("모든 프레임 로드 중...")
    all_frames = []
    cap = cv2.VideoCapture(input_video_path)
    
    # 버퍼 크기 최소화 (프레임 손실 방지)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # 모든 프레임 읽기
    for i in range(frame_count):
        ret, frame = cap.read()
        if ret:
            all_frames.append((i, frame))
        else:
            print(f"프레임 {i} 읽기 실패 - 이전 프레임 복제")
            if len(all_frames) > 0:
                # 이전 프레임 복제
                all_frames.append((i, all_frames[-1][1].copy()))
            else:
                # 첫 프레임부터 실패한 경우 검은색 프레임 생성
                all_frames.append((i, np.zeros((height, width, 3), dtype=np.uint8)))
    
    cap.release()
    print(f"총 {len(all_frames)}개 프레임 로드 완료")
    
    if mode == 1:
        output_size = (output_width, output_height)
        processed_frames = process_frames(
            all_frames, crop_regions, tracking_dict,
            temp_output_path, fps, output_size, draw_mode=1
        )
    else:
        output_size = (width, height)
        processed_frames = process_frames(
            all_frames, crop_regions, tracking_dict,
            temp_output_path, fps, output_size, draw_mode=2
        )

    # 결과 확인
    if os.path.exists(temp_output_path):
        file_size = os.path.getsize(temp_output_path)
        print(f"임시 출력 파일 생성 완료: {temp_output_path} (크기: {file_size} 바이트)")
    else:
        print(f"오류: 임시 출력 파일이 생성되지 않았습니다: {temp_output_path}")
    
    print(f"비디오 처리 완료: {processed_frames}/{frame_count} 프레임 처리됨")
    
    # ===== 여기서부터 오디오 추가 코드 =====
    try:
        # FFmpeg를 사용하여 원본 비디오의 오디오를 추출하고 새 비디오에 추가
        import subprocess
        
        # FFmpeg 명령어 구성: 비디오와 오디오 결합
        cmd = [
            'ffmpeg',
            '-i', temp_output_path,  # 비디오 입력 (오디오 없음)
            '-i', input_video_path,  # 오디오 소스
            '-c:v', 'copy',          # 비디오 코덱 복사
            '-c:a', 'aac',           # 오디오 코덱 AAC
            '-map', '0:v:0',         # 첫 번째 입력에서 비디오 스트림 가져오기
            '-map', '1:a:0',         # 두 번째 입력에서 오디오 스트림 가져오기
            '-shortest',             # 가장 짧은 스트림 길이에 맞추기
            '-vsync', 'passthrough', # 프레임 타임스탬프 유지
            '-fps_mode', 'passthrough', # 프레임 레이트 유지
            output_video_path        # 출력 파일
        ]
        
        # FFmpeg 실행
        print("오디오 추가 중...")
        subprocess.run(cmd, check=True)
        
        print(f"오디오가 추가된 최종 출력 파일 생성 완료: {output_video_path}")
        
        # 임시 파일 삭제
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
            print(f"임시 파일 삭제 완료: {temp_output_path}")
    
    except Exception as e:
        print(f"오디오 추가 중 오류 발생: {e}")
        print("오디오 없이 비디오만 생성됩니다.")
        
        # 오류 발생 시 임시 파일을 최종 출력으로 이동
        if os.path.exists(temp_output_path):
            import shutil
            shutil.copy(temp_output_path, output_video_path)
            print(f"임시 파일을 최종 출력으로 복사: {output_video_path}")
    
    # 임시 디렉토리 정리
    try:
        if os.path.exists(temp_dir) and len(os.listdir(temp_dir)) == 0:
            os.rmdir(temp_dir)
            print(f"임시 디렉토리 삭제 완료: {temp_dir}")
    except Exception as e:
        print(f"임시 디렉토리 정리 중 오류 발생: {e}")
    
    # 결과 확인
    if os.path.exists(output_video_path):
        file_size = os.path.getsize(output_video_path)
        print(f"최종 출력 파일 생성 완료: {output_video_path} (크기: {file_size} 바이트)")
    else:
        print(f"오류: 최종 출력 파일이 생성되지 않았습니다: {output_video_path}")

