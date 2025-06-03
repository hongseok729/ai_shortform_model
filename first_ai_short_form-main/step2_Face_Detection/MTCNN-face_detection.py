import cv2
import os
import json
import numpy as np
import time
import traceback
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Set
from mtcnn import MTCNN

# 상수 정의
DEFAULT_CONF_THRESHOLD = 0.96  # MTCNN 신뢰도 임계값
DEFAULT_EXPANSION_RATIO = 0.3  # 바운딩 박스 확장 비율

# 디렉토리가 존재하지 않을 경우 생성
def create_directory(directory_path: str) -> str:
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    return directory_path

# shot boundary 문자열 -> 초 변환
def time_to_seconds(time_str: str) -> float:
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)

def seconds_to_frame(seconds: float, fps: float) -> int:
    return int(seconds * fps)

# 비디오에서 모든 프레임 추출
def extract_all_frames(video_path: str, output_dir: str) -> int:
    """
    비디오의 모든 프레임을 추출하고 정확한 프레임 수를 반환
    Args:
        video_path: 비디오 파일 경로
        output_dir: 프레임을 저장할 디렉토리 경로
        
    Returns:
        추출된 프레임 수
    """
    create_directory(output_dir)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("비디오 파일을 열 수 없습니다.")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    reported_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"비디오 정보: {width}x{height}, {fps} FPS, 메타데이터 프레임 수: {reported_frames}")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 프레임 저장
        frame_path = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_count += 1
        
        # 진행 상황 표시
        if frame_count % 100 == 0:
            elapsed = time.time() - start_time
            estimated_total = (elapsed / frame_count) * reported_frames
            remaining = estimated_total - elapsed
            print(f"{frame_count}개 프레임 추출 완료... (예상 남은 시간: {remaining:.1f}초)")
    
    cap.release()
    
    end_time = time.time()
    print(f"총 {frame_count}개 프레임 추출 완료 (소요시간: {end_time - start_time:.2f}초)")
    
    # 메타데이터와 실제 프레임 수 비교
    if reported_frames != frame_count:
        print(f"주의: 메타데이터 프레임 수({reported_frames})와 실제 프레임 수({frame_count})가 일치하지 않습니다.")
        print(f"차이: {abs(reported_frames - frame_count)}개")
    
    # 비디오 정보 저장
    video_info = {
        "video_path": video_path,
        "fps": fps,
        "width": width,
        "height": height,
        "reported_frames": reported_frames,
        "actual_frames": frame_count,
        "extraction_time": end_time - start_time
    }
    
    with open(os.path.join(output_dir, "video_info.json"), 'w') as f:
        json.dump(video_info, f, indent=2)
    
    return frame_count

# 추출된 프레임에 MTCNN 모델을 적용하여 얼굴 감지 및 바운딩 박스 표시
def process_extracted_frames(frames_dir: str, output_dir: str, shot_boundaries_path: Optional[str] = None,
                           conf_threshold: float = DEFAULT_CONF_THRESHOLD,
                           expansion_ratio: float = DEFAULT_EXPANSION_RATIO) -> int:
    """
    추출된 프레임에 MTCNN 모델을 적용하여 얼굴 감지 및 바운딩 박스를 그립니다.
    
    Args:
        frames_dir: 추출된 프레임이 저장된 디렉토리 경로
        output_dir: 결과를 저장할 디렉토리 경로
        shot_boundaries_path: shot 경계 정보가 저장된 JSON 파일 경로 (옵션)
        conf_threshold: 신뢰도 임계값
        expansion_ratio: 바운딩 박스 확장 비율
        
    Returns:
        처리된 프레임 수
    """
    # 출력 디렉토리 생성
    img_dir = create_directory(os.path.join(output_dir, "img"))
    json_dir = create_directory(os.path.join(output_dir, "json"))
    
    # 비디오 정보 로드
    video_info_path = os.path.join(frames_dir, "video_info.json")
    if os.path.exists(video_info_path):
        with open(video_info_path, 'r') as f:
            video_info = json.load(f)
        print(f"비디오 정보 로드: {video_info['width']}x{video_info['height']}, {video_info['fps']} FPS")
    else:
        video_info = {"fps": 30.0}  # 기본값
        print("비디오 정보 파일을 찾을 수 없습니다. 기본값 사용.")
    
    # Shot 경계 정보 로드 (있는 경우)
    shot_info = {}
    if shot_boundaries_path and os.path.exists(shot_boundaries_path):
        with open(shot_boundaries_path, 'r') as f:
            shot_boundaries = json.load(f)
        
        # 프레임 번호별 shot 인덱스 매핑 생성
        for shot_idx, shot in enumerate(shot_boundaries):
            if "start_frame" in shot and "end_frame" in shot:
                for frame_num in range(shot["start_frame"], shot["end_frame"] + 1):
                    shot_info[frame_num] = shot_idx
            else:
                # 시간 기반으로 프레임 번호 계산
                start_seconds = time_to_seconds(shot['start'])
                end_seconds = time_to_seconds(shot['end'])
                start_frame = seconds_to_frame(start_seconds, video_info["fps"])
                end_frame = seconds_to_frame(end_seconds, video_info["fps"])
                for frame_num in range(start_frame, end_frame + 1):
                    shot_info[frame_num] = shot_idx
        
        print(f"Shot 경계 정보 로드 완료: {len(shot_boundaries)}개의 shot")
    
    # MTCNN 모델 로드
    print("MTCNN 모델 로드 중...")
    detector = MTCNN()
    print("MTCNN 모델 로드 완료")
    
    # 모든 프레임 파일 가져오기
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.startswith("frame_") and f.endswith(".jpg")])
    total_frames = len(frame_files)
    print(f"처리할 프레임 수: {total_frames}")
    
    # 모든 얼굴 데이터를 저장할 리스트
    all_faces_data = []
    
    # 각 shot별 얼굴 데이터
    shot_faces_data = {}
    
    processed_frames = 0
    start_time = time.time()
    
    for frame_file in frame_files:
        # 프레임 번호 추출
        frame_number = int(frame_file.split("_")[1].split(".")[0])
        
        # 해당 프레임이 속한 shot 찾기
        shot_idx = shot_info.get(frame_number, 0)  # 기본값은 0
        
        # 프레임 로드
        frame_path = os.path.join(frames_dir, frame_file)
        frame = cv2.imread(frame_path)
        
        if frame is None:
            print(f"프레임 {frame_number} 로드 실패, 건너뜁니다.")
            continue
        
        # RGB로 변환 (MTCNN은 RGB 형식 사용)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 얼굴 감지
        detections = detector.detect_faces(rgb_frame)
        
        # 바운딩 박스 및 JSON 데이터 생성
        faces = []
        frame_with_boxes = frame.copy()
        
        for detection in detections:
            confidence = detection["confidence"]
            
            # 신뢰도가 임계값보다 낮으면 건너뜀
            if confidence < conf_threshold:
                continue
                
            # MTCNN은 (x, y, width, height) 형식으로 반환
            x, y, width, height = detection["box"]
            x1, y1 = x, y
            x2, y2 = x + width, y + height
            
            # 바운딩 박스 확장
            expand_x = int(width * expansion_ratio)
            expand_y = int(height * expansion_ratio)
            
            new_x1 = max(0, x1 - expand_x)
            new_y1 = max(0, y1 - expand_y)
            new_x2 = min(frame.shape[1], x2 + expand_x)
            new_y2 = min(frame.shape[0], y2 + expand_y)
            
            expanded_bbox = [new_x1, new_y1, new_x2, new_y2, float(confidence)]
            faces.append(expanded_bbox)
            
            # 바운딩 박스 그리기
            cv2.rectangle(frame_with_boxes, (new_x1, new_y1), (new_x2, new_y2), (0, 255, 0), 2)
            
            # 신뢰도 점수 표시
            conf_text = f"{confidence:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame_with_boxes, (new_x1, new_y1 - 20), (new_x1 + text_width, new_y1), (0, 255, 0), -1)
            cv2.putText(frame_with_boxes, conf_text, (new_x1, new_y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
            # 얼굴 특징점(landmarks) 그리기 (선택 사항)
            # keypoints = detection["keypoints"]
            # for point in keypoints.values():
            #     cv2.circle(frame_with_boxes, point, 2, (0, 0, 255), 2)
        
        # 파일명 형식 설정
        file_prefix = f"shot_{shot_idx:04d}_frame_{frame_number:04d}"
        
        # 이미지 저장
        output_img_path = os.path.join(img_dir, f"{file_prefix}.jpg")
        cv2.imwrite(output_img_path, frame_with_boxes)
        
        # JSON 데이터 생성
        json_data = {
            "shot": shot_idx,
            "frame": frame_number,
            "faces": faces,
            "original_image_path": f"img/{file_prefix}.jpg"
        }
        
        # JSON 저장
        json_path = os.path.join(json_dir, f"{file_prefix}.json")
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        # 얼굴 데이터 수집
        all_faces_data.append(json_data)
        
        # Shot별 얼굴 데이터 수집
        if shot_idx not in shot_faces_data:
            shot_faces_data[shot_idx] = []
        shot_faces_data[shot_idx].append(json_data)
        
        processed_frames += 1
        
        # 진행 상황 표시
        if processed_frames % 10 == 0:
            elapsed = time.time() - start_time
            frames_per_second = processed_frames / elapsed if elapsed > 0 else 0
            estimated_total = (elapsed / processed_frames) * total_frames
            remaining = estimated_total - elapsed
            print(f"{processed_frames}/{total_frames} 프레임 처리 완료... "
                  f"({frames_per_second:.1f} FPS, 남은 시간: {remaining:.1f}초)")
    
    # 각 shot의 모든 얼굴 데이터를 하나의 JSON 파일로 저장
    for shot_idx, shot_data in shot_faces_data.items():
        shot_faces_json_path = os.path.join(json_dir, f"shot_{shot_idx:04d}_faces.json")
        with open(shot_faces_json_path, 'w') as f:
            json.dump(shot_data, f, indent=2)
    
    # 모든 얼굴 데이터를 JSON 파일로 저장
    faces_json_path = os.path.join(output_dir, "faces_detection.json")
    with open(faces_json_path, 'w') as f:
        json.dump(all_faces_data, f, indent=2)
    
    end_time = time.time()
    print(f"총 {processed_frames}개 프레임 처리 완료 (소요시간: {end_time - start_time:.2f}초)")
    
    # 결과 정보 저장
    result = {
        "frames_dir": frames_dir,
        "output_dir": output_dir,
        "shot_boundaries_path": shot_boundaries_path,
        "conf_threshold": conf_threshold,
        "expansion_ratio": expansion_ratio,
        "total_frames": total_frames,
        "processed_frames": processed_frames,
        "processing_time": end_time - start_time,
        "fps": processed_frames / (end_time - start_time) if (end_time - start_time) > 0 else 0
    }
    
    result_json_path = os.path.join(output_dir, "result.json")
    with open(result_json_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    return processed_frames

# 메인 함수
def face_detection(
    video_path: str, 
    output_dir: str, 
    shot_boundaries_path: Optional[str] = None,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    expansion_ratio: float = DEFAULT_EXPANSION_RATIO
) -> Dict[str, Any]:
    """
    비디오에서 얼굴을 감지하고 바운딩 박스를 그려서 저장 (img, json)
    
    Args:
        video_path: 비디오 파일 경로
        output_dir: 결과를 저장할 디렉토리 경로
        shot_boundaries_path: shot 경계 정보가 저장된 JSON 파일 경로 (옵션)
        conf_threshold: 신뢰도 임계값
        expansion_ratio: 바운딩 박스 확장 비율
        
    Returns:
        처리 결과 정보를 담은 딕셔너리
    """
    try:
        # 결과 디렉토리 생성
        create_directory(output_dir)
        frames_dir = os.path.join(output_dir, "frames")
        
        # 1단계: 모든 프레임 추출
        print("===== 1단계: 모든 프레임 추출 =====")
        frame_count = extract_all_frames(video_path, frames_dir)
        
        # 2단계: 추출된 프레임에 MTCNN 모델 적용
        print("\n===== 2단계: 얼굴 감지 및 바운딩 박스 그리기 =====")
        processed_frames = process_extracted_frames(
            frames_dir, 
            output_dir, 
            shot_boundaries_path,
            conf_threshold=conf_threshold,
            expansion_ratio=expansion_ratio
        )
        
        # 결과 확인
        print("\n===== 처리 결과 =====")
        print(f"추출된 프레임 수: {frame_count}")
        print(f"처리된 프레임 수: {processed_frames}")
        
        result = {
            "status": "success",
            "video_path": video_path,
            "output_dir": output_dir,
            "shot_boundaries_path": shot_boundaries_path,
            "frame_count": frame_count,
            "processed_frames": processed_frames,
            "conf_threshold": conf_threshold,
            "expansion_ratio": expansion_ratio
        }
        
        if frame_count != processed_frames:
            print(f"주의: 추출된 프레임 수({frame_count})와 처리된 프레임 수({processed_frames})가 일치하지 않습니다.")
            print(f"차이: {abs(frame_count - processed_frames)}개")
            result["warning"] = f"프레임 수 불일치: 차이 {abs(frame_count - processed_frames)}개"
        else:
            print("모든 프레임이 성공적으로 처리되었습니다.")
        
        return result
        
    except Exception as e:
        error_message = f"오류 발생: {e}"
        print(error_message)
        traceback.print_exc()
        return {
            "status": "error",
            "error_message": error_message,
            "video_path": video_path,
            "output_dir": output_dir
        }


def main():
    # 설정
    video_path = "../input_data/data_1.mp4"
    output_dir = "../mtcnn_output_data/1"
    shot_boundaries_path = "../output_data/output_shot_boundaries.json"
    
    # 파라미터
    conf_threshold = DEFAULT_CONF_THRESHOLD  # 0.96으로 변경 (default 0.9)
    expansion_ratio = DEFAULT_EXPANSION_RATIO
    
    # face_detection 함수 호출
    result = face_detection(
        video_path=video_path,
        output_dir=output_dir,
        shot_boundaries_path=shot_boundaries_path,
        conf_threshold=conf_threshold,
        expansion_ratio=expansion_ratio
    )
    
    # 결과 출력
    if result["status"] == "success":
        print("\n처리 완료. 결과:")
        print(json.dumps(result, indent=2))
    else:
        print("\n처리 실패. 오류:")
        print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
