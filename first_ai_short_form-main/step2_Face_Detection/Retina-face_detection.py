import cv2
import os
import json
import numpy as np
import time
import traceback
from retinaface import RetinaFace

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

# 상수
DEFAULT_CONF_THRESHOLD = 0.8
DEFAULT_EXPANSION_RATIO = 0.3

# 디렉토리 생성
def create_output_dir(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    return directory_path

# 모든 프레임 저장 (frame/frame_0000.jpg ...)
def save_all_frames(video_path, frame_dir):
    cap = cv2.VideoCapture(video_path)
    idx = 0
    create_output_dir(frame_dir)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(frame_dir, f"frame_{idx:04d}.jpg"), frame)
        idx += 1
    cap.release()
    return idx

# shot boundary json 로드
def load_shot_boundaries(shot_boundaries_path):
    with open(shot_boundaries_path, 'r') as f:
        shot_boundaries = json.load(f)
    return shot_boundaries

# RetinaFace 얼굴 탐지
def detect_faces_retina(model, frame, conf_threshold=DEFAULT_CONF_THRESHOLD):
    # model은 RetinaFace에서 필요 없음 (호환성 유지용)
    faces = RetinaFace.detect_faces(frame)
    results = []
    if faces:
        for face_idx, face_data in faces.items():
            confidence = face_data.get("score", 0)
            if confidence < conf_threshold:
                continue
            x1, y1, x2, y2 = face_data["facial_area"]
            results.append([x1, y1, x2, y2, confidence])
    return results

# 바운딩 박스 확장
def expand_bounding_box(bbox, expansion_ratio=0.2, frame_shape=None):
    x1, y1, x2, y2, confidence = bbox
    width = x2 - x1
    height = y2 - y1
    expand_x = int(width * expansion_ratio)
    expand_y = int(height * expansion_ratio)
    new_x1 = x1 - expand_x
    new_y1 = y1 - expand_y
    new_x2 = x2 + expand_x
    new_y2 = y2 + expand_y
    if frame_shape is not None:
        h, w = frame_shape[:2]
        new_x1 = max(0, new_x1)
        new_y1 = max(0, new_y1)
        new_x2 = min(w, new_x2)
        new_y2 = min(h, new_y2)
    return [new_x1, new_y1, new_x2, new_y2, confidence]

# 바운딩 박스 그리기
def draw_bounding_box(frame, bbox, color=(0, 255, 0), thickness=2):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    return frame

# JSON 저장
def save_bounding_boxes_json(shot_idx, frame_number, bboxes, json_dir):
    json_data = {"shot": shot_idx, "frame": frame_number, "faces": bboxes}
    json_path = os.path.join(json_dir, f"shot_{shot_idx:04d}_frame_{frame_number:04d}.json")
    with open(json_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=2, cls=NpEncoder)
    return json_path

# 메인 얼굴 탐지 함수
def face_detection(input_video_path, shot_boundaries_path, output_directory,
                   conf_threshold=DEFAULT_CONF_THRESHOLD, expansion_ratio=DEFAULT_EXPANSION_RATIO):
    # 폴더 구조 생성
    base_output_dir = create_output_dir(output_directory)
    frame_dir = create_output_dir(os.path.join(base_output_dir, "frame"))
    img_dir = create_output_dir(os.path.join(base_output_dir, "img"))
    json_dir = create_output_dir(os.path.join(base_output_dir, "json"))

    # 1. 모든 프레임 저장
    print("모든 프레임을 frame 폴더에 저장합니다...")
    total_frames = save_all_frames(input_video_path, frame_dir)
    print(f"총 {total_frames} 프레임 저장 완료")

    # 2. shot boundary 로드
    shot_boundaries = load_shot_boundaries(shot_boundaries_path)
    print(f"Shot boundary {len(shot_boundaries)}개 로드 완료")

    all_faces_data = []

    # 3. shot별로 반복
    for shot_idx, shot in enumerate(shot_boundaries):
        start_frame = shot["start_frame"]
        end_frame = shot["end_frame"]
        print(f"Shot {shot_idx}: 프레임 {start_frame}~{end_frame} 처리 중...")

        for frame_number in range(start_frame, end_frame + 1):
            frame_path = os.path.join(frame_dir, f"frame_{frame_number:04d}.jpg")
            frame = cv2.imread(frame_path)
            if frame is None:
                print(f"프레임 {frame_number} 로드 실패, 건너뜀")
                continue

            # 얼굴 탐지
            bboxes = detect_faces_retina(None, frame, conf_threshold)
            expanded_bboxes = [expand_bounding_box(b, expansion_ratio, frame.shape) for b in bboxes]

            # 바운딩 박스 그리기
            frame_with_boxes = frame.copy()
            for bbox in expanded_bboxes:
                frame_with_boxes = draw_bounding_box(frame_with_boxes, bbox[:4], color=(0,255,0))

            # img 저장 (shot_0000_frame_0000.jpg)
            img_filename = f"shot_{shot_idx:04d}_frame_{frame_number:04d}.jpg"
            cv2.imwrite(os.path.join(img_dir, img_filename), frame_with_boxes)

            # json 저장 (shot_0000_frame_0000.json)
            save_bounding_boxes_json(shot_idx, frame_number, expanded_bboxes, json_dir)

            # 전체 얼굴 데이터 저장
            all_faces_data.append({
                "shot": shot_idx,
                "frame": frame_number,
                "faces": expanded_bboxes
            })

            # 진행상황 출력
            if frame_number % 20 == 0:
                print(f"  프레임 {frame_number} 처리 완료")

    # 전체 얼굴 데이터 JSON 저장
    faces_json_path = os.path.join(base_output_dir, "faces_detection.json")
    with open(faces_json_path, 'w') as f:
        json.dump(all_faces_data, f, indent=2, cls=NpEncoder)

    print(f"얼굴탐지 완료! 총 {len(all_faces_data)} 프레임 처리됨.")

    # 결과 반환
    result = {
        "input_video": input_video_path,
        "shot_boundaries": shot_boundaries_path,
        "faces_detection": faces_json_path,
        "output_directory": base_output_dir
    }
    result_json_path = os.path.join(base_output_dir, "result.json")
    with open(result_json_path, 'w') as f:
        json.dump(result, f, indent=2, cls=NpEncoder)
    return result

def main():
    input_video_path = "../input_data/data_2.mp4"
    shot_boundaries_path = "../input_data/shot_boundary_2.json"
    output_directory = "../retina_output_data/2"
    conf_threshold = 0.8
    expansion_ratio = 0.3

    try:
        result = face_detection(
            input_video_path,
            shot_boundaries_path,
            output_directory,
            conf_threshold=conf_threshold,
            expansion_ratio=expansion_ratio
        )
        print("처리 완료.\n결과 요약:")
        print(json.dumps(result, indent=2, ensure_ascii=False, cls=NpEncoder))
    except Exception as e:
        print(f"비디오 처리 오류: {e}")

if __name__ == "__main__":
    main()
