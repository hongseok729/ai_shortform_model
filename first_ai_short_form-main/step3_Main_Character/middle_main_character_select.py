import os
import json
import cv2
import numpy as np

def select_main_character(input_directory, output_directory, shot_boundaries_path=None, weights=None):
    """
    각 shot에서 주인공을 선별하는 함수
    Args:
        input_directory: step2의 결과가 저장된 디렉토리 경로
        output_directory: 주인공 선별 결과를 저장할 디렉토리 경로
        shot_boundaries_path: shot 경계 정보가 저장된 JSON 파일 경로 (옵션)
        weights: 주인공 선별 시 사용할 가중치 딕셔너리 (기본값: None)
    Returns:
        주인공 선별 결과 정보
    """
    # 출력 디렉토리 생성
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 기본 가중치 설정
    default_weights = {
        "quality": 0.4,    # 얼굴 품질(선명도+확률) 가중치
        "size": 0.2,       # 얼굴 크기 가중치
        "center": 0.4      # 중앙 위치 가중치
    }

    # 사용자 지정 가중치가 있으면 사용
    if weights:
        for key in default_weights:
            if key in weights:
                default_weights[key] = weights[key]

    # 가중치 정규화
    total_weight = sum(default_weights.values())
    if total_weight > 0:
        for key in default_weights:
            default_weights[key] /= total_weight

    print(f"사용 가중치: {default_weights}")

    # shot boundary 정보 로드
    if shot_boundaries_path:
        with open(shot_boundaries_path, 'r') as f:
            shot_boundaries = json.load(f)
        num_shots = len(shot_boundaries)
        print(f"shot boundary {num_shots}개 로드 완료")
    else:
        raise ValueError("shot_boundaries_path가 반드시 필요합니다.")

    # 모든 shot의 주인공 정보를 저장할 리스트
    all_main_characters = []

    # json, img 디렉토리 경로
    json_dir = os.path.join(input_directory, "json")
    img_dir = os.path.join(input_directory, "img")

    # 각 shot별로 처리
    for shot_idx, shot in enumerate(shot_boundaries):
        print(f"Shot {shot_idx} 주인공 선별 중...")

        # shot의 모든 프레임 얼굴 정보 로드 (json/shot_0000_faces.json)
        shot_faces_path = os.path.join(json_dir, f"shot_{shot_idx:04d}_faces.json")
        if not os.path.exists(shot_faces_path):
            print(f"Warning: {shot_faces_path} 파일이 존재하지 않습니다. 건너뜀")
            continue

        with open(shot_faces_path, 'r') as f:
            shot_faces = json.load(f)

        if not shot_faces:
            print(f"Shot {shot_idx}에서 얼굴 데이터를 찾을 수 없습니다.")
            continue

        # 주인공 선별
        main_character = select_main_in_shot(shot_faces, shot_idx, img_dir, json_dir, default_weights)

        if main_character:
            # 후처리를 통한 일관성 유지
            main_character = post_process_main_characters(main_character)
            all_main_characters.extend(main_character)

            # 주인공 얼굴에 바운딩 박스 그리기 및 저장
            create_main_character_images(main_character, img_dir, output_directory)

    # 모든 주인공 정보를 JSON 파일로 저장
    main_json_path = os.path.join(output_directory, "main_characters.json")
    with open(main_json_path, 'w') as f:
        json.dump(all_main_characters, f, indent=2)

    # 결과 정보 생성
    result = {
        "shot_boundaries_path": shot_boundaries_path,
        "main_characters": main_json_path,
        "output_directory": output_directory,
        "weights": default_weights
    }

    # 결과 정보 저장
    result_path = os.path.join(output_directory, "result.json")
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)

    return result

def select_main_in_shot(shot_faces, shot_idx, img_dir, json_dir, weights):
    """
    한 shot 내에서 주인공을 선별하는 함수
    Args:
        shot_faces: shot의 모든 프레임 얼굴 정보
        shot_idx: shot 인덱스
        img_dir: 이미지 디렉토리 경로
        json_dir: json 디렉토리 경로
        weights: 가중치 딕셔너리
    Returns:
        주인공 정보 리스트
    """
    # 프레임 크기 자동 감지
    frame_width, frame_height = detect_frame_size(shot_faces, img_dir, shot_idx)
    print(f"감지된 프레임 크기: {frame_width}x{frame_height}")

    # 얼굴 ID를 부여하기 위한 클러스터링
    face_clusters = cluster_faces(shot_faces)

    # 각 얼굴 클러스터별 점수 계산
    face_scores = calculate_face_scores(face_clusters, shot_faces, img_dir, shot_idx, weights, frame_width, frame_height)

    # 점수가 가장 높은 얼굴 클러스터를 주인공으로 선정
    if not face_scores:
        print(f"Shot {shot_idx}에서 얼굴이 감지되지 않았습니다.")
        return []

    main_cluster_id = max(face_scores, key=face_scores.get)
    print(f"Shot {shot_idx}의 주인공: 클러스터 ID {main_cluster_id}, 점수 {face_scores[main_cluster_id]:.2f}")

    # 주인공 정보 생성
    main_characters = []
    for frame_data in shot_faces:
        frame_num = frame_data["frame"]
        faces = frame_data["faces"]

        # 해당 프레임에서 주인공 얼굴 찾기
        main_face = find_main_face_in_frame(faces, main_cluster_id, face_clusters)

        if main_face:
            main_characters.append({
                "shot": shot_idx,
                "frame": frame_num,
                "main": main_face
            })

    return main_characters

def detect_frame_size(shot_faces, img_dir, shot_idx):
    """
    프레임 크기 자동 감지
    """
    default_width, default_height = 1920, 1080
    for frame_data in shot_faces:
        frame_num = frame_data["frame"]
        img_path = os.path.join(img_dir, f"shot_{shot_idx:04d}_frame_{frame_num:04d}.jpg")
        if os.path.exists(img_path):
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    height, width = img.shape[:2]
                    return width, height
            except Exception as e:
                print(f"이미지 로드 오류 ({img_path}): {e}")
    return default_width, default_height

def cluster_faces(shot_faces, iou_threshold=0.3):
    all_faces = []
    for frame_data in shot_faces:
        frame_num = frame_data["frame"]
        for face_idx, face in enumerate(frame_data["faces"]):
            all_faces.append({
                "frame": frame_num,
                "face_idx": face_idx,
                "bbox": face[:4],  # [x1, y1, x2, y2]
                "conf": face[4] if len(face) > 4 else 0.5
            })
    if not all_faces:
        return {}

    face_clusters = {}
    cluster_id = 0
    face_clusters[cluster_id] = [all_faces[0]]
    for face in all_faces[1:]:
        assigned = False
        for cid, cluster_faces in face_clusters.items():
            if is_same_face(face, cluster_faces[0], iou_threshold):
                face_clusters[cid].append(face)
                assigned = True
                break
        if not assigned:
            cluster_id += 1
            face_clusters[cluster_id] = [face]
    print(f"클러스터링 결과: {len(face_clusters)}개의 얼굴 클러스터 생성")
    return face_clusters

def is_same_face(face1, face2, iou_threshold=0.3):
    bbox1 = face1["bbox"]
    bbox2 = face2["bbox"]
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    if x2 < x1 or y2 < y1:
        return False
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection
    iou = intersection / union if union > 0 else 0
    return iou >= iou_threshold

def calculate_face_quality(face_img, conf):
    if face_img is None or face_img.size == 0:
        return 0
    min_size = 20
    h, w = face_img.shape[:2]
    if h < min_size or w < min_size:
        scale = max(min_size / h, min_size / w)
        face_img = cv2.resize(face_img, (int(w * scale), int(h * scale)))
    if len(face_img.shape) == 3:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_img
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = np.mean(np.sqrt(sobelx**2 + sobely**2))
    sharpness = (lap_var / 1000.0 + tenengrad / 100.0) / 2.0
    sharpness = min(1.0, max(0.0, sharpness))
    quality = sharpness * 0.7 + conf * 0.3
    return min(1.0, max(0.0, quality))

def calculate_face_scores(face_clusters, shot_faces, img_dir, shot_idx, weights, frame_width, frame_height):
    scores = {}
    for cluster_id, cluster_faces in face_clusters.items():
        # 1. 얼굴 크기 점수
        sizes = []
        for face in cluster_faces:
            bbox = face["bbox"]
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            size = width * height
            sizes.append(size)
        avg_size = np.mean(sizes) if sizes else 0
        max_possible_size = frame_width * frame_height
        size_score = avg_size / max_possible_size

        # 2. 얼굴 품질 점수
        quality_scores = []
        for face in cluster_faces:
            frame_num = face["frame"]
            bbox = face["bbox"]
            conf = face["conf"]
            img_path = os.path.join(img_dir, f"shot_{shot_idx:04d}_frame_{frame_num:04d}.jpg")
            if os.path.exists(img_path):
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        x1, y1, x2, y2 = map(int, bbox)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
                        if x1 < x2 and y1 < y2:
                            face_img = img[y1:y2, x1:x2]
                            quality = calculate_face_quality(face_img, conf)
                            quality_scores.append(quality)
                except Exception as e:
                    print(f"이미지 처리 오류 ({img_path}): {e}")
        quality_score = np.mean(quality_scores) if quality_scores else 0

        # 3. 중앙 위치 점수
        center_distances = []
        for face in cluster_faces:
            bbox = face["bbox"]
            face_center_x = (bbox[0] + bbox[2]) / 2
            face_center_y = (bbox[1] + bbox[3]) / 2
            dist_x = abs(face_center_x - frame_width/2) / (frame_width/2)
            dist_y = abs(face_center_y - frame_height/2) / (frame_height/2)
            center_distance = 1 - (dist_x + dist_y) / 2
            center_distances.append(center_distance)
        center_score = np.mean(center_distances) if center_distances else 0

        total_score = (
            weights["quality"] * quality_score +
            weights["size"] * size_score +
            weights["center"] * center_score
        )
        scores[cluster_id] = total_score
        print(f"클러스터 {cluster_id}: 품질={quality_score:.3f}, 크기={size_score:.3f}, 중앙={center_score:.3f}, 총점={total_score:.3f}")
    return scores

def find_main_face_in_frame(faces, main_cluster_id, face_clusters):
    if main_cluster_id not in face_clusters or not face_clusters[main_cluster_id]:
        return None
    main_face_repr = face_clusters[main_cluster_id][0]
    main_bbox_repr = main_face_repr["bbox"]
    best_match = None
    best_iou = -1
    for face in faces:
        current_bbox = face[:4]
        x1 = max(main_bbox_repr[0], current_bbox[0])
        y1 = max(main_bbox_repr[1], current_bbox[1])
        x2 = min(main_bbox_repr[2], current_bbox[2])
        y2 = min(main_bbox_repr[3], current_bbox[3])
        if x2 < x1 or y2 < y1:
            continue
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (main_bbox_repr[2] - main_bbox_repr[0]) * (main_bbox_repr[3] - main_bbox_repr[1])
        area2 = (current_bbox[2] - current_bbox[0]) * (current_bbox[3] - current_bbox[1])
        union = area1 + area2 - intersection
        iou = intersection / union if union > 0 else 0
        if iou > best_iou:
            best_iou = iou
            best_match = face
    return best_match if best_iou > 0.1 else None

def post_process_main_characters(main_characters, max_size_change=0.5, window_size=5):
    if not main_characters:
        return []
    sorted_main = sorted(main_characters, key=lambda x: x["frame"])
    for i in range(1, len(sorted_main)):
        prev = sorted_main[i-1]["main"]
        curr = sorted_main[i]["main"]
        prev_size = (prev[2] - prev[0]) * (prev[3] - prev[1])
        curr_size = (curr[2] - curr[0]) * (curr[3] - curr[1])
        if prev_size > 0 and curr_size > 0:
            size_ratio = curr_size / prev_size
            if size_ratio > (1 + max_size_change) or size_ratio < (1 - max_size_change):
                print(f"프레임 {sorted_main[i]['frame']}: 바운딩 박스 크기 급변 보정 (비율: {size_ratio:.2f})")
                center_x = (curr[0] + curr[2]) / 2
                center_y = (curr[1] + curr[3]) / 2
                prev_width = prev[2] - prev[0]
                prev_height = prev[3] - prev[1]
                new_width = prev_width * (1 + np.sign(size_ratio - 1) * min(abs(size_ratio - 1), 0.2))
                new_height = prev_height * (1 + np.sign(size_ratio - 1) * min(abs(size_ratio - 1), 0.2))
                new_x1 = max(0, center_x - new_width / 2)
                new_y1 = max(0, center_y - new_height / 2)
                new_x2 = new_x1 + new_width
                new_y2 = new_y1 + new_height
                sorted_main[i]["main"] = [new_x1, new_y1, new_x2, new_y2] + curr[4:]
    return sorted_main

def create_main_character_images(main_characters, img_dir, output_directory):
    main_img_dir = os.path.join(output_directory, "main_images")
    if not os.path.exists(main_img_dir):
        os.makedirs(main_img_dir)
    for data in main_characters:
        shot_idx = data["shot"]
        frame_num = data["frame"]
        main_bbox = data["main"]
        img_path = os.path.join(img_dir, f"shot_{shot_idx:04d}_frame_{frame_num:04d}.jpg")
        if not os.path.exists(img_path):
            print(f"Warning: {img_path} 파일이 존재하지 않습니다.")
            continue
        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: {img_path} 이미지를 로드할 수 없습니다.")
                continue
            x1, y1, x2, y2 = map(int, main_bbox[:4])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            output_img_path = os.path.join(main_img_dir, f"shot{shot_idx:02d}_frame_{frame_num:04d}_main.jpg")
            cv2.imwrite(output_img_path, img)
        except Exception as e:
            print(f"이미지 처리 오류 ({img_path}): {e}")

def main():
    input_directory = "../mtcnn_output_data/2"
    output_directory = "../main_test3-2"
    shot_boundaries_path = "../input_data/shot_boundary_2.json"
    weights = {
        "quality": 0.5,
        "size": 0.3,
        "center": 0.2
    }
    try:
        result = select_main_character(input_directory, output_directory, shot_boundaries_path, weights)
        print("주인공 선별 완료:")
        print(json.dumps(result, indent=2))
    except Exception as e:
        import traceback
        print(f"오류 발생: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
