import os
import json
import cv2
import numpy as np

def select_main_character(input_directory, output_directory, shot_boundaries_path):
    """
    각 shot에서 주인공을 선별하는 함수
    Args:
        input_directory: step2의 결과가 저장된 디렉토리 경로
        output_directory: 주인공 선별 결과를 저장할 디렉토리 경로
        shot_boundaries_path: shot 경계 정보가 저장된 JSON 파일 경로
    Returns:
        주인공 선별 결과 정보
    """
    # 출력 디렉토리 생성
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Shot boundaries 로드
    with open(shot_boundaries_path, 'r') as f:
        shot_boundaries = json.load(f)

    print(f"Shot boundaries 로드 완료: {len(shot_boundaries)}개의 shot")

    # 모든 shot의 주인공 정보를 저장할 리스트
    all_main_characters = []

    # json, img, frame 디렉토리 경로
    json_dir = os.path.join(input_directory, "json")
    img_dir = os.path.join(input_directory, "img")
    frame_dir = os.path.join(input_directory, "frames")

    # 각 shot별로 처리
    for shot_idx, shot in enumerate(shot_boundaries):
        print(f"Shot {shot_idx} 주인공 선별 중...")

        # shot의 모든 프레임 얼굴 정보 로드 (json/shot_0000_faces.json)
        shot_faces_path = os.path.join(json_dir, f"shot_{shot_idx:04d}_faces.json")

        if not os.path.exists(shot_faces_path):
            print(f"Warning: {shot_faces_path} 파일이 존재하지 않습니다.")
            continue

        with open(shot_faces_path, 'r') as f:
            shot_faces = json.load(f)

        # main 선별
        main_character = select_main_in_shot(shot_faces, shot_idx)

        if main_character:
            all_main_characters.extend(main_character)

            # main 얼굴에 바운딩 박스 그리기 및 저장
            create_main_character_images(main_character, frame_dir, output_directory)

    # 모든 main 정보를 JSON 파일로 저장
    main_json_path = os.path.join(output_directory, "main_characters.json")
    with open(main_json_path, 'w') as f:
        json.dump(all_main_characters, f, indent=2)

    # 결과 정보 생성
    result = {
        "shot_boundaries": shot_boundaries_path,
        "main_characters": main_json_path,
        "output_directory": output_directory
    }

    # 결과 정보 저장
    result_path = os.path.join(output_directory, "result.json")
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)

    return result

def select_main_in_shot(shot_faces, shot_idx):
    """
    한 shot 내에서 주인공을 선별하는 함수
    Args:
        shot_faces: shot의 모든 프레임 얼굴 정보
        shot_idx: shot 인덱스
    Returns:
        주인공 정보 리스트
    """
    # 얼굴 ID를 부여하기 위한 클러스터링
    face_clusters = cluster_faces(shot_faces)

    # 각 얼굴 클러스터별 점수 계산
    face_scores = calculate_face_scores(face_clusters, shot_faces)

    # 점수가 가장 높은 얼굴 클러스터를 main 선정
    if not face_scores:
        print(f"Shot {shot_idx}에서 얼굴이 감지되지 않았습니다.")
        return []

    main_cluster_id = max(face_scores, key=face_scores.get)
    print(f"Shot {shot_idx}의 main: 클러스터 ID {main_cluster_id}, 점수 {face_scores[main_cluster_id]:.2f}")

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

# 얼굴 클러스터링 (동일 인물 얼굴 그룹화)
def cluster_faces(shot_faces):
    all_faces = []
    for frame_data in shot_faces:
        frame_num = frame_data["frame"]
        for face_idx, face in enumerate(frame_data["faces"]):
            all_faces.append({
                "frame": frame_num,
                "face_idx": face_idx,
                "bbox": face[:4],  # [x1, y1, x2, y2]
                "conf": face[4]
            })

    if not all_faces:
        return {}

    face_clusters = {}
    cluster_id = 0
    face_clusters[cluster_id] = [all_faces[0]]

    for face in all_faces[1:]:
        assigned = False
        for cid, cluster_faces in face_clusters.items():
            if is_same_face(face, cluster_faces[0]):
                face_clusters[cid].append(face)
                assigned = True
                break
        if not assigned:
            cluster_id += 1
            face_clusters[cluster_id] = [face]
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

def calculate_face_scores(face_clusters, shot_faces):
    weights = {
        "frequency": 0.1,
        "size": 0.7,
        "center": 0.2
    }
    frame_width = 1920
    frame_height = 1080
    scores = {}
    total_frames = len(shot_faces)
    for cluster_id, cluster_faces in face_clusters.items():
        frequency = len(cluster_faces) / total_frames
        frequency_score = frequency
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
            weights["frequency"] * frequency_score +
            weights["size"] * size_score +
            weights["center"] * center_score
        )
        scores[cluster_id] = total_score
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

def create_main_character_images(main_characters, frame_dir, output_directory):
    main_img_dir = os.path.join(output_directory, "main_images")
    if not os.path.exists(main_img_dir):
        os.makedirs(main_img_dir)
    for data in main_characters:
        shot_idx = data["shot"]
        frame_num = data["frame"]
        main_bbox = data["main"]
        img_path = os.path.join(frame_dir, f"frame_{frame_num:04d}.jpg")
        if not os.path.exists(img_path):
            print(f"Warning: {img_path} 파일이 존재하지 않습니다.")
            continue
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: {img_path} 이미지를 로드할 수 없습니다.")
            continue
        x1, y1, x2, y2 = map(int, main_bbox[:4])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        output_img_path = os.path.join(main_img_dir, f"shot{shot_idx:02d}_frame_{frame_num:04d}_main.jpg")
        cv2.imwrite(output_img_path, img)

def main():
    input_directory = "../mtcnn_output_data/2"
    output_directory = "../main_day2-2"
    shot_boundaries_path = "../input_data/shot_boundary_2.json"
    try:
        result = select_main_character(input_directory, output_directory, shot_boundaries_path)
        print("주인공 선별 완료:")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    main()
