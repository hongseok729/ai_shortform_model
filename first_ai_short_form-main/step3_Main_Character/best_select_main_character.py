import os
import json
import cv2
import numpy as np
import shutil
from skimage.filters import laplace
from skimage.util import img_as_float
import traceback
from typing import Dict, List, Tuple, Any, Optional

# ============================================================================
# 상수 정의 - 알고리즘 파라미터 설정
# ============================================================================
DEFAULT_MIN_FACE_SIZE = 40
DEFAULT_MIN_SHARPNESS = 15
DEFAULT_SMOOTHING_FACTOR = 0.6
DEFAULT_IOU_THRESHOLD = 0.3
DEFAULT_MAX_SIZE_CHANGE = 0.5
DEFAULT_WINDOW_SIZE = 5
DEFAULT_QUALITY_THRESHOLD = 0.3

# ============================================================================
# 유틸리티 함수들
# ============================================================================

def load_faces_from_json_dir(json_dir: str) -> List[Dict]:
    """JSON 디렉토리에서 얼굴 데이터 로드"""
    all_faces_data = []
    
    if not os.path.exists(json_dir):
        return all_faces_data
    
    for filename in sorted(os.listdir(json_dir)):
        if filename.endswith('.json'):
            json_path = os.path.join(json_dir, filename)
            try:
                with open(json_path, 'r') as f:
                    face_data = json.load(f)
                    if face_data and 'faces' in face_data:
                        all_faces_data.append(face_data)
            except Exception as e:
                print(f"JSON 파일 로드 오류 ({json_path}): {e}")
    
    return all_faces_data

def detect_frame_size(all_faces_data: List[Dict], img_dir: str) -> Tuple[int, int]:
    """첫 번째 이미지에서 프레임 크기 감지"""
    default_width, default_height = 1920, 1080
    
    if not all_faces_data:
        return default_width, default_height
    
    # 첫 번째 프레임의 이미지 파일 찾기
    first_frame = all_faces_data[0]
    shot_idx = first_frame.get("shot", 0)
    frame_num = first_frame.get("frame", 0)
    
    img_filename = f"shot_{shot_idx:04d}_frame_{frame_num:04d}.jpg"
    img_path = os.path.join(img_dir, img_filename)
    
    if os.path.exists(img_path):
        try:
            img = cv2.imread(img_path)
            if img is not None:
                height, width = img.shape[:2]
                return width, height
        except Exception as e:
            print(f"프레임 크기 감지 오류: {e}")
    
    return default_width, default_height

# ============================================================================
# 1단계: 향상된 선명도 계산 모듈
# ============================================================================

def calculate_enhanced_sharpness(face_img: np.ndarray) -> float:
    """향상된 선명도 계산 - 4가지 방법 조합"""
    if face_img is None or face_img.size == 0:
        return 0
    
    # 최소 크기 보장
    min_size = 20
    h, w = face_img.shape[:2]
    if h < min_size or w < min_size:
        scale = max(min_size / h, min_size / w)
        face_img = cv2.resize(face_img, (int(w * scale), int(h * scale)))
    
    # 그레이스케일 변환
    if len(face_img.shape) == 3:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_img
    
    # 1. 라플라시안 필터 기반 선명도 (가중치 40%)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # 2. 소벨 필터 기반 선명도 (가중치 30%)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_mean = np.mean(sobel_magnitude)
    
    # 3. 텐넨그래드 기반 선명도 (가중치 20%)
    tenengrad = np.sum(sobel_magnitude**2)
    
    # 4. 분산 기반 선명도 (가중치 10%)
    variance = np.var(gray)
    
    # 가중 평균으로 최종 선명도 계산
    sharpness_score = (
        0.4 * min(1.0, laplacian_var / 500.0) +
        0.3 * min(1.0, sobel_mean / 50.0) +
        0.2 * min(1.0, tenengrad / (gray.size * 10000)) +
        0.1 * min(1.0, variance / 1000.0)
    )
    
    return min(1.0, max(0.0, sharpness_score))

def calculate_face_quality_enhanced(face_img: np.ndarray, conf: float) -> float:
    """향상된 얼굴 이미지 품질 계산"""
    if face_img is None or face_img.size == 0:
        return 0
    
    # 선명도 계산
    sharpness = calculate_enhanced_sharpness(face_img)
    
    # 얼굴 크기 점수
    face_area = face_img.shape[0] * face_img.shape[1]
    size_score = min(1.0, face_area / (100 * 100))
    
    # 밝기 균일성 점수
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if len(face_img.shape) == 3 else face_img
    mean_brightness = np.mean(gray)
    brightness_score = 1.0 - abs(mean_brightness - 128) / 128
    
    # 최종 품질 점수 계산
    quality = (
        0.6 * sharpness +
        0.2 * conf +
        0.1 * size_score +
        0.1 * brightness_score
    )
    
    return min(1.0, max(0.0, quality))

# ============================================================================
# 2단계: 고품질 얼굴 선별 모듈
# ============================================================================

def get_best_quality_faces_per_shot(shot_faces: List[Dict], img_dir: str, 
                                  min_face_size: int = DEFAULT_MIN_FACE_SIZE,
                                  quality_threshold: float = DEFAULT_QUALITY_THRESHOLD) -> List[Dict]:
    """Shot 내에서 가장 선명한 얼굴들을 선별"""
    best_faces_data = []
    
    for frame_data in shot_faces:
        shot_idx = frame_data.get("shot", 0)
        frame_num = frame_data.get("frame", 0)
        faces = frame_data.get("faces", [])
        
        # 이미지 로드
        img_filename = f"shot_{shot_idx:04d}_frame_{frame_num:04d}.jpg"
        img_path = os.path.join(img_dir, img_filename)
        
        if not os.path.exists(img_path):
            continue
            
        try:
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # 각 얼굴의 품질 계산
            face_qualities = []
            for face_idx, face in enumerate(faces):
                x1, y1, x2, y2 = map(int, face[:4])
                conf = face[4] if len(face) > 4 else 0.5
                
                # 얼굴 크기 확인
                width = x2 - x1
                height = y2 - y1
                
                if width < min_face_size or height < min_face_size:
                    continue
                
                # 경계 확인
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
                
                if x1 < x2 and y1 < y2:
                    face_img = img[y1:y2, x1:x2]
                    quality = calculate_face_quality_enhanced(face_img, conf)
                    
                    if quality >= quality_threshold:
                        face_qualities.append({
                            'face_idx': face_idx,
                            'face': face,
                            'quality': quality,
                            'bbox': [x1, y1, x2, y2],
                            'conf': conf
                        })
            
            # 품질 순으로 정렬하여 상위 얼굴들만 선택
            face_qualities.sort(key=lambda x: x['quality'], reverse=True)
            
            selected_faces = []
            for face_data in face_qualities[:3]:
                if face_data['quality'] >= 0.5:
                    selected_faces.append(face_data['face'])
            
            if selected_faces:
                best_faces_data.append({
                    "shot": shot_idx,
                    "frame": frame_num,
                    "faces": selected_faces,
                    "qualities": [f['quality'] for f in face_qualities[:len(selected_faces)]]
                })
                
        except Exception as e:
            print(f"이미지 처리 오류 ({img_path}): {e}")
    
    return best_faces_data

# ============================================================================
# 3단계: 바운딩 박스 스무딩 모듈
# ============================================================================

def smooth_faces_across_frames(shot_faces: List[Dict], 
                             smoothing_factor: float = DEFAULT_SMOOTHING_FACTOR) -> List[Dict]:
    """프레임 간 얼굴 바운딩 박스를 스무딩"""
    if not shot_faces:
        return []
    
    # 프레임 번호로 정렬
    sorted_frames = sorted(shot_faces, key=lambda x: x["frame"])
    
    # 결과 리스트
    smoothed_shot_faces = []
    
    # 첫 번째 프레임은 그대로 사용
    smoothed_shot_faces.append(sorted_frames[0])
    prev_faces = sorted_frames[0]["faces"]
    
    for i in range(1, len(sorted_frames)):
        frame_data = sorted_frames[i].copy()
        curr_faces = frame_data["faces"]
        
        # 현재 프레임의 각 얼굴에 대해 스무딩 적용
        smoothed_faces = []
        
        for curr_face in curr_faces:
            # 이전 프레임의 가장 유사한 얼굴 찾기
            best_match = None
            best_iou = 0.3
            
            for prev_face in prev_faces:
                # IoU 계산
                x1 = max(curr_face[0], prev_face[0])
                y1 = max(curr_face[1], prev_face[1])
                x2 = min(curr_face[2], prev_face[2])
                y2 = min(curr_face[3], prev_face[3])
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                intersection = (x2 - x1) * (y2 - y1)
                curr_area = (curr_face[2] - curr_face[0]) * (curr_face[3] - curr_face[1])
                prev_area = (prev_face[2] - prev_face[0]) * (prev_face[3] - prev_face[1])
                union = curr_area + prev_area - intersection
                
                iou = intersection / union if union > 0 else 0
                
                if iou > best_iou:
                    best_iou = iou
                    best_match = prev_face
            
            # 일치하는 이전 얼굴이 있으면 스무딩 적용
            if best_match:
                smoothed_face = [
                    curr_face[0] * (1 - smoothing_factor) + best_match[0] * smoothing_factor,
                    curr_face[1] * (1 - smoothing_factor) + best_match[1] * smoothing_factor,
                    curr_face[2] * (1 - smoothing_factor) + best_match[2] * smoothing_factor,
                    curr_face[3] * (1 - smoothing_factor) + best_match[3] * smoothing_factor,
                    curr_face[4]
                ]
                smoothed_faces.append(smoothed_face)
            else:
                smoothed_faces.append(curr_face)
        
        frame_data["faces"] = smoothed_faces
        smoothed_shot_faces.append(frame_data)
        prev_faces = smoothed_faces
    
    return smoothed_shot_faces

# ============================================================================
# 4단계: 얼굴 클러스터링 모듈
# ============================================================================

def is_same_face(face1: Dict, face2: Dict, 
                iou_threshold: float = DEFAULT_IOU_THRESHOLD) -> bool:
    """두 얼굴이 동일 인물인지 판단"""
    bbox1 = face1["bbox"]
    bbox2 = face2["bbox"]
    
    # IoU 계산
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

def cluster_faces_enhanced(shot_faces: List[Dict], img_dir: str, 
                         iou_threshold: float = DEFAULT_IOU_THRESHOLD) -> Dict[int, List[Dict]]:
    """향상된 얼굴 클러스터링 (품질 기반)"""
    # 모든 얼굴 정보 수집
    all_faces = []
    for frame_data in shot_faces:
        frame_num = frame_data["frame"]
        shot_idx = frame_data["shot"]
        faces = frame_data["faces"]
        qualities = frame_data.get("qualities", [0.5] * len(faces))
        
        for face_idx, face in enumerate(faces):
            quality = qualities[face_idx] if face_idx < len(qualities) else 0.5
            all_faces.append({
                "shot": shot_idx,
                "frame": frame_num,
                "face_idx": face_idx,
                "bbox": face[:4],
                "conf": face[4] if len(face) > 4 else 0.5,
                "quality": quality
            })
    
    if not all_faces:
        return {}
    
    # 품질 순으로 정렬
    all_faces.sort(key=lambda x: x["quality"], reverse=True)
    
    # 얼굴 클러스터 초기화
    face_clusters = {}
    cluster_id = 0
    
    # 첫 번째 얼굴을 첫 번째 클러스터에 할당
    face_clusters[cluster_id] = [all_faces[0]]
    
    # 나머지 얼굴들에 대해 클러스터링
    for face in all_faces[1:]:
        assigned = False
        for cid, cluster_faces in face_clusters.items():
            for cluster_face in cluster_faces:
                if is_same_face(face, cluster_face, iou_threshold):
                    face_clusters[cid].append(face)
                    assigned = True
                    break
            
            if assigned:
                break
        
        if not assigned:
            cluster_id += 1
            face_clusters[cluster_id] = [face]
    
    # 클러스터 크기 기준 필터링 (최소 3개 이상)
    filtered_clusters = {cid: faces for cid, faces in face_clusters.items() if len(faces) >= 3}
    
    # 클러스터를 평균 품질 순으로 정렬하여 ID 재할당
    cluster_avg_quality = {}
    for cid, faces in filtered_clusters.items():
        avg_quality = np.mean([f["quality"] for f in faces])
        cluster_avg_quality[cid] = avg_quality
    
    remapped_clusters = {}
    sorted_clusters = sorted(cluster_avg_quality.items(), key=lambda x: x[1], reverse=True)
    
    for new_id, (old_id, avg_quality) in enumerate(sorted_clusters):
        remapped_clusters[new_id] = filtered_clusters[old_id]
        print(f"클러스터 {new_id}: {len(filtered_clusters[old_id])}개 얼굴, 평균 품질: {avg_quality:.3f}")
    
    return remapped_clusters

# ============================================================================
# 5단계: 주인공 점수 계산 모듈
# ============================================================================

def calculate_face_scores_enhanced(face_clusters: Dict[int, List[Dict]], shot_faces: List[Dict], 
                                 img_dir: str, weights: Dict[str, float], 
                                 frame_width: int, frame_height: int) -> Dict[int, float]:
    """향상된 얼굴 클러스터 점수 계산"""
    scores = {}
    
    for cluster_id, cluster_faces in face_clusters.items():
        # 1. 평균 품질 점수
        quality_scores = [face["quality"] for face in cluster_faces]
        avg_quality = np.mean(quality_scores)
        max_quality = np.max(quality_scores)
        quality_score = 0.7 * avg_quality + 0.3 * max_quality
        
        # 2. 얼굴 크기 점수
        sizes = []
        for face in cluster_faces:
            bbox = face["bbox"]
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            size = width * height
            sizes.append(size)
        
        avg_size = np.mean(sizes) if sizes else 0
        max_possible_size = frame_width * frame_height
        size_score = min(1.0, avg_size / (max_possible_size * 0.1))
        
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
        
        # 4. 빈도 점수
        total_faces = sum(len(cluster) for cluster in face_clusters.values())
        freq_score = len(cluster_faces) / total_faces if total_faces > 0 else 0
        
        # 복합 점수 계산
        total_score = (
            weights["quality"] * quality_score +
            weights["size"] * size_score +
            weights["center"] * center_score +
            weights.get("frequency", 0.1) * freq_score
        )
        
        scores[cluster_id] = total_score
        
        print(f"클러스터 {cluster_id} ({len(cluster_faces)}개 얼굴): "
              f"품질={quality_score:.3f}, 크기={size_score:.3f}, "
              f"중앙={center_score:.3f}, 빈도={freq_score:.3f}, 총점={total_score:.3f}")
    
    return scores

# ============================================================================
# 6단계: 주인공 선별 모듈
# ============================================================================

def find_main_face_in_frame(faces: List, main_cluster_id: int, 
                           face_clusters: Dict[int, List[Dict]]) -> Optional[List]:
    """특정 프레임에서 주인공 얼굴을 찾기"""
    if main_cluster_id not in face_clusters or not face_clusters[main_cluster_id]:
        return None
    
    main_cluster_faces = face_clusters[main_cluster_id]
    
    best_match = None
    best_iou = 0.3
    
    for face in faces:
        current_bbox = face[:4]
        
        for cluster_face in main_cluster_faces:
            cluster_bbox = cluster_face["bbox"]
            
            # IoU 계산
            x1 = max(current_bbox[0], cluster_bbox[0])
            y1 = max(current_bbox[1], cluster_bbox[1])
            x2 = min(current_bbox[2], cluster_bbox[2])
            y2 = min(current_bbox[3], cluster_bbox[3])
            
            if x2 < x1 or y2 < y1:
                continue
            
            intersection = (x2 - x1) * (y2 - y1)
            area1 = (current_bbox[2] - current_bbox[0]) * (current_bbox[3] - current_bbox[1])
            area2 = (cluster_bbox[2] - cluster_bbox[0]) * (cluster_bbox[3] - cluster_bbox[1])
            union = area1 + area2 - intersection
            
            iou = intersection / union if union > 0 else 0
            
            if iou > best_iou:
                best_iou = iou
                best_match = face
    
    return best_match

def select_main_in_shot_enhanced(shot_faces: List[Dict], shot_idx: int, img_dir: str, 
                               face_clusters: Dict[int, List[Dict]], weights: Dict[str, float], 
                               frame_width: int, frame_height: int) -> List[Dict]:
    """향상된 shot 내 주인공 선별"""
    # 각 클러스터별 점수 계산
    cluster_scores = calculate_face_scores_enhanced(face_clusters, shot_faces, img_dir, weights, frame_width, frame_height)
    
    if not cluster_scores:
        print(f"Shot {shot_idx}에서 얼굴 클러스터가 없습니다.")
        return []
    
    main_cluster_id = max(cluster_scores, key=cluster_scores.get)
    print(f"Shot {shot_idx}의 주인공: 클러스터 ID {main_cluster_id}, 점수 {cluster_scores[main_cluster_id]:.3f}")
    
    # 주인공 정보 생성
    main_characters = []
    for frame_data in shot_faces:
        frame_num = frame_data["frame"]
        faces = frame_data["faces"]
        
        main_face = find_main_face_in_frame(faces, main_cluster_id, face_clusters)
        
        if main_face:
            main_characters.append({
                "shot": shot_idx,
                "frame": frame_num,
                "main": main_face
            })
    
    return main_characters

# ============================================================================
# 7단계: 후처리 모듈
# ============================================================================

def post_process_main_characters(main_characters: List[Dict], 
                               max_size_change: float = DEFAULT_MAX_SIZE_CHANGE) -> List[Dict]:
    """주인공 선택 결과를 후처리하여 일관성을 유지"""
    if not main_characters:
        return []
    
    # 프레임 번호로 정렬
    sorted_main = sorted(main_characters, key=lambda x: x["frame"])
    
    # 바운딩 박스 크기 급변 보정
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

# ============================================================================
# 8단계: 결과 생성 모듈
# ============================================================================

def create_main_character_images(main_characters: List[Dict], img_dir: str, output_directory: str):
    """주인공 얼굴에 바운딩 박스를 그린 이미지를 생성"""
    main_img_dir = os.path.join(output_directory, "main_images")
    if not os.path.exists(main_img_dir):
        os.makedirs(main_img_dir)
    
    for data in main_characters:
        shot_idx = data["shot"]
        frame_num = data["frame"]
        main_bbox = data["main"]
        
        img_filename = f"shot_{shot_idx:04d}_frame_{frame_num:04d}.jpg"
        img_path = os.path.join(img_dir, img_filename)
        
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

# ============================================================================
# 9단계: 메인 함수
# ============================================================================

def select_main_character(input_directory: str, output_directory: str, 
                        shot_boundaries_path: Optional[str] = None, 
                        weights: Optional[Dict[str, float]] = None, 
                        by_shot: bool = True) -> Dict[str, Any]:
    """각 shot에서 주인공을 선별하는 메인 함수"""
    try:
        # 출력 디렉토리 생성
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        
        # 기본 가중치 설정
        default_weights = {
            "quality": 0.7,
            "size": 0.15,
            "center": 0.1,
            "frequency": 0.05
        }
        
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
        
        # 데이터 로드
        img_dir = os.path.join(input_directory, "img")
        json_dir = os.path.join(input_directory, "json")
        
        faces_detection_path = os.path.join(input_directory, "faces_detection.json")
        
        if os.path.exists(faces_detection_path):
            print(f"전체 얼굴 데이터 로드 중: {faces_detection_path}")
            with open(faces_detection_path, 'r') as f:
                all_faces_data = json.load(f)
        else:
            print(f"개별 JSON 파일에서 로드합니다.")
            all_faces_data = load_faces_from_json_dir(json_dir)
            
        if not all_faces_data:
            print(f"얼굴 데이터를 찾을 수 없습니다.")
            return {"error": "얼굴 데이터 없음"}
        
        # 프레임 크기 감지
        frame_width, frame_height = detect_frame_size(all_faces_data, img_dir)
        print(f"감지된 프레임 크기: {frame_width}x{frame_height}")
        
        # 필터링 매개변수
        min_face_size = DEFAULT_MIN_FACE_SIZE
        quality_threshold = DEFAULT_QUALITY_THRESHOLD
        
        all_main_characters = []
        
        # Shot별 데이터 그룹화
        print("Shot별로 주인공 선별 진행...")
        shots_data = {}
        for face_data in all_faces_data:
            shot_idx = face_data.get("shot", 0)
            if shot_idx not in shots_data:
                shots_data[shot_idx] = []
            shots_data[shot_idx].append(face_data)
        
        # 각 shot별 처리
        for shot_idx, shot_faces in shots_data.items():
            print(f"\n=== Shot {shot_idx} 주인공 선별 중... ({len(shot_faces)}개 프레임) ===")
            
            if not shot_faces:
                print(f"Shot {shot_idx}에 얼굴 데이터가 없습니다.")
                continue
            
            # 1. 고품질 얼굴 선별
            print("1단계: 고품질 얼굴 선별 중...")
            best_quality_faces = get_best_quality_faces_per_shot(shot_faces, img_dir, min_face_size, quality_threshold)
            print(f"고품질 얼굴 선별 후 {len(best_quality_faces)}개 프레임 남음")
            
            if not best_quality_faces:
                print(f"Shot {shot_idx}에 품질 기준을 통과한 얼굴이 없습니다.")
                continue
            
            # 2. 바운딩 박스 스무딩
            print("2단계: 바운딩 박스 스무딩 중...")
            smoothed_shot_faces = smooth_faces_across_frames(best_quality_faces)
            print(f"바운딩 박스 스무딩 완료")
            
            # 3. 얼굴 클러스터링
            print("3단계: 얼굴 클러스터링 중...")
            face_clusters = cluster_faces_enhanced(smoothed_shot_faces, img_dir)
            print(f"클러스터링 결과: {len(face_clusters)}개의 얼굴 클러스터")
            
            if not face_clusters:
                print(f"Shot {shot_idx}에 유효한 얼굴 클러스터가 없습니다.")
                continue
            
            # 4. 주인공 선별
            print("4단계: 주인공 선별 중...")
            shot_main_character = select_main_in_shot_enhanced(
                smoothed_shot_faces, 
                shot_idx, 
                img_dir, 
                face_clusters,
                default_weights,
                frame_width,
                frame_height
            )
            
            if shot_main_character:
                # 5. 후처리
                print("5단계: 후처리 중...")
                shot_main_character = post_process_main_characters(shot_main_character)
                all_main_characters.extend(shot_main_character)
                print(f"Shot {shot_idx} 주인공 선별 완료: {len(shot_main_character)}개 프레임")
        
        # 결과 생성 및 저장
        print("\n6단계: 결과 생성 중...")
        
        # 주인공 얼굴에 바운딩 박스 그리기 및 저장
        create_main_character_images(all_main_characters, img_dir, output_directory)
        
        # 모든 주인공 정보를 JSON 파일로 저장
        main_json_path = os.path.join(output_directory, "main_characters.json")
        with open(main_json_path, 'w') as f:
            json.dump(all_main_characters, f, indent=2)
        
        # 결과 정보 생성
        result = {
            "shot_boundaries_path": shot_boundaries_path,
            "main_characters": main_json_path,
            "output_directory": output_directory,
            "weights": default_weights,
            "by_shot": by_shot,
            "filter_settings": {
                "min_face_size": min_face_size,
                "quality_threshold": quality_threshold
            },
            "total_main_characters": len(all_main_characters),
            "processed_shots": len([shot for shot, faces in shots_data.items() if faces])
        }
        
        # 결과 정보 저장
        result_path = os.path.join(output_directory, "result.json")
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\n=== 주인공 선별 완료 ===")
        print(f"총 {len(all_main_characters)}개 프레임에서 주인공 검출")
        print(f"처리된 shot 수: {len([shot for shot, faces in shots_data.items() if faces])}")
        
        return result
    
    except Exception as e:
        print(f"오류 발생: {e}")
        traceback.print_exc()
        return {"error": str(e)}

def main():
    """메인 실행 함수"""
    # 입력 디렉토리 (step2의 결과)
    input_directory = "../mtcnn_output_data/1"
    
    # 출력 디렉토리 (step3의 결과)
    output_directory = "../output_data"
    
    # Shot boundary 파일 경로
    shot_boundaries_path = "../output_data/output_shot_boundaries.json"
    
    # 가중치 설정 (품질 중심)
    weights = {
        "quality": 0.6,
        "size": 0.2,
        "center": 0.15,
        "frequency": 0.05
    }
    
    # Shot별 주인공 선별 (고정)
    by_shot = True
    
    # 주인공 선별 실행
    result = select_main_character(input_directory, output_directory, shot_boundaries_path, weights, by_shot)
    
    # 결과 출력
    if "error" not in result:
        print("주인공 선별 완료:")
        print(json.dumps(result, indent=2))
    else:
        print(f"오류 발생: {result['error']}")

if __name__ == "__main__":
    main()
