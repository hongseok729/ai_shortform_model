import json
import numpy as np
from scipy.interpolate import interp1d

# 경로 설정
main_path = "/workspace/kt_media1/all_json/main_characters1.json"
faces_path = "/workspace/kt_media1/all_json/faces_detection1.json"
output_json_path = "/workspace/kt_media1/results2/data1.json"

# JSON 로딩
with open(main_path) as f:
    main_data = json.load(f)
with open(faces_path) as f:
    face_data = json.load(f)

# 궤도 보간 함수 생성
main_by_shot = {}
for item in main_data:
    shot = item["shot"]
    frame = item["frame"]
    x1, y1, x2, y2 = item["main"][:4]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    if shot not in main_by_shot:
        main_by_shot[shot] = {"frames": [], "centers": []}
    main_by_shot[shot]["frames"].append(frame)
    main_by_shot[shot]["centers"].append((cx, cy))

orbit_paths = {}
for shot, data in main_by_shot.items():
    if len(data["frames"]) >= 3:
        cx_list, cy_list = zip(*data["centers"])
        interp_cx = interp1d(data["frames"], cx_list, kind='quadratic', fill_value='extrapolate')
        interp_cy = interp1d(data["frames"], cy_list, kind='quadratic', fill_value='extrapolate')
        orbit_paths[shot] = (interp_cx, interp_cy)

faces_by_frame = {item["frame"]: item for item in face_data}
tracking_result = []

prev_box = None
initial_w = initial_h = None
fallback_counter = 0
current_shot = None
main_start_frame = -1
main_start_box = None

# 파라미터
FORCE_RESET_DIST = 100
MAX_FALLBACK_FRAMES = 10
SCALE_LIMIT = 1.5
CENTER_DRIFT_LIMIT = 20
CENTER_DRIFT_LIMIT_Y = 10
SMOOTHING_ALPHA = 0.8
MAX_BBOX_WIDTH = 400
MAX_BBOX_HEIGHT = 400
PRED_CY_OFFSET = 20
SCALE_RATE = 1.01

for frame_idx in sorted(faces_by_frame.keys()):
    face_info = faces_by_frame[frame_idx]
    shot = face_info["shot"]
    faces = face_info["faces"]
    best_box = None

    if shot != current_shot:
        current_shot = shot
        shot_mains = [m for m in main_data if m["shot"] == shot]
        if shot_mains:
            first_main = sorted(shot_mains, key=lambda x: x["frame"])[0]
            main_start_frame = first_main["frame"]
            main_start_box = first_main["main"][:4]

    if main_start_box and frame_idx < main_start_frame:
        pred_cx = (main_start_box[0] + main_start_box[2]) / 2
        pred_cy = (main_start_box[1] + main_start_box[3]) / 2 + PRED_CY_OFFSET
        if prev_box:
            prev_cx = (prev_box[0] + prev_box[2]) / 2
            prev_cy = (prev_box[1] + prev_box[3]) / 2
            dx = np.clip(pred_cx - prev_cx, -CENTER_DRIFT_LIMIT, CENTER_DRIFT_LIMIT)
            dy = np.clip(pred_cy - prev_cy, -CENTER_DRIFT_LIMIT_Y, CENTER_DRIFT_LIMIT_Y)
            new_cx = prev_cx + dx
            new_cy = prev_cy + dy
            w = prev_box[2] - prev_box[0]
            h = prev_box[3] - prev_box[1]
            best_box = [new_cx - w / 2, new_cy - h / 2, new_cx + w / 2, new_cy + h / 2]
        else:
            w = main_start_box[2] - main_start_box[0]
            h = main_start_box[3] - main_start_box[1]
            best_box = [pred_cx - w / 2, pred_cy - h / 2, pred_cx + w / 2, pred_cy + h / 2]
        prev_box = best_box
        tracking_result.append({"frame": frame_idx, "shot": shot, "main": best_box})
        continue

    pred_cx = pred_cy = None
    if shot in orbit_paths:
        interp_cx, interp_cy = orbit_paths[shot]
        pred_cx = float(interp_cx(frame_idx))
        pred_cy = float(interp_cy(frame_idx)) + PRED_CY_OFFSET

    main_item = next((m for m in main_data if m["frame"] == frame_idx), None)
    if main_item:
        best_box = main_item["main"][:4]
        initial_w = best_box[2] - best_box[0]
        initial_h = best_box[3] - best_box[1]
        fallback_counter = 0

    elif faces and pred_cx and pred_cy:
        candidates = []
        for box in faces:
            if box[4] < 0.7:
                continue
            x1, y1, x2, y2 = box[:4]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            dist = np.linalg.norm([cx - pred_cx, cy - pred_cy])
            area = (x2 - x1) * (y2 - y1)
            expected_area = initial_w * initial_h if initial_w and initial_h else area
            score = 0.6 * box[4] + 0.4 * (1 / (1 + dist + abs(area - expected_area)))
            candidates.append((score, [x1, y1, x2, y2]))
        if candidates:
            best_box = max(candidates, key=lambda x: x[0])[1]
            fallback_counter = 0

    elif pred_cx and pred_cy:
        fallback_counter += 1
        if fallback_counter <= MAX_FALLBACK_FRAMES and prev_box:
            prev_x1, prev_y1, prev_x2, prev_y2 = prev_box
            prev_cx = (prev_x1 + prev_x2) / 2
            prev_cy = (prev_y1 + prev_y2) / 2
            dx = np.clip(pred_cx - prev_cx, -CENTER_DRIFT_LIMIT, CENTER_DRIFT_LIMIT)
            dy = np.clip(pred_cy - prev_cy, -CENTER_DRIFT_LIMIT_Y, CENTER_DRIFT_LIMIT_Y)
            new_cx = prev_cx + dx
            new_cy = prev_cy + dy
            prev_w = prev_x2 - prev_x1
            prev_h = prev_y2 - prev_y1
            decay_scale = min(1.1, max(1.0, SCALE_RATE - 0.01 * fallback_counter))
            new_w = min(prev_w * decay_scale, MAX_BBOX_WIDTH)
            new_h = min(prev_h * decay_scale, MAX_BBOX_HEIGHT)
            best_box = [new_cx - new_w / 2, new_cy - new_h / 2, new_cx + new_w / 2, new_cy + new_h / 2]
        else:
            continue

    if best_box and pred_cx and pred_cy:
        cx = (best_box[0] + best_box[2]) / 2
        cy = (best_box[1] + best_box[3]) / 2
        dist = np.linalg.norm([cx - pred_cx, cy - pred_cy])
        if dist > FORCE_RESET_DIST:
            w = best_box[2] - best_box[0]
            h = best_box[3] - best_box[1]
            best_box = [pred_cx - w / 2, pred_cy - h / 2, pred_cx + w / 2, pred_cy + h / 2]

    if best_box and prev_box:
        best_box = [SMOOTHING_ALPHA * b + (1 - SMOOTHING_ALPHA) * p for b, p in zip(best_box, prev_box)]

    if best_box:
        prev_box = best_box
        tracking_result.append({"frame": frame_idx, "shot": shot, "main": best_box})

# JSON 저장
with open(output_json_path, "w") as f:
    json.dump(tracking_result, f, indent=2)

print(f" 저장 : {output_json_path}")
