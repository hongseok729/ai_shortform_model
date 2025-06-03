import os
import cv2
import yaml
import torch
import gc
import math
from models.dybdet_model import DyBDet
import json


def is_recently_detected(current_frame_idx, boundaries, within=3):
    for _, detected_frame in reversed(boundaries):
        if current_frame_idx - detected_frame <= within:
            return True
        if current_frame_idx - detected_frame > within:
            break  # 시간 초과된 오래된 컷이므로 더 이상 확인할 필요 없음
    return False

def format_timestamp(frame_num, fps):
    total_seconds = frame_num / fps
    days = int(total_seconds // 86400)
    total_seconds %= 86400
    hours = int(total_seconds // 3600)
    total_seconds %= 3600
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    return f"{days:02d}{hours:02d}{minutes:02d}{seconds:02d}"

def filter_close_predictions(preds, min_frame_gap=10):
    if not preds:
        return []

    filtered = []
    group = [preds[0]]

    for i in range(1, len(preds)):
        prev_ts, prev_frame = group[-1]
        curr_ts, curr_frame = preds[i]

        if curr_frame - prev_frame < min_frame_gap:
            group.append(preds[i])
        else:
            # 현재 그룹에서 마지막 프레임만 남기고 저장
            filtered.append(group[-1])
            group = [preds[i]]

    # 마지막 그룹도 반영
    if group:
        filtered.append(group[-1])

    return filtered

def infer_video(video_path, model, device, threshold):
    print(f"[INFO] 영상 로딩 중: {video_path}")
    last_pred_prob = None
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 실제 프레임 수 계산 (cap.get 사용하지 않고 직접 계산)
    temp_cap = cv2.VideoCapture(video_path)
    total_frames = 0
    while True:
        ret, _ = temp_cap.read()
        if not ret:
            break
        total_frames += 1
    temp_cap.release()

    print(f"[INFO] 총 프레임 수 (실제 카운트): {total_frames}")

    boundaries = []

    prev2 = prev1 = None
    current_idx = 0

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        current_idx += 1

        if current_idx < 3:
            prev2 = prev1
            prev1 = frame_rgb
            continue

        try:
            frame_t = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
            prev1_t = torch.from_numpy(prev1).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
            prev2_t = torch.from_numpy(prev2).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

            frame_t = (frame_t - mean) / std
            prev1_t = (prev1_t - mean) / std
            prev2_t = (prev2_t - mean) / std

            diff1_t = torch.abs(frame_t - prev1_t)
            diff2_t = torch.abs(frame_t - prev2_t)

            with torch.no_grad():
                pred_logit, exit_level, feature = model.forward_exit(
                    frame_t, diff1_t, diff2_t, threshold=threshold, return_feature=True
                )

                if isinstance(pred_logit, torch.Tensor):
                    pred_prob = torch.sigmoid(pred_logit).item()
                else:
                    pred_prob = 1 / (1 + math.exp(-pred_logit))

                print(f"[DEBUG] Frame {current_idx}: pred_prob={pred_prob:.4f}, exit={exit_level}")
                # 특이 컷 감지: 이전 확률과 4배 이상 차이 나면 컷 추가
                if last_pred_prob is not None and pred_prob > 0 and last_pred_prob > 0:
                    ratio = max(pred_prob / last_pred_prob, last_pred_prob / pred_prob)
                    recent_frame_idx = current_idx - 1
                    recent_prob = last_pred_prob

                    if ratio >= 5.0:
                        # 확률 0.04 이상 조건 먼저 보지 않고 lookahead 조건까지 포함
                        if pred_prob >= 0.04 or recent_prob >= 0.04:
                            if not is_recently_detected(recent_frame_idx, boundaries, within=3):
                                time_str = format_timestamp(recent_frame_idx, fps)
                                print(f"[SPECIAL DETECT] 컷 감지! Frame {recent_frame_idx}, Timestamp: {time_str}, 비율 차이: {ratio:.2f}")
                                boundaries.append((time_str, recent_frame_idx))
                        else:
                            # lookahead: 다음 프레임들과 비교
                            lookahead_idx = 0
                            lookahead_limit = 2  # 2프레임 정도만 검사
                            while lookahead_idx < lookahead_limit:
                                ret_next, frame_next = cap.read()
                                if not ret_next:
                                    break
                                current_idx += 1
                                frame_rgb_next = cv2.cvtColor(frame_next, cv2.COLOR_BGR2RGB)
                                frame_t_next = torch.from_numpy(frame_rgb_next).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
                                frame_t_next = (frame_t_next - mean) / std

                                diff1_t_next = torch.abs(frame_t_next - frame_t)
                                diff2_t_next = torch.abs(frame_t_next - prev1_t)

                                with torch.no_grad():
                                    pred_logit_next, _, _ = model.forward_exit(
                                        frame_t_next, diff1_t_next, diff2_t_next, threshold=threshold, return_feature=False
                                    )
                                    pred_prob_next = torch.sigmoid(pred_logit_next).item() if isinstance(pred_logit_next, torch.Tensor) else 1 / (1 + math.exp(-pred_logit_next))

                                ratio_next = max(pred_prob_next / recent_prob, recent_prob / pred_prob_next)
                                if ratio_next >= 5.0 and (pred_prob_next >= 0.04 or recent_prob >= 0.04):
                                    if not is_recently_detected(recent_frame_idx, boundaries, within=3):
                                        time_str = format_timestamp(recent_frame_idx, fps)
                                        print(f"[SPECIAL DETECT] 컷 감지 (lookahead)! Frame {recent_frame_idx}, Timestamp: {time_str}, 비율 차이: {ratio_next:.2f}")
                                        boundaries.append((time_str, recent_frame_idx))
                                        break

                                lookahead_idx += 1


                last_pred_prob = pred_prob

                # 일반 컷 감지
                if pred_prob >= threshold and not is_recently_detected(current_idx, boundaries, within=3):
                    time_str = format_timestamp(current_idx, fps)
                    print(f"[DETECT] 컷 경계 감지! Frame {current_idx}, Timestamp: {time_str}, 확률: {pred_prob:.4f}")
                    boundaries.append((time_str, current_idx))

            del frame_t, prev1_t, prev2_t, diff1_t, diff2_t
            torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            print(f"[ERROR] Frame {current_idx} 처리 중 오류 발생: {e}")
            break

        prev2 = prev1
        prev1 = frame_rgb

    cap.release()
    print(f"[INFO] 추론 완료. 컷 감지 수: {len(boundaries)}")
    return boundaries

def run_inference(video_path):
    print("[INFO] config.yaml 로드 중...")
    config_path = os.path.join(os.path.dirname(__file__), "configs/config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(config['model']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] 디바이스 설정 완료: {device}")

    model = DyBDet().to(device)
    model_path = os.path.join(os.path.dirname(__file__), "../input_data/dybdet_epoch.pth")
    print(f"[INFO] 모델 가중치 로드 중: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("[INFO] 모델 로딩 완료, 추론 시작")

    threshold = config['model']['threshold']
    boundaries = infer_video(video_path, model, device, threshold)
    return boundaries

if __name__ == "__main__":
    video_file = "input_data/data_1.mp4"
    preds = run_inference(video_file)

    preds = filter_close_predictions(preds, min_frame_gap=10)

    print(f"\n 총 {len(preds)}개의 컷 경계가 감지되었습니다.")
    for ts, frame in preds:
        print(f" - Frame {frame} @ {ts}")

    # fps 다시 계산
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = 0
    while cap.read()[0]:
        total_frames += 1
    cap.release()

    # 컷 프레임만 추출
    cut_frames = [frame for _, frame in preds]

    # shot 단위로 변환
    shots = []
    prev_frame = 0
    for idx, cut_frame in enumerate(cut_frames):
        shots.append({
            "shot": idx,
            "start": format_timestamp(prev_frame, fps),
            "end": format_timestamp(cut_frame - 1, fps),
            "start_frame": prev_frame,
            "end_frame": cut_frame - 1
        })
        prev_frame = cut_frame

    # 마지막 shot 추가
    shots.append({
        "shot": len(shots),
        "start": format_timestamp(prev_frame, fps),
        "end": format_timestamp(total_frames - 1, fps),
        "start_frame": prev_frame,
        "end_frame": total_frames - 1
    })

    # JSON 저장
    output_path = os.path.join(os.path.dirname(__file__), "../output_data/dybdet_epoch.pth")
    with open(output_path, "w") as f:
        json.dump(shots, f, indent=2)

    print(f"\nJSON 저장 완료: {output_path}")
