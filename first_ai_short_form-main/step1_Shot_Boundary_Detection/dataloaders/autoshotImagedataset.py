import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class AutoShotImageDataset(Dataset):
    """
    영상에서 추출된 이미지 프레임 기반 AutoShot Dataset
    - 3장의 연속 프레임(frame, prev1, prev2)을 기반으로 diff1, diff2 계산
    - 프레임이 깨졌거나 없을 경우 더미 텐서 반환 (모델 학습 안 끊기게)
    - txt 파일 기준으로 boundary 라벨 구성
    """

    def __init__(self, frame_root, label_root, transform=None):
        self.samples = []  # (video_id, frame_idx, label)
        self.frame_root = frame_root
        self.label_root = label_root
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        video_dirs = [d for d in os.listdir(frame_root) if os.path.isdir(os.path.join(frame_root, d))]
        print(f"[DEBUG] 총 영상 수: {len(video_dirs)}")

        for video_id in video_dirs:
            frame_dir = os.path.join(frame_root, video_id)
            print(f"[check] id : {video_id}")
            label_file = os.path.join(label_root, f"{video_id}.txt")
            if not os.path.exists(label_file):
                print(f"[WARNING] 라벨 파일 없음: {label_file}")
                continue

            # boundary 라벨 파싱
            boundaries = set()
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().replace(',', ' ').split()
                    if len(parts) == 2:
                        try:
                            _, end = map(int, parts)
                            boundaries.add(end)
                        except ValueError:
                            print(f"[WARNING] 잘못된 숫자: {line.strip()}")

            if len(boundaries) == 0:
                print(f"[WARNING] boundary 수가 0입니다: {label_file}")
            # 프레임 수 계산
            # 찾았다 범인

            total_frames = len([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])
            print(f"[DEBUG] 영상 {video_id} - 프레임 수: {total_frames}, boundary 수: {len(boundaries)}")

            for idx in range(2, total_frames):
                label = 1 if idx in boundaries else 0
                self.samples.append((video_id, idx, label))

        print(f"[DEBUG] 총 샘플 수: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def load_image(self, video_id, frame_idx):
        path = os.path.join(self.frame_root, video_id, f"frame_{frame_idx:05d}.jpg")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing frame: {path}")
        image = Image.open(path).convert("RGB")
        return self.transform(image)

    def __getitem__(self, idx):
        video_id, frame_idx, label = self.samples[idx]

        try:
            frame = self.load_image(video_id, frame_idx)
            prev1 = self.load_image(video_id, frame_idx - 1)
            prev2 = self.load_image(video_id, frame_idx - 2)

            diff1 = torch.abs(frame - prev1)
            diff2 = torch.abs(frame - prev2)

            label = torch.tensor(label, dtype=torch.float32)
            return frame, diff1, diff2, label

        except Exception as e:
            # print(f"[ERROR] 프레임 로드 실패 -> video_id: {video_id}, frame_idx: {frame_idx} :: {e}")
            # 찾음

            dummy = torch.zeros((3, 224, 224))
            return dummy, dummy, dummy, torch.tensor(0.0)
