import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class AutoShotDataset(Dataset):
    """
    AutoShot데이터 로딩 버전3...메모리가 안터지니 프레임이 문제네...일단 수정 완료
    """

    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.root_dir = root_dir
        self.transform = transform

        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

        videos = [f for f in os.listdir(root_dir) if f.endswith(".mp4")]

        for video_file in videos:
            video_path = os.path.join(root_dir, video_file)
            txt_file = os.path.splitext(video_path)[0] + ".txt"
            if not os.path.exists(txt_file):
                continue

            boundaries = set()
            with open(txt_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) != 2:
                        continue
                    _, end = map(int, parts)
                    boundaries.add(end)
            

            # 프레임 갯수 확인...
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            for idx in range(2, total_frames):
                label = 1 if idx in boundaries else 0
                self.samples.append((video_path, idx, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, frame_idx, label = self.samples[idx]

        cap = cv2.VideoCapture(video_path)

        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 2)
            ret1, prev2 = cap.read()
            ret2, prev1 = cap.read()
            ret3, frame = cap.read()
            cap.release()

            if not (ret1 and ret2 and ret3):
                # #메모리 해결했더이 이제 영상에 프레임이 없다고 합니다...
                
                print(f"비상~ {video_id} {frame_idx}: {e}")
                frame = self.load_image(video_id, frame_idx)  # 현재 프레임만 fallback으로 사용
                prev1 = prev2 = frame  # 반복
                # print(f"왜 에러나는지 모르겠지만 프레임 에러 {frame_idx} 안에있는 {video_path}")
                # dummy_tensor = torch.zeros((3, 224, 224))
                # return dummy_tensor, dummy_tensor, dummy_tensor, torch.tensor(0.0)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            prev1 = cv2.cvtColor(prev1, cv2.COLOR_BGR2RGB)
            prev2 = cv2.cvtColor(prev2, cv2.COLOR_BGR2RGB)

            frame_t = self.transform(frame)
            prev1_t = self.transform(prev1)
            prev2_t = self.transform(prev2)

            diff1 = torch.abs(frame_t - prev1_t)
            diff2 = torch.abs(frame_t - prev2_t)

            label = torch.tensor(label, dtype=torch.float32)

            return frame_t, diff1, diff2, label

        # 더미 텐서보다는 이전 텐서를 사용하는게 흐름상 더 좋을듯해서 바꿈
        except Exception as e:
            print(f"비상~ {video_id} {frame_idx}: {e}")
            frame = self.load_image(video_id, frame_idx)  # 현재 프레임만 fallback으로 사용
            prev1 = prev2 = frame  # 반복

        # except Exception as e:
        #     print(f"[Error] Exception while processing frame {frame_idx} in {video_path}: {e}")
        #     # 더미 텐서 반환 - 이전 배치랑 크기는 유지해야하니까 반환해줍니다.
        #     dummy_tensor = torch.zeros((3, 224, 224))
        #     return dummy_tensor, dummy_tensor, dummy_tensor, torch.tensor(0.0)

