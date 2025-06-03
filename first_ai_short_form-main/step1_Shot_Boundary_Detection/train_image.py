# 이거도 결국 extract_frames.py이후에 하는 코드이지만...포기
import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.dybdet_model import DyBDet
from dataloaders.autoshotImagedataset import AutoShotImageDataset


def train():
    print("1. Configuration 파일 로드 시작")
    with open("configs/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    print("1. Configuration 파일 로드 완료")

    # GPU 확인
    print("2. GPU 확인")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 중인 디바이스: {device}")

    # Dataset 준비
    print("3. Dataset 준비 시작")
    dataset_path = config['dataset']['root_dir']
    frame_path = config['dataset']['frame_root']
    label_path = config['dataset']['label_root']
    print(f"Dataset Path: {dataset_path}")

    # Dataset 생성
    try:
        dataset = AutoShotImageDataset(frame_root=frame_path, label_root=dataset_path)
        print("3. Dataset 준비 완료")
    except Exception as e:
        print(f"Dataset 로드 중 오류 발생: {e}")
        return

    # DataLoader 준비
    print("4. DataLoader 준비 시작")
    try:
        dataloader = DataLoader(
            dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=4
        )
        print("4. DataLoader 준비 완료")
    except Exception as e:
        print(f"DataLoader 생성 중 오류 발생: {e}")
        return

    # 모델 초기화
    print("5. 모델 초기화 시작")
    try:
        model = DyBDet().to(device)
        print("5. 모델 초기화 완료")
    except Exception as e:
        print(f"모델 생성 중 오류 발생: {e}")
        return

    print("6. 손실 함수 및 옵티마이저 설정 시작")
    try:
        # 강제 형변환 추가
        learning_rate = float(config['training']['learning_rate'])
        pos_weight_value = float(config['training']['pos_weight'])
        
        # 디버깅 출력
        print(f"Learning Rate: {learning_rate} (Type: {type(learning_rate)})")
        print(f"Pos Weight: {pos_weight_value} (Type: {type(pos_weight_value)})")

        # pos_weight 텐서로 변환
        pos_weight = torch.tensor(pos_weight_value, device=device)

        # 손실 함수 및 옵티마이저
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        print("6. 손실 함수 및 옵티마이저 설정 완료")
    except Exception as e:
        print(f"손실 함수 또는 옵티마이저 설정 중 오류 발생: {e}")
        return
    
    # Training 시작
    print("7. Training 시작")
    for epoch in range(config['training']['epochs']):
        print(f"\n=== Epoch [{epoch+1}/{config['training']['epochs']}] 시작 ===")
        model.train()
        running_loss = 0.0

        for i, (frames, diff1, diff2, labels) in enumerate(dataloader):
            try:
                # 데이터 로딩 확인
                print(f"Batch {i+1} - 데이터 로드 완료")

                frames = frames.to(device)
                diff1 = diff1.to(device)
                diff2 = diff2.to(device)
                labels = labels.to(device)
                print(f"Batch {i+1} - 데이터 GPU로 이동 완료")

                out1, out2, out3 = model(frames, diff1, diff2)
                print(f"Batch {i+1} - Forward pass 완료")

                # 손실 계산
                loss1 = criterion(out1, labels)
                loss2 = criterion(out2, labels)
                loss3 = criterion(out3, labels)

                teacher_probs = torch.sigmoid(out3).detach()
                distill1 = F.mse_loss(torch.sigmoid(out1), teacher_probs)
                distill2 = F.mse_loss(torch.sigmoid(out2), teacher_probs)

                # 총 손실 계산
                loss = loss1 + loss2 + loss3 + config['training']['distill_weight'] * (distill1 + distill2)
                print(f"Batch {i+1} - 손실 계산 완료: {loss.item():.4f}")

                # 아마도 역전파
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                # 10 스텝마다 로그 출력
                if (i + 1) % 10 == 0:
                    print(f"Step [{i+1}/{len(dataloader)}] - Loss: {loss.item():.4f}")

            except Exception as e:
                print(f"Batch {i+1} 처리 중 오류 발생: {e}")
                continue

        avg_loss = running_loss / len(dataloader)
        print(f"=== Epoch [{epoch+1}/{config['training']['epochs']}] 완료 - 평균 Loss: {avg_loss:.4f} ===")

        # 모델 저장
        try:
            os.makedirs(config['output']['save_path'], exist_ok=True)
            save_path = os.path.join(config['output']['save_path'], f"dybdet_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"모델 저장 완료: {save_path}")
        except Exception as e:
            print(f"err: {e}")

if __name__ == "__main__":
    print("=== 학습 시작 ===")
    train()
