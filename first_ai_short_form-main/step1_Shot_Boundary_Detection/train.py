# train.py
import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.dybdet_model import DyBDet
from dataloaders.autoshot import AutoShotDataset

def train():
    print("1. Configuration 파일 로드 시작")
    
    with open("configs/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    print("1. Configuration 파일 로드 완료")

    print("2. GPU 확인")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 중인 디바이스: {device}")

    print("3. Dataset 준비 시작")
    dataset_path = config['dataset']['root_dir']
    print(f"Dataset Path: {dataset_path}")

    try:
        dataset = AutoShotDataset(root_dir=dataset_path)
        print("3번 완료")
    except Exception as e:
        print(f"Dataset 로드 중 오류 발생: {e}")
        return

    print("4. DataLoader 준비 시작")
    try:
        dataloader = DataLoader(
            dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=4
        )
        print("4qjs 완료")
    except Exception as e:
        print(f"err: {e}")
        return

    # 모델 초기화
    print("5. 모델 초기화 시작")
    try:
        model = DyBDet().to(device)
        print("5번 완료")
    except Exception as e:
        print(f"err: {e}")
        return

    print("각ㅈㅇ 옵티마이저")
    try:
        learning_rate = float(config['training']['learning_rate'])
        pos_weight_value = float(config['training']['pos_weight'])
        
        # 디버깅 출력
        print(f"lR: {learning_rate} (Type: {type(learning_rate)})")
        print(f"Pos Weight: {pos_weight_value} (Type: {type(pos_weight_value)})")

        # pos_weight 텐서로 변환
        pos_weight = torch.tensor(pos_weight_value, device=device)

        # 손실 함수 및 옵티마이저
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    except Exception as e:
        print(f"err: {e}")
        return
    
    # Training 시작
    print("7. Training 시작")
    for epoch in range(config['training']['epochs']):
        print(f"\n=== Epoch [{epoch+1}/{config['training']['epochs']}] 시작 ===")
        model.train()
        running_loss = 0.0

        for i, (frames, diff1, diff2, labels) in enumerate(dataloader):
            try:
                print(f"Batch {i+1} - 데이터 로드 완료")

                frames = frames.to(device)
                diff1 = diff1.to(device)
                diff2 = diff2.to(device)
                labels = labels.to(device)
                print(f"Batch {i+1} - 데이터 GPU로 이동 완료")

                out1, out2, out3 = model(frames, diff1, diff2)
                print(f"Batch {i+1} - Forward pass 완료")

                loss1 = criterion(out1, labels)
                loss2 = criterion(out2, labels)
                loss3 = criterion(out3, labels)

                teacher_probs = torch.sigmoid(out3).detach()
                distill1 = F.mse_loss(torch.sigmoid(out1), teacher_probs)
                distill2 = F.mse_loss(torch.sigmoid(out2), teacher_probs)

                loss = loss1 + loss2 + loss3 + config['training']['distill_weight'] * (distill1 + distill2)
                print(f"Batch {i+1} - 완료: {loss.item():.4f}")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                if (i + 1) % 10 == 0:
                    print(f"Step [{i+1}/{len(dataloader)}] - Loss: {loss.item():.4f}")

            except Exception as e:
                print(f"Batch {i+1} err: {e}")
                continue

        avg_loss = running_loss / len(dataloader)
        print(f"=== Epoch [{epoch+1}/{config['training']['epochs']}] 완료 - 평균 Loss: {avg_loss:.4f} ===")

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
