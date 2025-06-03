
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class MultiOrderDiffEncoder(nn.Module):
    def __init__(self):
        super(MultiOrderDiffEncoder, self).__init__()
        # 이 부분은 논문을 좀더 참고해봐야합니다.
        self.conv1 = nn.Conv2d(6, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, diff1, diff2):
        x = torch.cat((diff1, diff2), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x

class DyBDet(nn.Module):
    def __init__(self):
        super(DyBDet, self).__init__()
        backbone = torchvision.models.resnet50(pretrained=False)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.mde = MultiOrderDiffEncoder()

        self.exit1_fc = nn.Linear(256, 1)
        self.exit2_fc = nn.Linear(512, 1)
        self.exit3_fc = nn.Linear(2048, 1)

    def forward(self, x, diff1, diff2):
        diff_feats = self.mde(diff1, diff2)
        x = self.conv1(x)
        x = x + diff_feats
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        out1 = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        out1 = self.exit1_fc(out1).squeeze(-1)

        x = self.layer2(x)

        out2 = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        out2 = self.exit2_fc(out2).squeeze(-1)

        x = self.layer3(x)
        
        x = self.layer4(x)
        
        out3 = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        out3 = self.exit3_fc(out3).squeeze(-1)

        return out1, out2, out3

    def forward_exit(self, x, diff1, diff2, threshold=0.8, return_feature=False):
        # 탈출 함수
        diff_feats = self.mde(diff1, diff2)
        x = self.conv1(x) + diff_feats
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        out1 = self.exit1_fc(torch.flatten(F.adaptive_avg_pool2d(x1, (1, 1)), 1))
        prob1 = torch.sigmoid(out1)
        if prob1.item() >= threshold or (1 - prob1.item()) >= threshold:
            if return_feature:
                return out1.item(), 1, x1
            return out1.item(), 1, None

        x2 = self.layer2(x1)
        out2 = self.exit2_fc(torch.flatten(F.adaptive_avg_pool2d(x2, (1, 1)), 1))
        prob2 = torch.sigmoid(out2)
        if prob2.item() >= threshold or (1 - prob2.item()) >= threshold:
            if return_feature:
                return out2.item(), 2, x2
            return out2.item(), 2, None

        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        out3 = self.exit3_fc(torch.flatten(F.adaptive_avg_pool2d(x4, (1, 1)), 1))
        if return_feature:
            return out3.item(), 3, x4
        return out3.item(), 3, None
