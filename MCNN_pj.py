import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


ORIG_IMG_DIR = '/root/testAnything/op_data'       
ORIG_ANN_DIR = '/root/testAnything/annotations'      
MCNN_WEIGHT_PATH = '/root/testAnything/mcnn_density_model.pt'
OUTPUT_VIS_DIR = '/root/testAnything/orig_compare_results'

IMG_HEIGHT = 384
IMG_WIDTH = 768
SCALE_FACTOR = 100.0

# 2. MCNN 모델 구조
class MCNN(nn.Module):
    def __init__(self):
        super(MCNN, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=9, padding=4), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=7, padding=3), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 16, kernel_size=7, padding=3), nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, kernel_size=7, padding=3), nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=7, padding=3), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(20, 40, kernel_size=5, padding=2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(40, 20, kernel_size=5, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(20, 10, kernel_size=5, padding=2), nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, padding=2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 24, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(24, 12, kernel_size=3, padding=1), nn.ReLU(inplace=True)
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(30, 1, kernel_size=1)
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        return self.fuse(torch.cat((x1, x2, x3), 1))

# 3. 원본 이미지 전체 일괄 추론 및 레드 도트 비교
def run_original_evaluation():
    if not os.path.exists(ORIG_IMG_DIR):
        print(f"[오류] 원본 이미지 폴더를 찾을 수 없습니다: {ORIG_IMG_DIR}")
        print("상단의 ORIG_IMG_DIR, ORIG_ANN_DIR 경로를 실제 원본 폴더 경로로 맞춰주세요.")
        return

    os.makedirs(OUTPUT_VIS_DIR, exist_ok=True)
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    img_files = sorted([f for f in os.listdir(ORIG_IMG_DIR) if f.lower().endswith(valid_exts)])

    if not img_files:
        print(f"[오류] {ORIG_IMG_DIR} 폴더에 이미지가 없습니다.")
        return

    mcnn = MCNN().to(device)
    mcnn.load_state_dict(torch.load(MCNN_WEIGHT_PATH, map_location=device, weights_only=True))
    mcnn.eval()

    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    total_mae, total_acc, valid_count = 0.0, 0.0, 0
    print("=" * 72)
    print(f"{'파일명':<25} | {'Red-Dot':>8} | {'MCNN 추정':>10} | {'오차':>8} | {'정확도':>7}")
    print("-" * 72)

    for fname in img_files:
        base_name = os.path.splitext(fname)[0]
        img_path = os.path.join(ORIG_IMG_DIR, fname)
        json_path = os.path.join(ORIG_ANN_DIR, base_name + '.json')

        orig_bgr = cv2.imread(img_path)
        if orig_bgr is None:
            continue
        h, w, _ = orig_bgr.shape

        # 원본 레드 도트 좌표 로드
        red_dots = []
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for shape in data.get('shapes', []):
                    for pt in shape.get('points', []):
                        red_dots.append((int(float(pt[0])), int(float(pt[1]))))
        dot_count = len(red_dots)

        # MCNN 추론
        pil_img = Image.fromarray(cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB))
        inp = transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_density = mcnn(inp)
            mcnn_count = int(round(max(0.0, pred_density.sum().item() / SCALE_FACTOR)))
            density_map = pred_density.squeeze().cpu().numpy()

        if dot_count > 0:
            diff = abs(mcnn_count - dot_count)
            acc = max(0.0, (1.0 - diff / dot_count) * 100.0)
            total_mae += diff
            total_acc += acc
            valid_count += 1
            print(f"{fname:<25} | {dot_count:>6}명 | {mcnn_count:>8d}명 | {diff:>6d}명 | {acc:>6.1f}%")
        else:
            print(f"{fname:<25} | {'JSON없음':>8} | {mcnn_count:>8f}명 | {'-':>8} | {'-':>7}")

        # 좌우 비교 이미지 저장 (좌: 원본 Red-Dot, 우: MCNN 히트맵)
        left_vis = orig_bgr.copy()
        for (cx, cy) in red_dots:
            cv2.circle(left_vis, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(left_vis, f"Red-Dot: {dot_count}", (25, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)

        norm_map = np.clip(density_map, 0, None)
        if norm_map.max() > 0:
            norm_map = (norm_map / norm_map.max() * 255).astype(np.uint8)
        else:
            norm_map = norm_map.astype(np.uint8)
        heatmap = cv2.applyColorMap(cv2.resize(norm_map, (w, h)), cv2.COLORMAP_JET)
        right_vis = cv2.addWeighted(orig_bgr, 0.55, heatmap, 0.45, 0)
        cv2.putText(right_vis, f"MCNN Density: {mcnn_count}", (25, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)

        combined = np.hstack((left_vis, right_vis))
        cv2.imwrite(os.path.join(OUTPUT_VIS_DIR, f"compare_{base_name}.jpg"), combined)

    print("=" * 72)
    if valid_count > 0:
        print(f"원본 전체 평균 MAE: {round(total_mae / valid_count)}명 | 평균 정확도: {round(total_acc / valid_count)}%")
    print(f"비교 결과 이미지 저장 폴더: {OUTPUT_VIS_DIR}/")

if __name__ == '__main__':
    run_original_evaluation()