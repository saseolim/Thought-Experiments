import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import math
import time
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sum_series(f, i, N):
    if N - i < 0:
        return 0
    return sum(f(x) for x in range(i, N + 1))

def NowDepthInCount(K, D, N):
    return N - sum_series(lambda x: K ** x, 1, D-1)

def ThisNodeChildLow(K, D, N):
    if D == 0:
        return 1
    return 1 + sum_series(lambda x: K ** x, 1, D) + K * (-1 + NowDepthInCount(K,D,N))

def ThisNodeChildHigh(K, D, N):
    if D == 0:
        return K
    return sum_series(lambda x: K ** x, 1, D) + K * (NowDepthInCount(K,D,N))

def ThisNodeParent(K,D,N):
    return math.ceil((N - sum_series(lambda x: K ** x, 1, D-1))/(K)) + sum_series(lambda x: K ** x, 1, D-2)

def ThisNodeDepth(K,N):
    Depth = 1
    if N == 0:
        return 0
    while True:
        if sum_series(lambda x: K ** x, 1, Depth-1) < N and N <= sum_series(lambda x: K **x,1,Depth):
            break
        Depth += 1
    return Depth

def GetThisChild(K,D,N):
    return range(ThisNodeChildLow(K,D,N),ThisNodeChildHigh(K,D,N))

def GetThisToRoot(K,D,N):
    dN = N
    Out = list()
    Out.append(N)
    for d in range(0,D):
        dN = ThisNodeParent(K,D,dN)
        Out.append(dN)
    return Out



def load_image(path):
    img = Image.open(path).convert("RGB")
    return torch.tensor(np.array(img), dtype=torch.float32).to(device)

def compute_axis_accuracy(block, axis): #가로나 세로의 정확도 : 1/(1 + sum{블록 끝과 끝의 RGB 차이 벡터 ** 2}) 즉 편미분 기반의 1D 변분 정확도
    if axis == 0 and block.shape[0] > 1:
        diff = block[1:, :, :] - block[:-1, :, :]
    elif axis == 1 and block.shape[1] > 1:
        diff = block[:, 1:, :] - block[:, :-1, :]
    else:
        return torch.tensor(1.0, device=device)
    return 1.0 / (1.0 + torch.sum(diff ** 2))

def compute_accuracy(block): #2D정확도
    diff_x = block[1:,:,:] - block[:-1,:,:]
    diff_y = block[:,1:,:] - block[:,:-1,:]
    total_diff = torch.sum(diff_x ** 2) + torch.sum(diff_y ** 2)
    return 1/ (1+ total_diff)

def generate_G_from_F_block(F, y1, y2, x1, x2):
    """
    F: 전체 이미지 텐서 (H, W, 3)
    y1, y2, x1, x2: 블록 경계
    return: 블록 G(x, y) (H, W, 3)
    """
    block = F[y1:y2, x1:x2, :]
    H, W = block.shape[:2]

    # 경계 기반으로 기울기 추정 (중앙값 이용)
    #gx = (block[:, -1, :] - block[:, 0, :]) / max(W - 1, 1)
    #gy = (block[-1, :, :] - block[0, :, :]) / max(H - 1, 1)

    #gx = torch.mean(block[:, 1:, :] - block[:, :-1, :], dim=(0,1))# * W / (W-1) if W > 1 else torch.zeros(3, device=device)
    #gy = torch.mean(block[1:, :, :] - block[:-1, :, :], dim=(0,1))# * H / (H-1) if H > 1 else torch.zeros(3, device=device)

    #gl = (block[0, 0, :] - block[-1, -1, :]) / math.sqrt((W - 1) ** 2 + (H - 1) ** 2) if W > 1 or H > 1 else torch.zeros(3, device=device)
    gx = (block[0, -1, :] - block[0, 0, :]) / (W - 1) if W > 1 else torch.zeros(3, device=device)
    gy = (block[-1, 0, :] - block[0, 0, :]) / (H - 1) if H > 1 else torch.zeros(3, device=device)
    #dg = (gl -(gx + gy))/2
    #gx += dg
    #gy += dg

    #gx = gx.mean(dim=0)  # (3,)
    #gy = gy.mean(dim=0)  # (3,)

    ##base = torch.amin(block.reshape(-1, 3), dim=0)
    base = block[0, 0, :]  # 왼쪽 위 기준

    # 좌표 생성
    i = torch.arange(H, device=F.device).view(-1, 1).expand(H, W).float()
    j = torch.arange(W, device=F.device).view(1, -1).expand(H, W).float()

    # G(x, y) = gx * x + gy * y + base
    G = gx.view(1, 1, 3) * j.unsqueeze(2) + gy.view(1, 1, 3) * i.unsqueeze(2) + base.view(1, 1, 3)

    return G

def compute_accuracy_variational_2d(f, g):
    grad_f_x = f[:, 1:, :] - f[:, :-1, :] #* f.shape[1] / (f.shape[1]-1) if f.shape[1] > 1 else f[:, 1:, :] - f[:, :-1, :]
    grad_f_y = f[1:, :, :] - f[:-1, :, :] #* f.shape[0] / (f.shape[0]-1) if f.shape[0] > 1 else f[1:, :, :] - f[:-1, :, :]
    grad_g_x = g[:, 1:, :] - g[:, :-1, :]
    grad_g_y = g[1:, :, :] - g[:-1, :, :]

    min_h = min(grad_f_x.shape[0], grad_f_y.shape[0])
    min_w = min(grad_f_x.shape[1], grad_f_y.shape[1])

    grad_f = torch.stack((grad_f_x[:min_h, :min_w], grad_f_y[:min_h, :min_w]), dim=0)
    grad_g = torch.stack((grad_g_x[:min_h, :min_w], grad_g_y[:min_h, :min_w]), dim=0)

    diff = grad_f - grad_g
    diff_squared = (diff ** 2).sqrt().sum(dim=0)
    total_error = diff_squared.sum()

    return 1.0 / (1.0 + total_error)

def compute_accuracy_variational_1d(f, g, axis):
    if axis == 0:  # y축
        grad_f = f[1:, :, :] - f[:-1, :, :] #* f.shape[0] / (f.shape[0]-1) if f.shape[0] > 1 else f[1:, :, :] - f[:-1, :, :]
        grad_g = g[1:, :, :] - g[:-1, :, :]
    elif axis == 1:  # x축
        grad_f = f[:, 1:, :] - f[:, :-1, :] #* f.shape[1] / (f.shape[1]-1) if f.shape[1] > 1 else f[:, 1:, :] - f[:, :-1, :]
        grad_g = g[:, 1:, :] - g[:, :-1, :]
    else:
        raise ValueError("axis must be 0 (y) or 1 (x)")

    diff = grad_f - grad_g
    diff_squared = (diff ** 2).sqrt().sum(dim=0)

    return 1.0 / (1.0 + diff_squared.sum())


def adaptive_split(F, acc_threshold=0.98, min_size=1, max_depth=12):
    H, W, _ = F.shape
    stack = [(0, 0, 0, W, H, 0)]
    blocks = []
    matablocks = []

    while stack:
        id, x1, y1, x2, y2, depth = stack.pop()
        w, h = x2 - x1, y2 - y1
        matablocks.append((id, x1, y1, x2, y2, depth))
        if w <= 0 or h <= 0 or depth >= max_depth:
            blocks.append((id, x1, y1, x2, y2, depth)) # 버그 발생시 이 부분 삭제 바람
            del matablocks[matablocks.__len__() - 1]
            continue

        block = F[y1:y2, x1:x2, :]
        G = generate_G_from_F_block(F,y1,y2,x1,x2)
        acc_x = compute_accuracy_variational_1d(block,G,1)
        acc_y = compute_accuracy_variational_1d(block,G,0)
        acc_total = compute_accuracy_variational_2d(block,G)
        #acc_x = compute_axis_accuracy(block,1)
        #acc_y = compute_axis_accuracy(block,0)
        #acc_total = compute_accuracy(block)

        childId = ThisNodeChildLow(4,depth,id)

        if acc_total < acc_threshold and w > 1 and h > 1:
            mx, my = (x1 + x2) // 2, (y1 + y2) // 2
            stack.extend([
                (childId, x1, y1, mx, my, depth+1),
                (childId + 1, mx, y1, x2, my, depth+1),
                (childId + 2, x1, my, mx, y2, depth+1),
                (childId + 3, mx, my, x2, y2, depth+1),
            ])
        elif acc_x < acc_threshold and acc_y < acc_threshold and w > 1 and h > 1:
            mx, my = (x1 + x2) // 2, (y1 + y2) // 2
            stack.extend([
                (childId, x1, y1, mx, my, depth+1),
                (childId + 1, mx, y1, x2, my, depth+1),
                (childId + 2, x1, my, mx, y2, depth+1),
                (childId + 3, mx, my, x2, y2, depth+1),
            ])
        elif acc_x < acc_threshold and w > 1:
            mx = (x1 + x2) // 2
            stack.extend([(childId, x1, y1, mx, y2, depth+1), (childId + 1, mx, y1, x2, y2, depth+1)])
        elif acc_y < acc_threshold and h > 1:
            my = (y1 + y2) // 2
            stack.extend([(childId, x1, y1, x2, my, depth+1), (childId + 2, x1, my, x2, y2, depth+1)])
        else:
            blocks.append((id, x1, y1, x2, y2, depth))
            del matablocks[matablocks.__len__() - 1]
    return blocks, matablocks

def compute_block_vectors(F, blocks):
    vectors = []
    for id, x1, y1, x2, y2, d in blocks:
        region = F[y1:y2, x1:x2, :]
        h, w = region.shape[:2]
        #x_s = max(0, x1-1)
        #x_e = min(F.shape[1], x1+1)
        #y_s = max(0, y1-1)
        #y_e = min(F.shape[0], y1+1)
        #TxTregion = F[y_s:y_e, x_s:x_e, :]
        ##gx = torch.mean(region[:, 1:, :] - region[:, :-1, :], dim=(0, 1)) if w > 1 else torch.zeros(3, device=device) # torch.mean(TxTregion[:, 1:, :] - TxTregion[:, :-1, :], dim=(0, 1)) #torch.zeros(3, device=device)
        ##gy = torch.mean(region[1:, :, :] - region[:-1, :, :], dim=(0, 1)) if h > 1 else torch.zeros(3, device=device) # torch.mean(TxTregion[1:, :, :] - TxTregion[:-1, :, :], dim=(0, 1)) #torch.zeros(3, device=device)
        #gl = (region[0, 0, :] - region[-1, -1, :]) / math.sqrt((w - 1) ** 2 + (h - 1) ** 2) if w > 1 or h > 1 else torch.zeros(3, device=device)
        gx = (region[0, -1, :] - region[0, 0, :]) / (w - 1) if w > 1 else torch.zeros(3, device=device)
        gy = (region[-1, 0, :] - region[0, 0, :]) / (h - 1) if h > 1 else torch.zeros(3, device=device)
        #dg = (gl -(gx + gy))/2
        #gx += dg
        #gy += dg

        ##base = torch.amin(region.reshape(-1, 3), dim=0)
        base = region[0,0,:]
        vectors.append({
            'id' : id,
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'depth': d,
            'grad_x': gx.detach().cpu().tolist(),
            'grad_y': gy.detach().cpu().tolist(),
            'min_val': base.detach().cpu().tolist()
        })
    return vectors

def reconstruct(F_shape, vectors):
    H, W, _ = F_shape
    img = np.zeros((H, W, 3), dtype=np.float32)
    for v in vectors:
        x1, y1, x2, y2 = v['x1'], v['y1'], v['x2'], v['y2']
        gx, gy, base = v['grad_x'], v['grad_y'], v['min_val']
        w = np.float32(x2 - x1)
        h = np.float32(y2 - y1)
        for i in range(y2 - y1):
            for j in range(x2 - x1):
                for c in range(3):
                    """
                    wc = 1
                    hc = 1
                    if w > 1:
                        wc = w/(w-1)
                    if h > 1:
                        hc = h/(h-1)
                    """
                    img[y1+i, x1+j, c] = gx[c]*j + gy[c]*i + base[c]
    return np.clip(img, 0, 255).astype(np.uint8)

def linearreconstruct(F_shape, vectors):
    H, W, _ = F_shape
    img = np.zeros((H, W, 3), dtype=np.float32)
    for v in vectors:
        x1, y1, x2, y2 = v['x1'], v['y1'], v['x2'], v['y2']
        gx, gy, base = v['grad_x'], v['grad_y'], v['min_val']
        for i in range(y2 - y1):
            for j in range(x2 - x1):
                for c in range(3):
                    img[y1+i, x1+j, c] = ((x2 - x1 - 1) * gx[c])/2 + ((y2 - y1 - 1) * gy[c])/2 + base[c]
    return np.clip(img, 0, 255).astype(np.uint8)

def eta(t): return (1) * (t ** 2) * ((1 - t) ** 2)  #(t ** 2) * ((1 - t) ** 2) #t * (1 - t) #(t - 1/(t+1)) 아트틱

def lsjs_fill(F_shape, vectors, patch=4): #4
    H, W, _ = F_shape
    img = np.zeros((H, W, 3), dtype=np.float32)
    block_map = np.empty((H, W), dtype=object)

    for v in vectors:
        x1, y1, x2, y2 = v['x1'], v['y1'], v['x2'], v['y2']
        gx, gy, base = v['grad_x'], v['grad_y'], v['min_val']
        for i in range(y2 - y1):
            for j in range(x2 - x1):
                yy, xx = y1 + i, x1 + j
                for c in range(3):
                    img[yy, xx, c] = gx[c]*j + gy[c]*i + base[c]
                block_map[yy, xx] = v

    imgX = img.copy()
    imgY = img.copy()

    for y in range(H):
        for x in range(W - 1):
            b1, b2 = block_map[y, x], block_map[y, x+1]
            close1 = b1['x2'] - b1['x1'] > patch//2
            close2 = b2['x2'] - b2['x1'] > patch//2
            if b1 and b2 and b1['id'] != b2['id']: #not np.allclose(b1['min_val'], b2['min_val']):
                for i in range(patch):
                    xx = x - patch//2 + i
                    xxs = max(x - patch//2,0)
                    xxe = min(x + patch//2, W-1)
                    if not close1 and xx <= x:
                        continue
                    if not close2 and xx > x:
                        continue
                    if 0 <= xx < W:
                        t = i / max(patch - 1, 1)
                        for c in range(3):
                            k = (1 / ( 1 + abs((b2['grad_x'][c] * i + b2['min_val'][c]) - (b1['grad_x'][c] * i + b1['min_val'][c])))) * 0.5 + 0.5
                            imgX[y, xx, c] = (1 - max(eta(t) * k, 1)) * img[y, xxs, c] + max(eta(t) * k, 1) * img[y, xxe, c]

    for x in range(W):
        for y in range(H - 1):
            b1, b2 = block_map[y, x], block_map[y+1, x]
            close1 = b1['y2'] - b1['y1'] > patch//2
            close2 = b2['y2'] - b2['y1'] > patch//2
            if b1 and b2 and b1['id'] != b2['id']: #not np.allclose(b1['min_val'], b2['min_val']):
                for i in range(patch):
                    yy = y - patch//2 + i
                    yys = max(y - patch//2,0)
                    yye = min(y + patch//2, H-1)
                    if not close1 and yy <= y:
                        continue
                    if not close2 and yy > y:
                        continue
                    if 0 <= yy < H:
                        t = i / max(patch - 1, 1)
                        for c in range(3):
                            k = (1 / ( 1 + abs((b2['grad_y'][c] * i + b2['min_val'][c]) - (b1['grad_y'][c] * i + b1['min_val'][c])))) * 0.5 + 0.5
                            imgY[yy, x, c] = (1 - max(eta(t) * k, 1)) * img[yys, x, c] + max(eta(t) * k, 1) * img[yye, x, c]
    
    for x in range(W):
        for y in range(H):
            for c in range(3):
                img[y,x,c] = round((imgX[y,x,c] + imgY[y,x,c]) / 2)
    return np.clip(img, 0, 255).astype(np.uint8)

def visualize_blocks(image_tensor, vectors):
    img = np.array(image_tensor.detach().to("cpu").tolist()).astype(np.uint8)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)
    for v in vectors:
        x1, y1, x2, y2 = v['x1'], v['y1'], v['x2'], v['y2']
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=1, edgecolor='cyan', facecolor='none')
        ax.add_patch(rect)
    ax.set_title("Block Decomposition")
    ax.axis("off")
    plt.tight_layout()
    plt.show()

def save_to_csv(vectors, fname="C:\\Users\\user\\Documents\\AI\\1. Python\\saseolimNet\\DBBD\\block_vectors1L2.csv"):
    rows = []
    for v in vectors:
        rows.append({
            "id" : v['id'],
            "x1": v['x1'], "y1": v['y1'], "x2": v['x2'], "y2": v['y2'], "depth": v['depth'],
            "gx_r": v['grad_x'][0], "gx_g": v['grad_x'][1], "gx_b": v['grad_x'][2],
            "gy_r": v['grad_y'][0], "gy_g": v['grad_y'][1], "gy_b": v['grad_y'][2],
            "min_r": v['min_val'][0], "min_g": v['min_val'][1], "min_b": v['min_val'][2],
        })
    pd.DataFrame(rows).to_csv(fname, index=False)
    print(f"CSV 저장 완료: {fname}")

def save_to_csv_meta(metablocks, fname="C:\\Users\\user\\Documents\\AI\\1. Python\\saseolimNet\\DBBD\\block_meta1L2.csv"):
    rows = []
    for id, x1, y1, x2, y2, d in metablocks:
        rows.append({
            "id" : id,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2, "depth": d
        })
    pd.DataFrame(rows).to_csv(fname, index=False)
    print(f"메타 CSV 저장 완료: {fname}")

def save_to_npz(vectors, path="npz1M_G_ACC_0.01.npz"):
    np.savez(path, block_vectors=np.array(vectors, dtype=object))
    print(f"npz 저장 완료: {path}")

# 실행부
if __name__ == "__main__":
    path = "1.jpeg"
    MaxDepth = math.ceil(math.log(Image.open(path).width * Image.open(path).height, 2))
    HalfLogDepth = math.ceil(math.log(Image.open(path).width * Image.open(path).height, 2 ** 2) * ( 2 ** (-4/5)))
    LogDepth = math.ceil(math.log(Image.open(path).width * Image.open(path).height, 2 ** 2) * ( 2 ** (-2/3))) #전체 분해 가능한 깊이의 약 63%만 허용함
    LogLogDepth = math.ceil(math.log(Image.open(path).width * Image.open(path).height, 2 ** 2) * ( 2 ** (-1/2))) #전체 분해 가능한 깊이의 약 xx%만 허용함
    PixelDepth = math.ceil(math.log(Image.open(path).width * Image.open(path).height, 2 ** 2)) - 3
    SmollLossDepth = PixelDepth + 1
    F = load_image(path)

    SetDepth = MaxDepth #Set
    print(f"SetDepth : {SetDepth}")
    print(f"ThisMaxDepth : {math.ceil(math.log(Image.open(path).width * Image.open(path).height, 2 ** 2))}")

    start_time = time.perf_counter()

    blocks, matablocks = adaptive_split(F, acc_threshold=0.01, min_size=1, max_depth=SetDepth) #표준 정확도는 0.98 #G의 표준 정확도는 1.00

    block_time = time.perf_counter()

    print(f"GPU에서 블록 분해까지 걸린 시간 : {block_time-start_time:.6f}초")

    vectors = compute_block_vectors(F, blocks)

    vector_time = time.perf_counter()

    print(f"CPU에서 벡터화까지 걸린 시간 : {vector_time-start_time:.6f}초")

    #save_to_csv(vectors)
    #save_to_csv_meta(matablocks)
    save_to_npz(vectors)

    Save_time = time.perf_counter()

    print(f"CPU에서 저장까지 걸린 시간 : {Save_time-start_time:.6f}초")

    recon = reconstruct(F.shape, vectors)
    cv2.imwrite("output.png", cv2.cvtColor(recon, cv2.COLOR_RGB2BGR))

    linearrecon = linearreconstruct(F.shape, vectors)
    cv2.imwrite("output_linear.png", cv2.cvtColor(linearrecon, cv2.COLOR_RGB2BGR))
    
    GUIR_time = time.perf_counter()

    print(f"CPU에서 시각화(일반)까지 걸린 시간 : {GUIR_time-start_time:.6f}초")

    smooth = lsjs_fill(F.shape, vectors)
    cv2.imwrite("output_LSJS.png", cv2.cvtColor(smooth, cv2.COLOR_RGB2BGR))
    #smooth = recon

    GUIL_time = time.perf_counter()

    print(f"CPU에서 시각화(LSJS)까지 걸린 시간 : {GUIL_time-start_time:.6f}초")
    visualize_blocks(F, vectors)

    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1); plt.imshow(np.array(F.detach().to("cpu").tolist()).astype(np.uint8)); plt.title("Original"); plt.axis("off")
    plt.subplot(1, 3, 2); plt.imshow(recon); plt.title("Reconstructed"); plt.axis("off")
    plt.subplot(1, 3, 3); plt.imshow(smooth); plt.title("LSJS Smoothed"); plt.axis("off")
    plt.tight_layout(); plt.show()
