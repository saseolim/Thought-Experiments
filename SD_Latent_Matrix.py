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

#K진수 트리 관련 함수에서 사용하는 유한합은 선형변환임으로 나중에 텐서로 바꿀 것

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

def ThisDepthCount(K,D):
    if D == 0:
        return(0,0)
    if D == 1:
        return(1,K)
    return (sum_series(lambda x: K ** x, 1, D-1) + 1, sum_series(lambda x: K **x,1,D))

def load_image(path):
    img = Image.open(path).convert("RGB")
    return torch.tensor(np.array(img), dtype=torch.float32).to(device)

def compute_axis_accuracy(block, axis):
    if axis == 0 and block.shape[0] > 1:
        diff = block[1:, :, :] - block[:-1, :, :]
    elif axis == 1 and block.shape[1] > 1:
        diff = block[:, 1:, :] - block[:, :-1, :]
    else:
        return torch.tensor(1.0, device=device)
    return 1.0 / (1.0 + torch.sum(diff ** 2))

def adaptive_split(F, max_depth : int, Inputstack : list, depth_lock = False, acc_threshold=0.98):
    H, W, _ = F.shape
    stack = Inputstack
    blocks = []

    while stack:
        id, x1, y1, x2, y2, depth = stack.pop()
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0 or depth >= max_depth:
            blocks.append((id, x1, y1, x2, y2, depth)) # 버그 발생시 이 부분 삭제 바람
            continue

        block = F[int(y1):int(y2), int(x1):int(x2), :]
        acc_x = compute_axis_accuracy(block, 1)
        acc_y = compute_axis_accuracy(block, 0)
        childId = ThisNodeChildLow(4,int(depth),int(id))

        if (acc_x < acc_threshold and acc_y < acc_threshold and w > 1 and h > 1) or (depth_lock and w > 1 and h > 1):
            mx, my = (x1 + x2) // 2, (y1 + y2) // 2
            stack.extend([
                (childId, x1, y1, mx, my, depth+1),
                (childId + 1, mx, y1, x2, my, depth+1),
                (childId + 2, x1, my, mx, y2, depth+1),
                (childId + 3, mx, my, x2, y2, depth+1),
            ])
        elif (acc_x < acc_threshold and w > 1) or (depth_lock and w > 1):
            mx = (x1 + x2) // 2
            stack.extend([(childId, x1, y1, mx, y2, depth+1), (childId + 1, mx, y1, x2, y2, depth+1)])
        elif (acc_y < acc_threshold and h > 1) or (depth_lock and h > 1):
            my = (y1 + y2) // 2
            stack.extend([(childId, x1, y1, x2, my, depth+1), (childId + 2, x1, my, x2, y2, depth+1)])
        else:
            blocks.append((id, x1, y1, x2, y2, depth))
    return blocks

def compute_block_vectors(F, blocks, block_vectors : torch.Tensor):
    vectors = block_vectors
    for id, x1, y1, x2, y2, d in blocks:
        region = F[int(y1):int(y2), int(x1):int(x2), :]
        h, w = region.shape[:2]
        #gx = torch.mean(region[:, 1:, :] - region[:, :-1, :], dim=(0, 1)) if w > 1 else torch.zeros(3, device=device)
        #gy = torch.mean(region[1:, :, :] - region[:-1, :, :], dim=(0, 1)) if h > 1 else torch.zeros(3, device=device)
        #base = torch.amin(region.reshape(-1, 3), dim=0)
        gx = (region[0, -1, :] - region[0, 0, :]) / (w - 1) if w > 1 else torch.zeros(3, device=device)
        gy = (region[-1, 0, :] - region[0, 0, :]) / (h - 1) if h > 1 else torch.zeros(3, device=device)
        base = region[0,0,:]
        Temp = torch.tensor([float(id), float(x1),float(y1),float(x2),float(y2),float(d), float(gx[0]), float(gx[1]), float(gx[2]), float(gy[0]), float(gy[1]), float(gy[2]), float(base[0]), float(base[1]), float(base[2])], dtype=torch.float32).to(device)
        Temp = Temp.unsqueeze(0)
        vectors = torch.cat((vectors, Temp), dim=0)
    return vectors

#mask = (ToSpace_block_vectors[:, 0] == id)
#if mask.any():
def GetGoToThatDepth(block_vectors : torch.Tensor, ThatDepth : int):
    Shape = [int(block_vectors[:, 4].max()),int(block_vectors[:,3].max()),3]
    recon = reconstruct(Shape,block_vectors.tolist())
    cv2.imwrite("Temp.png", cv2.cvtColor(recon, cv2.COLOR_RGB2BGR))
    F = load_image("Temp.png")
    for D in range(0,ThatDepth):
        start, end = ThisDepthCount(4,D)
        for id in range(start,end+1):
            mask = (block_vectors[:, 0] == id)
            if mask.any():
                print(id)
                v = block_vectors[mask][0]
                TempStack = [(v[0], v[1], v[2], v[3], v[4],v[5])]
                block_vectors = block_vectors[block_vectors[:, 0] != id] #block_vectors[~mask]
                TempBlock = adaptive_split(F,D+1,TempStack,depth_lock=True)
                block_vectors = compute_block_vectors(F, TempBlock, block_vectors)
    return block_vectors

def reconstruct(F_shape, vectors):
    H, W, _ = F_shape
    img = np.zeros((H, W, 3), dtype=np.float32)
    for v in vectors:
        x1, y1, x2, y2 = v['x1'], v['y1'], v['x2'], v['y2']
        gx, gy, base = v['grad_x'], v['grad_y'], v['min_val']
        for i in range(y2 - y1):
            for j in range(x2 - x1):
                for c in range(3):
                    img[y1+i, x1+j, c] = gx[c]*j + gy[c]*i + base[c]
    return np.clip(img, 0, 255).astype(np.uint8)

def reconstructToBigScale(F_shape, vectors, Scale : int):
    H, W, _ = F_shape
    img = np.zeros((H * Scale, W * Scale, 3), dtype=np.float32)
    for v in vectors:
        x1, y1, x2, y2 = v['x1'], v['y1'], v['x2'], v['y2']
        gx, gy, base = v['grad_x'], v['grad_y'], v['min_val']
        x1 *= Scale
        y1 *= Scale
        x2 *= Scale
        y2 *= Scale
        for i in range(y2 - y1):
            for j in range(x2 - x1):
                for c in range(3):
                    img[y1+i, x1+j, c] = (gx[c]/Scale)*j + (gy[c]/Scale)*i + base[c]
    return np.clip(img, 0, 255).astype(np.uint8)

def GetToBigScale(vectors : list, Scale : int):
    for i in range(vectors.__len__()):
        vectors[i]['x1'] *= Scale
        vectors[i]['y1'] *= Scale
        vectors[i]['x2'] *= Scale
        vectors[i]['y2'] *= Scale
        vectors[i]['grad_x'][0] /= Scale
        vectors[i]['grad_y'][0] /= Scale
        vectors[i]['grad_x'][1] /= Scale
        vectors[i]['grad_y'][1] /= Scale
        vectors[i]['grad_x'][2] /= Scale
        vectors[i]['grad_y'][2] /= Scale
    return vectors

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
                        #t = i / max(patch - 1, 1)
                        for c in range(3):
                            #k = (1 / ( 1 + abs((b2['grad_x'][c] * i + b2['min_val'][c]) - (b1['grad_x'][c] * i + b1['min_val'][c])))) * 0.5 + 0.5
                            #imgX[y, xx, c] = (1 - max(eta(t) * k, 1)) * img[y, xxs, c] + max(eta(t) * k, 1) * img[y, xxe, c]
                            k = 1 + (((b2['grad_x'][c] * i + b2['min_val'][c]) - (b1['grad_x'][c] * i + b1['min_val'][c]))/255) * 0.25
                            avg = patch/2
                            avgv = (img[y,xxe,c] - img[y,xxs,c])/2
                            grad = (img[y,xxe,c]-img[y,xxs,c])/max(patch - 1, 1)
                            high = max(img[y,xxe,c], img[y,xxs,c])
                            low = min(img[y,xxe,c], img[y,xxs,c])
                            imgX[y,xx,c] = max(low,min(grad * k * (i-avg) + avgv, high))


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
                        #t = i / max(patch - 1, 1)
                        for c in range(3):
                            #k = (1 / ( 1 + abs((b2['grad_y'][c] * i + b2['min_val'][c]) - (b1['grad_y'][c] * i + b1['min_val'][c])))) * 0.5 + 0.5
                            #imgY[yy, x, c] = (1 - max(eta(t) * k, 1)) * img[yys, x, c] + max(eta(t) * k, 1) * img[yye, x, c]
                            k = 1 + (((b2['grad_y'][c] * i + b2['min_val'][c]) - (b1['grad_y'][c] * i + b1['min_val'][c]))/255) * 0.25
                            avg = patch/2
                            avgv = (img[yye,x,c]-img[yys,x,c])/2
                            grad = (img[yye,x,c]-img[yys,x,c])/max(patch - 1, 1)
                            high = max(img[yye,x,c],img[yys,x,c])
                            low = min(img[yye,x,c],img[yys,x,c])
                            imgY[yy,x,c] = max(low,min(grad * k * (i-avg) + avgv, high))
    
    for x in range(W):
        for y in range(H):
            for c in range(3):
                img[y,x,c] = round((imgX[y,x,c] + imgY[y,x,c]) / 2)
    return np.clip(img, 0, 255).astype(np.uint8)
    
def load_block(path):
    data = np.load(path, allow_pickle=True)["block_vectors"]
    vectors = data.tolist()

    #id, x1, y1, x2, y2 ,|, depth, grad_x[0~2], grad_y[0] ,|, grad_y[1~2], min_val[0~2]
    Out = torch.tensor([[float(v['id']),
                           float(v['x1']), float(v['y1']),
                             float(v['x2']), float(v['y2']),
                               float(v['depth']),
                                 float(v['grad_x'][0]), float(v['grad_x'][1]), float(v['grad_x'][2]),
                                   float(v['grad_y'][0]), float(v['grad_y'][1]), float(v['grad_y'][2]),
                                     float(v['min_val'][0]), float(v['min_val'][1]), float(v['min_val'][2])] for v in vectors], dtype=torch.float32).to(device)
    #Shape = [int(Out[:, 4].max()),int(Out[:,3].max()),3]
    #recon = reconstruct(Shape,vectors)
    #cv2.imwrite("Temp.png", cv2.cvtColor(recon, cv2.COLOR_RGB2BGR))

    Scale = 8
    #recon = reconstructToBigScale(Shape, vectors, Scale)
    #cv2.imwrite(f"Temp_x{Scale}.png", cv2.cvtColor(recon, cv2.COLOR_RGB2BGR))

    ScaleVector = GetToBigScale(vectors=vectors, Scale=Scale)
    ShapeScale = [int(Out[:, 4].max()) * Scale,int(Out[:,3].max()) * Scale,3]
    smooth = []
    for i in range(1): #range(int(math.log2(Scale))):
        smooth.append(lsjs_fill(ShapeScale, ScaleVector, patch=max(int(round(Scale / 2 ** i)), 1)).astype(np.float32))
        print(f'{int(round(Scale / 2 ** i))} end')
    outsmooth = np.zeros((ShapeScale[0], ShapeScale[1], 3), dtype=np.float32)
    for s in smooth:
        outsmooth += s
    outsmooth /= smooth.__len__()
    outsmooth = np.clip(outsmooth, 0, 255).astype(np.uint8)
    cv2.imwrite(f"Temp_x{Scale}_LSJS.png", cv2.cvtColor(outsmooth, cv2.COLOR_RGB2BGR))
    return Out

def GetToSpace_block_vectors(block_vectors : torch.Tensor):
    a = 100 / block_vectors[:,3].max().item()
    for i in range(0, 1):
        block_vectors[:, 1] *= a
        block_vectors[:, 2] *= a
        block_vectors[:, 3] *= a
        block_vectors[:, 4] *= a
        block_vectors[:, 6] /= a
        block_vectors[:, 7] /= a
        block_vectors[:, 8] /= a
        block_vectors[:, 9] /= a
        block_vectors[:, 10] /= a
        block_vectors[:, 11] /= a
    return block_vectors

def Latent_Vector(block_vectors: torch.Tensor, EndDepth = 1):
    max_depth = int(block_vectors[:, 5].max().item())
    max_id = int(block_vectors[:, 0].max().item())
    Temp = torch.zeros((max_id + 1, 2), dtype=torch.float32).to(device)

    # 아래서 위로 올라가며 평균을 계산
    for D in reversed(range(EndDepth, max_depth + 1)):
        print(f"Depth : {D}")
        start, end = ThisDepthCount(4, D)
        for id in range(start, min(end + 1, max_id + 1)):
            # 블록 벡터 존재할 경우
            mask = (block_vectors[:, 0] == id)
            if mask.any():
                avg_val = block_vectors[mask].mean().item()
                parent = ThisNodeParent(4, D, id)
                #print(parent)
                Temp[id, 0] += avg_val
                Temp[id, 1] += 1
                Temp[parent, 0] += avg_val
                Temp[parent, 1] += 1
            # 하위 노드에서 온 값 활용
            elif Temp[id, 1] > 0:
                parent = ThisNodeParent(4, D, id)
                #print(parent)
                Temp[parent, 0] += Temp[id, 0]
                Temp[parent, 1] += 1

        for v in Temp[:,1]:
            if int(v) > 4:
                print(f"PP{int(v)}")
        # 평균화 (이전 단계)
        if D > 1:
            p_start, p_end = ThisDepthCount(4, D - 1)
            for pid in range(p_start, min(p_end, max_id + 1)):
                if Temp[pid, 1] > 0:
                    Temp[pid, 0] /= Temp[pid, 1]
                    Temp[pid, 1] = 1

    return Temp

def Latent_Vector_Cut(Latent_Vector : torch.Tensor, ThisDepth : int):
    return Latent_Vector[sum_series(lambda x: 4 ** x,1, ThisDepth - 1) + 1:sum_series(lambda x: 4 ** x,1,ThisDepth) + 1,0]

def Latent_Vector_Cut_L(Latent_Vector : torch.Tensor, Start : int, End : int):
    return Latent_Vector[sum_series(lambda x: 4 ** x,1, Start - 1) + 1:sum_series(lambda x: 4 ** x,1,End) + 1,0]

def simmPrint(Latent_vector1 : torch.Tensor, Latent_vector2 : torch.Tensor, ThisDepth : int):
    print(f'This Depth is {ThisDepth}. simm is next line to get')
    ll = Latent_Vector_Cut(Latent_vector1, ThisDepth)
    lll = Latent_Vector_Cut(Latent_vector2,ThisDepth)
    sim = torch.nn.functional.cosine_similarity(ll, lll, dim=0)
    print(f'cos simm : {sim.item()}')
    lln = ll / ll.norm(p=2)
    llln = lll / lll.norm(p=2)
    norm = torch.norm(llln - lln, p=2)
    print(f'norm simm : {norm.item()}')
    return sim, norm

def simmPrint_L(Latent_vector1 : torch.Tensor, Latent_vector2 : torch.Tensor, Start : int, End : int):
    print(f'This Depth is {Start} ~ {End}. simm is next line to get')
    ll = []
    lll = []
    a = 0
    for d in range(Start, End + 1):
        a += 1 / 4 ** d
    for d in range(Start, End + 1):
        ll.append(Latent_Vector_Cut(Latent_vector1, d) * (1/4**d)/a)
        lll.append(Latent_Vector_Cut(Latent_vector2, d) * (1/4**d)/a)
    ll = torch.cat(ll,dim=-1).to(device)
    lll = torch.cat(lll,dim=-1).to(device)
    #ll = Latent_Vector_Cut_L(Latent_vector1, Start,End)
    #lll = Latent_Vector_Cut_L(Latent_vector2,Start,End)
    sim = torch.nn.functional.cosine_similarity(ll, lll, dim=0)
    print(f'cos simm : {sim.item()}')
    lln = ll / ll.norm(p=2)
    llln = lll / lll.norm(p=2)
    norm = torch.norm(llln - lln, p=2)
    print(f'norm simm : {norm.item()}')
    return sim, norm


def InputVector(block_vectors : torch.Tensor):
    Temp = torch.zeros((1,block_vectors.shape[0]),dtype=torch.float32).to(device)
    Shape = torch.tensor([block_vectors1[:, 3].max(),block_vectors1[:,4].max(),3], dtype=torch.float32).to(device)
    for i in range(0, int(block_vectors.shape[0])):
        Temp[0,i] = 1 #/ (1 + block_vectors[i,5])
        block_vectors[i,0] = block_vectors[i,0].log10() #id log화
    print(Temp.flatten().sum())
    Out = (Temp @ block_vectors).flatten() / Temp.flatten().sum()
    a = 1 #* block_vectors.shape[0] ** (2.5/4)
    print(a)
    Out *= torch.tensor([1,100/Shape[0],100/Shape[0],100/Shape[0],100/Shape[0],1,a,a,a,a,a,a,1,1,1],dtype=torch.float32).to(device)
    Out = Out / (Out.norm(p=2) + 1e-8)
    return Out

if __name__ == "__main__":
    block_vectors1 = load_block('npz1_D9_G.npz')
    block_vectors2 = load_block('npz1M_G.npz')
    print('f')

    max_id = block_vectors1[:, 0].max()
    max_depth = block_vectors1[:, 5].max()
    Shape = torch.tensor([block_vectors1[:, 3].max(),block_vectors1[:,4].max(),3], dtype=torch.float32).to(device)
    id = max_id

    SetDepth = torch.tensor([3, 4, 5], dtype=torch.int).to(device)

    mask = (block_vectors1[:, 5] <= SetDepth.max().item() - 1)
    Count = 0
    for v in block_vectors1[mask]:
        Count += 1
        print(v[5])
    print(f'{Count} ~ 1v end')
    mask = (block_vectors2[:, 5] <= SetDepth.max().item() - 1)
    Count = 0
    for v in block_vectors2[mask]:
        Count += 1
        print(v[5])
    print(f'{Count} ~ 2v end')

    block_vectors1 = GetGoToThatDepth(block_vectors1,SetDepth.max().item())
    print('1end')
    block_vectors2 = GetGoToThatDepth(block_vectors2,SetDepth.max().item())
    print('2end')

    block_vectors1 = GetToSpace_block_vectors(block_vectors1)
    block_vectors2 = GetToSpace_block_vectors(block_vectors2)
    print('ToSpace end')

    ll = Latent_Vector(block_vectors1, SetDepth.min().item())
    print(ll)
    lll = Latent_Vector(block_vectors2, SetDepth.min().item())
    print(lll)
    Count = 0
    simSum = []
    normSum = []
    for d in SetDepth:
        d = d.item()
        sim, norm = simmPrint(ll,lll,d)
        Count += 1
        simSum.append(sim)
        normSum.append(norm)
    Countaa = 0
    for d in SetDepth:
        Countaa += 1 / 4 ** d

    print(f'All Avg simm -> cos sim : {torch.tensor(simSum, dtype=torch.float32).sum().item() / Count} , norm sim : {torch.tensor(normSum, dtype=torch.float32).sum().item() / Count}')

    simSumaa = 0
    normSumaa = 0
    for i in range(0, SetDepth.shape[0]):
        simSumaa += simSum[i] * ((1 / 4 ** SetDepth[i]) / Countaa)
        normSumaa += normSum[i] * ((1 / 4 ** SetDepth[i]) / Countaa)

    print(f'All PSV simm -> cos sim : {simSumaa} , norm sim : {normSumaa}')

    simp, normp = simmPrint_L(ll,lll,SetDepth.min().item(),SetDepth.max().item())

    #Latent_Depth = 5
    #Latent_vector1 = Latent_Vector(block_vectors1, Latent_Depth)
    #print(Latent_vector1)
