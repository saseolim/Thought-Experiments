import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import time
from collections import defaultdict

# --- 0. 기본 설정 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
#torch.set_default_dtype(torch.float64)


# --- 1. GPU-Native F-분해 함수들 ---
def extract_blocks_vectorized(F_gpu, coords):
    """좌표 배치를 사용하여 원본 이미지에서 모든 블록을 단일 연산으로 추출합니다."""
    B = coords.shape[0]
    H_block, W_block = int(coords[0, 3] - coords[0, 1]), int(coords[0, 2] - coords[0, 0])
    
    if H_block <= 0 or W_block <= 0:
        return torch.empty(B, 0, 0, F_gpu.shape[2], device=DEVICE)

    iy = torch.arange(H_block, device=DEVICE).view(1, H_block, 1)
    ix = torch.arange(W_block, device=DEVICE).view(1, 1, W_block)

    x1s, y1s = coords[:, 0].view(B, 1, 1), coords[:, 1].view(B, 1, 1)
    abs_ix, abs_iy = x1s + ix, y1s + iy

    return F_gpu[abs_iy, abs_ix]

def calculate_accuracy_batched_gpu(blocks_tensor):
    """
    로직 추가 예고 : 지금은 xy 4분활만 가능하지만 x,y축 독립적으로 2분할하고 id가 같으면 x,y축이면 더 정확도가 낮은 쪽으로,
    id가 같으면서 x,xy나 y,xy라면 xy로 분해하고 분해대상이 아닌 블록은 삭제한다.
    그리고 추가로 블록이 무조건 같은 크기로 나오는 것이 아니니 블록이 나올 수 있는 모든 경우의 수를 CPU 멀티스래딩으로 처리하도록 한다.
    그리고 또한 extract_blocks_vectorized에서 실제와 분해 크기와 맞지 않는 치명적인 버그가 존제(이 버그를 고치지 않으면 정확도 1일시 MSE가 0이 나오는게 아니라 39~100정도가 나오는 문제가 있다)하는데 GPU연산의 일관성을 위하여 가장 큰 블록 크기로 텐서를 만들되
    슬라이싱 할 shape를 저장하는 것을 포함하는 strcut {Tensor 블록 텐서 집합, Tensor 블록 슬라이스 시종점}으로 저장하는 것을 고려한다. 즉 만약 블록 텐서 집합이 Blocks = Tensor[Batch, MaxH, MaxW, C]인데 Batch = 3에서 h < MaxH, w < MaxW일시
    슬라이싱할 시종점을 상대좌표로 Cut = Tensor[Batch, 4] => [Batch, {x1, y1, x2, y2}]로 저장한다. 그럼으로 예시에서 Batch = 3에서 calculate_accuracy_batched_gpu에 넣는 블록은 Blocks[3, Cut[3, 1]:Cut[3, 3], Cut[3, 0]:Cut[3, 2], :]으로 되는 것이다.
    그런가 하면 Blocks에서 MaxH, MaxW를 제대로 측정하고 빈틈없이 슬라이싱시 블록이 없는 경우가 없으면서도 안 겹치도록 만드는게 매우 중요할 것이다. 그리고 Blocks는 처음부터 Tensor[Batch, MaxH, MaxW, C]인 ones인 0텐서로 만들고 하나하나씩 대입하여 padding작업을 미연에 방지한다.

    로직은 일단 4분할에서 분해 크기와 맞지 않는 치명적인 버그를 고친 후 검증이 끝난 다음에 2분할도 추가하도록 한다.
    """
    B, H, W, C = blocks_tensor.shape
    if W <= 0 or H <= 0:
        return torch.ones(B, 3, device=DEVICE), torch.zeros(B, C, device=DEVICE), torch.zeros(B, C, device=DEVICE), blocks_tensor[:, 0, 0, :]
    
    blocks_tensor.to(torch.float32)

    params_gx = (blocks_tensor[:, 0, -1, :] - blocks_tensor[:, 0, 0, :]) / (W - 1) if W > 1 else torch.ones(B, 3, device=DEVICE) #(region[0, -1, :] - region[0, 0, :]) / (w - 1) if w > 1 else torch.zeros(3, device=device)
    params_gy = (blocks_tensor[:, -1, 0, :] - blocks_tensor[:, 0, 0, :]) / (H - 1) if H > 1 else torch.ones(B, 3, device=DEVICE)
    params_base = blocks_tensor[:, 0, 0, :]

    params_gx = torch.nan_to_num(params_gx, nan=0.0)
    params_gy = torch.nan_to_num(params_gy, nan=0.0)

    #if (params_gx.sum(dim=(0,1)) * W).abs() < 3 : print(params_gx.sum(dim=1) * W)
    #if (params_gy.sum(dim=(0,1)) * H).abs() < 3 : print(params_gy.sum(dim=1) * H)
    
    ii = torch.arange(H, device=DEVICE).view(1, H, 1, 1).float()
    jj = torch.arange(W, device=DEVICE).view(1, 1, W, 1).float()
    G = params_gx.view(B, 1, 1, C) * jj + ii * params_gy.view(B, 1, 1, C) + params_base.view(B, 1, 1, C)

    grad_F_x = blocks_tensor[:, :, 1:, :] - blocks_tensor[:, :, :-1, :]
    grad_F_y = blocks_tensor[:, 1:, :, :] - blocks_tensor[:, :-1, :, :]
    grad_G_x = G[:, :, 1:, :] - G[:, :, :-1, :]
    grad_G_y = G[:, 1:, :, :] - G[:, :-1, :, :]

    grad_F_x = torch.nn.functional.pad(grad_F_x, (0, 0, 0, 1, 0, 0))
    grad_F_y = torch.nn.functional.pad(grad_F_y, (0, 0, 0, 0, 0, 1))
    grad_G_x = torch.nn.functional.pad(grad_G_x, (0, 0, 0, 1, 0, 0))
    grad_G_y = torch.nn.functional.pad(grad_G_y, (0, 0, 0, 0, 0, 1))

    #min_h = min(grad_F_x.shape[1], grad_F_y.shape[1])
    #min_w = min(grad_F_x.shape[2], grad_F_y.shape[2])

    #grad_f = torch.stack((grad_F_x[:,:min_h, :min_w], grad_F_y[:,:min_h, :min_w]), dim=0)
    #grad_g = torch.stack((grad_G_x[:,:min_h, :min_w], grad_G_y[:,:min_h, :min_w]), dim=0)

    grad_f = torch.stack((grad_F_x, grad_F_y), dim=0)
    grad_g = torch.stack((grad_G_x, grad_G_y), dim=0)

    diff = grad_f - grad_g
    diff_squared = (diff ** 2).sum(dim=0).sqrt().sum(dim=(1,2))
    #int_total = blocks_tensor.sum(dim=(1,2)).sum(dim=1) - G.sum(dim=(1,2)).sum(dim=1)
    total_error = diff_squared.sum(dim=1)
    
    error_x = ((grad_F_x - grad_G_x) ** 2).sqrt().sum(dim=(1,2)).sum(dim=1)
    error_y = ((grad_F_y - grad_G_y) ** 2).sqrt().sum(dim=(1,2)).sum(dim=1)
    
    acc_total = 1.0 / (1.0 + total_error) 
    #acc_total = 1.0 / (1.0 + torch.sum(torch.stack([error_x, error_y], dim=0) ** 2,dim=0).sqrt()) #(1.0 + error_x + error_y)
    accuracies = torch.stack([acc_total, 1.0 / (1.0 + error_x), 1.0 / (1.0 + error_y)], dim=1)
    
    return accuracies, params_gx, params_gy, params_base

def adaptive_decomposition_gpu_native(F, acc_threshold, max_depth=10, K_side=2):
    """
    (수정) 크기별 그룹화를 통해 부정합 크기 버그를 해결하고,
    x/y축 독립 2분할 기능을 포함한 GPU-Native F-분해
    """
    print("Starting GPU-native decomposition (Grouped by size)...")
    H, W, C = F.shape
    F_gpu = F.to(DEVICE)
    
    all_nodes = []
    # 큐의 구조: [id, parent_id, x1, y1, x2, y2, depth]
    queue = torch.tensor([[0, -1, 0, 0, W, H, 0]], dtype=torch.long, device=DEVICE)
    id_counter = 1

    while queue.shape[0] > 0:
        current_depth = queue[0, 6].item()
        
        # 최대 깊이에 도달하면 현재 큐의 모든 노드를 말단 노드로 처리하고 종료
        if current_depth >= max_depth:
            stop_nodes = queue
            if stop_nodes.shape[0] > 0:
                # 이 말단 노드들도 정확도 계산을 위해 그룹화 처리
                coords = stop_nodes[:, 2:6]
                block_sizes = torch.stack([coords[:, 2] - coords[:, 0], coords[:, 3] - coords[:, 1]], dim=1)
                unique_sizes, inverse_indices = torch.unique(block_sizes, dim=0, return_inverse=True)
                
                for i in range(unique_sizes.shape[0]):
                    mask = (inverse_indices == i)
                    current_coords = coords[mask]
                    current_stop_nodes = stop_nodes[mask]

                    if current_coords.shape[0] == 0: continue
                    blocks_tensor = extract_blocks_vectorized(F_gpu, current_coords)
                    accuracies, gxs, gys, bases = calculate_accuracy_batched_gpu(blocks_tensor)

                    for j in range(current_stop_nodes.shape[0]):
                        node_info = current_stop_nodes[j]
                        all_nodes.append([
                            node_info[0].item(), node_info[1].item(),
                            *node_info[2:6].cpu().tolist(), node_info[6].item(),
                            *gxs[j].cpu().tolist(), *gys[j].cpu().tolist(), *bases[j].cpu().tolist(),
                            *accuracies[j].cpu().tolist(), 1.0
                        ])
            break # 메인 루프 종료
        
        print(f"Processing Depth {current_depth}, Num blocks in queue: {queue.shape[0]}")
        
        # --- 핵심 수정: 큐에 있는 블록들을 크기별로 그룹화 ---
        coords = queue[:, 2:6]
        block_sizes = torch.stack([coords[:, 2] - coords[:, 0], coords[:, 3] - coords[:, 1]], dim=1)
        unique_sizes, inverse_indices = torch.unique(block_sizes, dim=0, return_inverse=True)
        
        next_child_coords_list = []
        next_child_parent_ids = []

        # 크기가 같은 블록들끼리 묶어서 처리
        for i in range(unique_sizes.shape[0]):
            mask = (inverse_indices == i)
            current_coords = coords[mask]
            current_queue_info = queue[mask]

            if current_coords.shape[0] == 0: continue
            
            # 이 그룹 내에서는 모든 블록 크기가 같으므로, extract/calculate 함수가 안전하게 작동
            blocks_tensor = extract_blocks_vectorized(F_gpu, current_coords)
            accuracies, gxs, gys, bases = calculate_accuracy_batched_gpu(blocks_tensor)
            
            # 분할/멈춤 조건 결정 (2/4분할 로직)
            stop_mask = current_depth >= max_depth | (current_coords[:, 2] - current_coords[:, 0] < K_side) | (current_coords[:, 3] - current_coords[:, 1] < K_side)

            split_xy_mask = ~stop_mask & ((accuracies[:, 0] < acc_threshold) | ((accuracies[:, 1] < acc_threshold) & (accuracies[:, 2] < acc_threshold)))
            split_x_mask = ~stop_mask & ~split_xy_mask & (accuracies[:, 1] < acc_threshold)
            split_y_mask = ~stop_mask & ~split_xy_mask & ~split_x_mask & (accuracies[:, 2] < acc_threshold)
            needs_split_mask = split_xy_mask | split_x_mask | split_y_mask

            stop_mask = ~needs_split_mask | stop_mask

            # 분할되는 노드(중간 노드) 정보 저장
            combined_split_mask = split_xy_mask | split_x_mask | split_y_mask
            if torch.any(combined_split_mask):
                split_nodes = current_queue_info[combined_split_mask]
                split_accs, split_gxs, split_gys, split_bases = accuracies[combined_split_mask], gxs[combined_split_mask], gys[combined_split_mask], bases[combined_split_mask]
                for j in range(split_nodes.shape[0]):
                    node_info = split_nodes[j]
                    all_nodes.append([
                        node_info[0].item(), node_info[1].item(),
                        *node_info[2:6].cpu().tolist(), node_info[6].item(),
                        *split_gxs[j].cpu().tolist(), *split_gys[j].cpu().tolist(), *split_bases[j].cpu().tolist(),
                        *split_accs[j].cpu().tolist(), 0.0
                    ])

            # 분할 타입에 따라 자식 노드 생성
            for split_mask, split_type in zip([split_xy_mask, split_x_mask, split_y_mask], ['xy', 'x', 'y']):
                if torch.any(split_mask):
                    nodes_to_split = current_queue_info[split_mask]
                    for j in range(nodes_to_split.shape[0]):
                        parent_id, _, x1, y1, x2, y2, d = nodes_to_split[j]
                        w, h = x2 - x1, y2 - y1
                        
                        children_coords = []
                        if split_type == 'xy':
                            mx, my = x1 + w // K_side, y1 + h // K_side
                            children_coords = [[x1, y1, mx, my], [mx, y1, x2, my], [x1, my, mx, y2], [mx, my, x2, y2]]
                        elif split_type == 'x':
                            mx = x1 + w // K_side
                            children_coords = [[x1, y1, mx, y2], [mx, y1, x2, y2]]
                        elif split_type == 'y':
                            my = y1 + h // K_side
                            children_coords = [[x1, y1, x2, my], [x1, my, x2, y2]]
                        
                        valid_children = [c for c in children_coords if c[2] > c[0] and c[3] > c[1]]
                        next_child_coords_list.extend(valid_children)
                        next_child_parent_ids.extend([parent_id.item()] * len(valid_children))

            # 멈출 노드(말단 노드) 정보 저장
            if torch.any(stop_mask):
                stop_nodes = current_queue_info[stop_mask]
                stop_accs, stop_gxs, stop_gys, stop_bases = accuracies[stop_mask], gxs[stop_mask], gys[stop_mask], bases[stop_mask]
                for j in range(stop_nodes.shape[0]):
                    node_info = stop_nodes[j]
                    all_nodes.append([
                        node_info[0].item(), node_info[1].item(),
                        *node_info[2:6].cpu().tolist(), node_info[6].item(),
                        *stop_gxs[j].cpu().tolist(), *stop_gys[j].cpu().tolist(), *stop_bases[j].cpu().tolist(),
                        *stop_accs[j].cpu().tolist(), 1.0
                    ])

        # 모든 그룹 처리가 끝난 후, 다음 세대를 위한 큐를 한 번에 생성
        if not next_child_coords_list:
            queue = torch.empty((0, 7), dtype=torch.long, device=DEVICE)
        else:
            num_children = len(next_child_coords_list)
            child_ids = torch.arange(id_counter, id_counter + num_children, device=DEVICE).view(-1, 1)
            id_counter += num_children
            
            child_parent_ids_tensor = torch.tensor(next_child_parent_ids, dtype=torch.long, device=DEVICE).view(-1, 1)
            new_coords_tensor = torch.tensor(next_child_coords_list, dtype=torch.long, device=DEVICE)
            new_depths_tensor = torch.full((num_children, 1), current_depth + 1, dtype=torch.long, device=DEVICE)
            
            queue = torch.cat([child_ids, child_parent_ids_tensor, new_coords_tensor, new_depths_tensor], dim=1)

    print(f"Decomposition finished. Found {len(all_nodes)} total nodes.")
    if not all_nodes: 
        return torch.empty(0, 20)
    
    all_nodes.sort(key=lambda b: b[0])
    return torch.tensor(all_nodes, dtype=torch.float32)

def reconstruct_tree_from_full_nodes(all_nodes_tensor):
    print("\nReconstructing tree structure from all nodes...")
    if all_nodes_tensor.shape[0] == 0: return [], {}
    num_nodes = all_nodes_tensor.shape[0]
    id_to_idx = {int(node[0].item()): i for i, node in enumerate(all_nodes_tensor)}
    adj, parent_of = [[] for _ in range(num_nodes)], {}
    for i, node in enumerate(all_nodes_tensor):
        parent_id = int(node[1].item())
        if parent_id != -1 and parent_id in id_to_idx:
            parent_idx = id_to_idx[parent_id]
            adj[parent_idx].append(i)
            parent_of[i] = parent_idx
    print("Tree reconstruction complete.")
    return adj, parent_of

def reconstruct_from_blocks(F_shape, all_nodes_tensor):
    """
    (최적화 버전) 크기별 그룹화를 통해 이미지 복원 속도를 향상시킨 함수
    """
    H, W, C = F_shape
    reconstructed_image = torch.zeros((H, W, C), device=DEVICE, dtype=torch.float32)
    
    # CPU에서 텐서 조작 후 GPU로 보내는 것이 효율적일 수 있음
    nodes_cpu = all_nodes_tensor
    leaf_nodes = nodes_cpu[nodes_cpu[:, -1] == 1.0]

    if leaf_nodes.shape[0] == 0:
        return reconstructed_image

    # 말단 노드의 좌표와 크기 계산
    coords = leaf_nodes[:, 2:6].long()
    block_sizes = torch.stack([coords[:, 3] - coords[:, 1], coords[:, 2] - coords[:, 0]], dim=1) # H, W 순서
    
    # 고유한 블록 크기 찾기
    unique_sizes, inverse_indices = torch.unique(block_sizes, dim=0, return_inverse=True)

    print(f"\nReconstructing image with {leaf_nodes.shape[0]} leaf nodes, found {unique_sizes.shape[0]} unique block sizes.")

    # 고유한 크기별로 그룹화하여 처리
    for i in range(unique_sizes.shape[0]):
        h, w = unique_sizes[i]
        
        # 현재 크기에 해당하는 블록들의 인덱스 마스크 생성
        mask = (inverse_indices == i)
        
        # 현재 그룹의 노드 정보 추출
        current_nodes = leaf_nodes[mask].to(DEVICE)
        current_coords = current_nodes[:, 2:6].long()
        
        num_blocks_in_group = current_nodes.shape[0]

        # 파라미터 추출 (gx, gy, base)
        gxs = current_nodes[:, 7:10]
        gys = current_nodes[:, 10:13]
        bases = current_nodes[:, 13:16]
        
        # --- GPU 병렬 연산 시작 ---
        # ii, jj 좌표 그리드 생성 (이 그룹의 모든 블록에 재사용됨)
        ii = torch.arange(h, device=DEVICE).view(1, h, 1, 1).float()
        jj = torch.arange(w, device=DEVICE).view(1, 1, w, 1).float()
        
        # 브로드캐스팅을 이용해 B개의 G 블록을 한 번에 생성
        # B, 1, 1, C 형태로 reshape하여 브로드캐스팅 준비
        gxs_b = gxs.view(num_blocks_in_group, 1, 1, C)
        gys_b = gys.view(num_blocks_in_group, 1, 1, C)
        bases_b = bases.view(num_blocks_in_group, 1, 1, C)
        
        # G_batch의 shape: (B, h, w, C)
        G_batch = gxs_b * jj + gys_b * ii + bases_b
        # --- GPU 병렬 연산 종료 ---
        
        # 생성된 G_batch를 원본 이미지의 올바른 위치에 복사
        # 이 부분은 순차적으로 처리해야 하지만, 가장 무거운 계산은 이미 끝났음
        for k in range(num_blocks_in_group):
            x1, y1, x2, y2 = current_coords[k]
            reconstructed_image[y1:y2, x1:x2, :] = G_batch[k]
            
    return reconstructed_image

def save_results_to_npz(all_nodes_tensor, F_shape, adj, parent_of, path="output.npz"):
    print(f"Saving all node results to {path}...")
    np.savez_compressed(path, all_nodes=all_nodes_tensor.cpu().numpy(), original_shape=np.array(F_shape), adj_vector=np.array(adj, dtype=object), parent=np.array(parent_of))
    print("Save complete.")

def visualize_from_saved(image_path, save_path):
    print(f"\n--- Loading data from {save_path} for visualization ---")
    loaded_data = np.load(save_path)
    all_nodes_tensor = torch.tensor(loaded_data['all_nodes'], dtype=torch.float32)
    original_shape = loaded_data['original_shape']
    leaf_nodes = all_nodes_tensor[all_nodes_tensor[:, -1] == 1.0]
    print(f"Data loaded. Total nodes: {len(all_nodes_tensor)}, Leaf nodes: {len(leaf_nodes)}")
    with Image.open(image_path) as img:
        original_img = torch.tensor(np.array(img.convert("RGB")), dtype=torch.float32).to(DEVICE)
    print("Reconstructing image...")
    reconstructed_image = reconstruct_from_blocks(tuple(original_shape), all_nodes_tensor)
    original_np, reconstructed_np = original_img.cpu().numpy().astype(np.uint8), reconstructed_image.cpu().numpy().clip(0, 255).astype(np.uint8)
    MSE = torch.nn.functional.mse_loss(torch.Tensor(reconstructed_np),torch.Tensor(original_np))
    print(f"MSE : {MSE :.5f}")
    print(f"RMSE : {math.sqrt(MSE):.5f}[px] | pixel diff percent : {(math.sqrt(MSE)/255)*100:.5f}[%] | pixel accuracy : {(1 - math.sqrt(MSE)/255)*100:.5f}[%]")
    fig, axes = plt.subplots(1, 3, figsize=(24, 8)) #(18, 6)
    axes[0].imshow(original_np); axes[0].set_title("Original")
    axes[1].imshow(original_np); axes[1].set_title(f"Block Decomposition ({len(leaf_nodes)} leaf nodes)")
    for block in leaf_nodes:
        _, _, x1, y1, x2, y2, *_ = block
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='cyan', facecolor='none')
        axes[1].add_patch(rect)
    axes[2].imshow(reconstructed_np); axes[2].set_title("Reconstructed from Saved Data")
    for ax in axes: ax.axis("off")
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    IMAGE_PATH = "2.png" 
    SAVE_PATH = "2M_ALL_NODES_REFACTORED_ACC_1_00_00.npz"
    ACCURACY_THRESHOLD = 1
    K_SIDE = 2 
    
    try:
        with Image.open(IMAGE_PATH) as img:
            F_shape, F = (*img.size[::-1], 3), torch.tensor(np.array(img.convert("RGB")), dtype=torch.float32)
            MAX_DEPTH = math.ceil(math.log(F_shape[0] * F_shape[1], 2 ** 2))
            print(f"Image Loaded: {IMAGE_PATH}, Shape: {F_shape}, Max Depth: {MAX_DEPTH}")
    except FileNotFoundError:
        print(f"Error: Image file not found at {IMAGE_PATH}. Using a dummy image.")
        F_shape, F = (512, 512, 3), torch.randint(0, 256, (512, 512, 3), dtype=torch.float32)
        MAX_DEPTH = int(math.log(max(F_shape[:2]), K_SIDE))

    start_time = time.perf_counter()
    all_nodes_tensor = adaptive_decomposition_gpu_native(F, ACCURACY_THRESHOLD, MAX_DEPTH, K_SIDE)
    end_time = time.perf_counter()
    print(f"\nDecomposition Time: {end_time - start_time:.4f} seconds")

    if all_nodes_tensor.shape[0] > 0:
        adj, parent_of = reconstruct_tree_from_full_nodes(all_nodes_tensor)
        if adj and all_nodes_tensor.shape[0] > 1:
            print("\n--- Sample of Reconstructed Tree ---")
            root_idx = 0
            print(f"Root node (index {root_idx}, id {int(all_nodes_tensor[root_idx, 0])}) has {len(adj[root_idx])} children.")
            print(f"Children indices: {adj[root_idx]}")
        save_results_to_npz(all_nodes_tensor, F_shape, adj, parent_of, SAVE_PATH)
        visualize_from_saved(IMAGE_PATH, SAVE_PATH)
    else:
      print("No nodes were generated.")

        #id, 부모 주소, x1, y1, x2, y2 ,|
        # , depth, grad_x[0~2], grad_y[0] ,|
        # , grad_y[1~2], min_val[0~2] ,|
        # , 전체 정확도, x축 정확도, y축 정확도, 말단 플레그