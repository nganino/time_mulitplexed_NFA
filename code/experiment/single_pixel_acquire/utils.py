import numpy as np
import random
import torch 
import torch.nn.functional as F
import os
import time
import matplotlib.pyplot as plt
from scipy.signal import correlate2d
# import HEDS
# from hedslib.heds_types import *
import glob
import datetime
from holoeye import slmdisplaysdk
from pypylon import pylon

def init_slm():
    # slm = slmdisplaysdk.SLMDisplaySDK()
    slm = slmdisplaysdk.SLMInstance()
    # Check if the library implements the required version
    if not slm.requiresVersion(5):
        exit(1)

    # Detect SLMs and open a window on the selected SLM
    error = slm.open()
    assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)

    # slm.set_image_directory(os.path.join(os.path.dirname(__file__), 'images'))
    return slm

# def init_multiple_slm():

#     HEDS.SDK.PrintVersion()
#     err = HEDS.SDK.Init(4,1)
#     assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
#     slmFirst = HEDS.SLM.Init("-slm index:0 -nogui")
#     assert slmFirst.errorCode() == HEDSERR_NoError, HEDS.SDK.ErrorString(slmFirst.errorCode())
#     slmSecond = HEDS.SLM.Init("-slm index:1 -nogui")
#     assert slmSecond.errorCode() == HEDSERR_NoError, HEDS.SDK.ErrorString(slmSecond.errorCode())

#     return slmFirst, slmSecond

def init_cam():
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()
    return camera

def set_config_cam(camera, trigger_mode='Software', exposure_us=None, fps=None, gain=None, width=None, height=None, offset_x=None, offset_y=None):
    """
    Set camera exposure and ROI (region of interest) settings.

    Parameters:
        camera: Basler InstantCamera object (already Opened)
        exposure_us: Exposure time in microseconds
        width: Width of image (must be multiple of increment step)
        height: Height of image
        offset_x: X offset (starting column)
        offset_y: Y offset (starting row)
    """

    # Make sure camera is not grabbing when setting parameters
    if camera.IsGrabbing():
        camera.StopGrabbing()

    camera.TriggerSelector.SetValue("FrameStart")

    if trigger_mode == "Software":
        camera.TriggerMode.SetValue("On")
        camera.TriggerSource.SetValue("Software")
    elif trigger_mode == "FreeRun":
        camera.TriggerMode.SetValue("Off")
    else:
        raise ValueError("Unsupported trigger mode. Use 'FreeRun' or 'Software'.")

    # Turn off auto exposure if needed
    # if "ExposureAuto" in camera.GetNodeMap().GetNodeNames():
    camera.ExposureAuto.SetValue("Off")

    # --- Set exposure ---
    if exposure_us is not None:
        camera.ExposureTimeAbs.SetValue(exposure_us)
    exposure_us = camera.ExposureTimeAbs.Value
    if gain is not None:
        camera.GainRaw.Value = gain
    gain = camera.GainRaw.Value
    if fps is not None:
        camera.AcquisitionFrameRateEnable.Value = True
        camera.AcquisitionFrameRateAbs.Value = fps
    fps = camera.AcquisitionFrameRateAbs.Value

    # --- Set ROI (region of interest) ---

    # Set offset first (must be multiple of increment)
    if offset_x is not None:
        camera.OffsetX.SetValue(offset_x)
    if offset_y is not None:
        camera.OffsetY.SetValue(offset_y)
    offset_x, offset_y = camera.OffsetX.Value, camera.OffsetY.Value
    # Then set width and height (must also follow step size)
    if width is not None:
        camera.Width.SetValue(width)
    if height is not None:
        camera.Height.SetValue(height)
    width, height = camera.Width.Value, camera.Height.Value
    print(f"Configured camera: TriggerMode={trigger_mode}, Exposure={exposure_us/1000}ms, Gain={gain}db, FrameRate={fps}FPS, ROI=({width}x{height}) at ({offset_x},{offset_y})")

def print_save(msg, file):
    print(msg)
    with open(file, 'a') as f:
        f.write(msg + '\n')
        
def capture_img(cam, timeout_ms=2000):
    trigger_mode = cam.TriggerMode.Value
    if trigger_mode == "On":
        # Ensure camera is ready for software trigger
        if cam.WaitForFrameTriggerReady(timeout_ms, pylon.TimeoutHandling_ThrowException):
            cam.ExecuteSoftwareTrigger()

    # --- Retrieve image ---
    grab_result = cam.RetrieveResult(timeout_ms, pylon.TimeoutHandling_ThrowException)
    if grab_result.GrabSucceeded():
        img = grab_result.Array
        grab_result.Release()
    else:
        grab_result.Release()
        
        # Retry once
        if trigger_mode == "On":
            if cam.WaitForFrameTriggerReady(timeout_ms, pylon.TimeoutHandling_ThrowException):
                cam.ExecuteSoftwareTrigger()

        grab_result = cam.RetrieveResult(timeout_ms, pylon.TimeoutHandling_ThrowException)
        if grab_result.GrabSucceeded():
            img = grab_result.Array
            grab_result.Release()
        else:
            grab_result.Release()
            raise RuntimeError("Failed to grab image after retry.")

    grab_result.Release()
    return img

def wait_until_second(target_second=5, check_interval=0.01):
    """
    等待系统时间的秒值为 target_second(例如 5秒)
    当达到该时刻时执行后续代码。

    参数:
        target_second: int, 要对齐的秒(0-59)
        check_interval: float, 每次检查的时间间隔（秒）
    """
    print(f"wait time align to every {target_second} second...")

    while True:
        now = datetime.datetime.now()
        if (now.second % target_second == 0) and now.microsecond < 50_000:
            print(f"时间到达 {now.strftime('%H:%M:%S.%f')}")
            break
        # time.sleep(check_interval)

    return now

# ---------- Helpers ----------
def capture_avg(cam, caputre_img_func, num=10, delay=0.02):
    imgs = []
    _ = caputre_img_func(cam)
    time.sleep(delay)
    for _ in range(num):
        img = caputre_img_func(cam)
        imgs.append(np.array(img).astype(np.float32))
    return np.mean(np.stack(imgs), axis=0)

def init_random(seed):
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ---------- Pattern ----------
def create_pattern(slm_res=(256, 256), square_size=128):
    pattern = np.zeros(slm_res, dtype=np.float32)
    x0 = (slm_res[0] - square_size) // 2
    y0 = (slm_res[1] - square_size) // 2
    pattern[x0:x0+square_size, y0:y0+square_size] = 1.0
    return pattern



# def compute_snr(ref, cur):
#     signal_power = np.sum(ref ** 2)
#     noise_power = np.sum((ref - cur) ** 2) + 1e-8
#     return 10 * np.log10(signal_power / noise_power)
def compute_snr(ref, cur):
    # Normalize to zero mean, unit variance
    ref_norm = ref / np.max(ref)
    cur_norm = cur / np.max(cur)
    # ref_norm = (ref - np.mean(ref)) / (np.std(ref) + 1e-8)
    # cur_norm = (cur - np.mean(cur)) / (np.std(cur) + 1e-8)

    signal_power = np.sum(ref_norm ** 2)
    noise_power = np.sum((ref_norm - cur_norm) ** 2) + 1e-8
    return 10 * np.log10(signal_power / noise_power)


def compute_shift(ref, cur):
    ref = ref - np.mean(ref)
    cur = cur - np.mean(cur)
    corr = correlate2d(cur, ref, mode='same', boundary='symm')
    max_idx = np.unravel_index(np.argmax(corr), corr.shape)
    center = np.array(corr.shape) // 2
    shift = np.array(max_idx) - center
    return shift


# ---------- Main Functions ----------
def capture_reference(cam, slm, caputre_img_func, pattern, num_imgs_per_check=10, project_delay=0.3):
    slm.showData(pattern)
    time.sleep(project_delay)
    baseline_img = capture_avg(cam, caputre_img_func, num=num_imgs_per_check)
    return baseline_img


def crop_center(img, crop_size=200):
    h, w = img.shape[:2]
    start_x = w // 2 - crop_size // 2
    start_y = h // 2 - crop_size // 2
    return img[start_y:start_y+crop_size, start_x:start_x+crop_size]


def monitor_drift(cam, slm, caputre_img_func, reference_img, pattern, save_dir, check_id,
                  num_imgs_per_check=10, project_delay=0.3):
    os.makedirs(save_dir, exist_ok=True)

    slm.showData(pattern)
    time.sleep(project_delay)

    current_img = capture_avg(cam, caputre_img_func, num=num_imgs_per_check)

    # Crop to central 200x200
    reference_crop = crop_center(reference_img, 200)
    current_crop = crop_center(current_img, 200)
    # Compute SNR and shift
    snr = compute_snr(reference_crop, current_crop)
    shift = compute_shift(reference_crop, current_crop)
    # snr = compute_snr(reference_img, current_img)
    # shift = compute_shift(reference_img, current_img)

    # Save result
    np.save(os.path.join(save_dir, f"check_{check_id:03d}.npy"), current_img)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(reference_img, cmap='gray')
    plt.title("Reference")
    plt.subplot(1, 2, 2)
    plt.imshow(current_img, cmap='gray')
    plt.title(f"Check {check_id}")
    plt.suptitle(f"SNR={snr:.2f}dB, Shift={shift}")
    plt.savefig(os.path.join(save_dir, f"compare_{check_id:03d}.png"))
    plt.close()

    return snr, shift


def apply_shift_to_sensor(sensor, shift, threshold=3):
    """
    sensor: torch.Tensor of shape [1, 1, 10, H, W]
    shift: tuple (dy, dx)
    threshold: apply shift only if abs(dy) or abs(dx) >= threshold
    """
    dy, dx = int(shift[0]), int(shift[1])

    if abs(dy) >= threshold or abs(dx) >= threshold:
        # Apply shift to each channel along dim=2
        sensor = torch.roll(sensor, shifts=(dy, dx), dims=(-2, -1))

    return sensor


def create_two_box_array(H=360, W=360, box_size=16, offset=50, offset2=50):
    array = np.zeros((1, 2, H, W), dtype=np.float32)

    center_y, center_x = H // 2, W // 2

    # First box (left)
    y1_start = center_y - offset - box_size // 2
    x1_start = center_x - offset - box_size // 2
    array[0, 0, y1_start:y1_start + box_size, x1_start:x1_start + box_size] = 1

    # Second box (right)
    y2_start = center_y + offset2- box_size // 2
    x2_start = center_x + offset2- box_size // 2
    array[0, 1, y2_start:y2_start + box_size, x2_start:x2_start + box_size] = 1

    return array

def create_five_box_array(H=360, W=360, box_size=16, offset=50, offset2=50):
    array = np.zeros((1, 5, H, W), dtype=np.float32)
    center_y, center_x = H // 2, W // 2

    # First box (top-left relative to center)
    y1 = center_y - offset - box_size // 2
    x1 = center_x - offset - box_size // 2
    array[0, 0, y1:y1 + box_size, x1:x1 + box_size] = 1

    # Symmetric boxes using offset2
    coords = [
        (-1, -1),  # top-left
        (-1,  1),  # top-right
        ( 1,  1),  # bottom-right
        ( 1, -1),  # bottom-left
    ]

    for i, (dy, dx) in enumerate(coords):
        y = center_y + dy * offset2 - box_size // 2
        x = center_x + dx * offset2 - box_size // 2
        array[0, i + 1, y:y + box_size, x:x + box_size] = 1

    return array


def create_eight_box_array(H=360, W=360, box_size=16, offset1=50, offset2=80):
    array = np.zeros((1, 8, H, W), dtype=np.float32)
    center_y, center_x = H // 2, W // 2

    # Define corner directions
    corners = [
        (-1, -1),  # top-left
        (-1,  1),  # top-right
        ( 1,  1),  # bottom-right
        ( 1, -1),  # bottom-left
    ]

    offsets = [offset1, offset2]
    box_index = 0

    for offset in offsets:
        for dy, dx in corners:
            y = center_y + dy * offset - box_size // 2
            x = center_x + dx * offset - box_size // 2
            array[0, box_index, y:y + box_size, x:x + box_size] = 1
            box_index += 1

    return array

def create_sixteen_box_array(H=360, W=360, box_size=16, offset1=50, offset2=80):
    array = np.zeros((1, 16, H, W), dtype=np.float32)
    center_y, center_x = H // 2, W // 2

    # Define corner directions
    corners = [
        (-1, -1),  # top-left
        (-1,  1),  # top-right
        ( 1,  1),  # bottom-right
        ( 1, -1),  # bottom-left
    ]

    # Define edge midpoint directions
    edges = [
        (-1, 0),  # top-center
        (1, 0),   # bottom-center
        (0, -1),  # center-left
        (0, 1),   # center-right
    ]

    offsets = [offset1, offset2]
    box_index = 0

    for offset in offsets:
        # Corner boxes
        for dy, dx in corners:
            y = center_y + dy * offset - box_size // 2
            x = center_x + dx * offset - box_size // 2
            array[0, box_index, y:y + box_size, x:x + box_size] = 1
            box_index += 1

        # Edge midpoint boxes
        for dy, dx in edges:
            y = center_y + dy * offset - box_size // 2
            x = center_x + dx * offset - box_size // 2
            array[0, box_index, y:y + box_size, x:x + box_size] = 1
            box_index += 1

    return array


def create_32_box_array_with_gap(H=360, W=360, box_size=16, grid_gap=16, border_margin=8):
    """
    First 16: 4x4 grid, evenly distributed in the center with gap between boxes.
    Next 16: Distributed evenly around the edge (as a ring).
    """
    array = np.zeros((1, 32, H, W), dtype=np.float32)

    # --- 1. 4x4 Grid with gap ---
    grid_rows = 4
    grid_cols = 4
    grid_height = grid_rows * box_size + (grid_rows - 1) * grid_gap
    grid_width = grid_cols * box_size + (grid_cols - 1) * grid_gap
    start_y = (H - grid_height) // 2
    start_x = (W - grid_width) // 2

    box_index = 0

    # --- 2. 16 Around Edge ---
    per_edge = 4
    # Top edge (left to right)
    for i in range(per_edge):
        x = int(i * (W - box_size) / (per_edge - 1))
        y = border_margin
        array[0, box_index, y:y+box_size, x:x+box_size] = 1
        box_index += 1
    # Right edge (top to bottom)
    for i in range(per_edge):
        x = W - box_size - border_margin
        y = int(i * (H - box_size) / (per_edge - 1))
        array[0, box_index, y:y+box_size, x:x+box_size] = 1
        box_index += 1
    # Bottom edge (right to left)
    for i in range(per_edge):
        x = int((per_edge-1-i) * (W - box_size) / (per_edge - 1))
        y = H - box_size - border_margin
        array[0, box_index, y:y+box_size, x:x+box_size] = 1
        box_index += 1
    # Left edge (bottom to top)
    for i in range(per_edge):
        x = border_margin
        y = int((per_edge-1-i) * (H - box_size) / (per_edge - 1))
        array[0, box_index, y:y+box_size, x:x+box_size] = 1
        box_index += 1

    for i in range(grid_rows):
        for j in range(grid_cols):
            y0 = start_y + i * (box_size + grid_gap)
            x0 = start_x + j * (box_size + grid_gap)
            array[0, box_index, y0:y0 + box_size, x0:x0 + box_size] = 1
            box_index += 1

    return array

def create_concentric_box_array(
    H=360, W=360,
    radii=[50, 80],              # List of radii for circles
    num_boxes=[8, 16],           # Number of boxes per circle
    box_size=16
):
    """
    Create array with boxes arranged along concentric circles.
    
    Args:
        H, W: image size
        radii: list of radii for each circle (distance from center)
        num_boxes: list of number of boxes per circle
        box_size: box size (square)
    
    Returns:
        array: (1, sum(num_boxes), H, W)
    """
    assert len(radii) == len(num_boxes)
    total_boxes = sum(num_boxes)
    array = np.zeros((1, total_boxes, H, W), dtype=np.float32)
    center_y, center_x = H // 2, W // 2
    box_index = 0
    
    for circle_idx, (radius, n_boxes) in enumerate(zip(radii, num_boxes)):
        for i in range(n_boxes):
            theta = 2 * np.pi * i / n_boxes
            y = int(center_y + radius * np.sin(theta) - box_size // 2)
            x = int(center_x + radius * np.cos(theta) - box_size // 2)
            # clip to make sure in bounds
            y0, y1 = max(0, y), min(H, y + box_size)
            x0, x1 = max(0, x), min(W, x + box_size)
            array[0, box_index, y0:y1, x0:x1] = 1
            box_index += 1
    return array

def create_concentric_box_array_ordered_for_pairwise_division(
    H=360, W=360,
    total_boxes=32,
    num_rings=3,
    base_radius=40,
    gap=30,
    box_size=16
):
    center_y, center_x = H // 2, W // 2

    # Step 1: radius and circumference
    radii = [base_radius + i * gap for i in range(num_rings)]
    circumferences = [2 * np.pi * r for r in radii]
    total_circumference = sum(circumferences)
    raw_counts = [c / total_circumference * total_boxes for c in circumferences]
    layer_counts = np.round(raw_counts).astype(int)

    # Step 2: enforce even count
    for i in range(num_rings):
        if layer_counts[i] % 2 == 1:
            layer_counts[i] += 1
    while np.sum(layer_counts) > total_boxes:
        for i in reversed(range(num_rings)):
            if layer_counts[i] > 2:
                layer_counts[i] -= 2
                if np.sum(layer_counts) == total_boxes:
                    break
    while np.sum(layer_counts) < total_boxes:
        for i in range(num_rings):
            layer_counts[i] += 2
            if np.sum(layer_counts) == total_boxes:
                break

    assert np.sum(layer_counts) == total_boxes
    assert all(c % 2 == 0 for c in layer_counts)

    # Temporarily store boxes and pairs
    first_half = []
    second_half = []
    ring_pairs = []

    for r, n_boxes in zip(radii, layer_counts):
        indices = []
        coords = []
        for i in range(n_boxes):
            theta = 2 * np.pi * i / n_boxes
            y = int(center_y + r * np.sin(theta))
            x = int(center_x + r * np.cos(theta))
            coords.append((x, y))
        ring_pair = []
        for i in range(0, n_boxes, 2):
            ring_pair.append((coords[i], coords[i+1]))
            first_half.append(coords[i])
            second_half.append(coords[i+1])
        ring_pairs.append(ring_pair)

    # Now reorder into [first_half + second_half]
    all_coords = first_half + second_half
    reordered_array = np.zeros((1, total_boxes, H, W), dtype=np.float32)

    for idx, (x, y) in enumerate(all_coords):
        y0, y1 = max(0, y - box_size // 2), min(H, y + box_size // 2)
        x0, x1 = max(0, x - box_size // 2), min(W, x + box_size // 2)
        reordered_array[0, idx, y0:y1, x0:x1] = 1

    return reordered_array

def pad_to_1080x1920(tensor, pad_value=0):
    """
    Pads a 4D tensor [C, B, H, W] to height=1080 and width=1920.

    Returns:
        Padded tensor of shape [C, B, 1080, 1920]
    """
    C, B, H, W = tensor.shape
    pad_h = max(0, 1080 - H)
    pad_w = max(0, 1920 - W)

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    # F.pad uses pad order: (W_left, W_right, H_top, H_bottom)
    padded = F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom), value=pad_value)
    return padded



def get_gaussian_kernel(kernel_size=7, sigma=1.2):
    ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    ax = ax.view(-1, 1)
    xx = ax @ ax.T
    kernel = torch.exp(-xx / (2. * sigma ** 2))     
    kernel = kernel / kernel.sum()
    return kernel

def gaussian_smooth(tensor, kernel_size=7, sigma=1.2):
    kernel = get_gaussian_kernel(kernel_size, sigma).to(tensor.device)
    kernel = kernel.view(1, 1, kernel_size, kernel_size)

    M = tensor.shape[1]
    kernel = kernel.expand(M, 1, kernel_size, kernel_size)
    return F.conv2d(tensor, kernel, padding=kernel_size // 2, groups=M)

def bin_image(img_tensor, bin_factor=5):
    """
    Bin the input torch tensor by averaging over bin_factor x bin_factor blocks,
    then resize it back to the original dimensions using nearest neighbor interpolation.

    Parameters:
        img_tensor (torch.Tensor): Input image tensor normalized [0, 1], shape (B, M, H, W).
        bin_factor (int): Factor to bin the image.

    Returns:
        torch.Tensor: Resized image tensor normalized [0, 1], shape (B, M, H, W).
    """
    B, M, H, W = img_tensor.shape
    h_bin, w_bin = H // bin_factor, W // bin_factor

    img_binned = img_tensor[:, :, :h_bin * bin_factor, :w_bin * bin_factor]
    img_binned = img_binned.reshape(B, M, h_bin, bin_factor, w_bin, bin_factor).mean(dim=(3, 5))

    img_binned_resized = F.interpolate(img_binned, size=(H, W), mode='nearest')

    return img_binned_resized

def save_latest_checkpoints(agent, epoch, output_dir, max_keep=5):
    """
    Save model checkpoint and keep only the latest `max_keep` checkpoints.

    Parameters:
    - agent: the model/agent object, must have `.state_dict()` and `.optimizer.state_dict()`
    - epoch: current epoch number
    - output_dir: directory to save checkpoints
    - max_keep: maximum number of checkpoints to keep
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save current checkpoint
    checkpoint_path = os.path.join(output_dir, f'checkpoint_{epoch}.pth')
    torch.save({
        'epoch': epoch,
        'model': agent.state_dict(),
        'optimizer': agent.optimizer.state_dict()
    }, checkpoint_path)

    # Remove older checkpoints
    checkpoints = sorted(glob.glob(os.path.join(output_dir, 'checkpoint_*.pth')),
                         key=os.path.getmtime)
    if len(checkpoints) > max_keep:
        for ckpt in checkpoints[:-max_keep]:
            os.remove(ckpt)

def save_best_checkpoints(agent, epoch, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, f'best.pth')
    torch.save({
        'epoch': epoch,
        'model': agent.state_dict(),
        'optimizer': agent.optimizer.state_dict()
    }, checkpoint_path)


import glob
import re

def load_latest_checkpoint(base_save_dir, agent, all_target_files):
    """
    Finds the latest checkpoint in the output directory, loads its state, and
    determines the exact point to resume training.

    Args:
        base_save_dir (str): The base directory where results are saved.
        agent (policy.PolicyGradientAgent): The agent instance to load state into.
        all_target_files (list): A sorted list of all target image paths.

    Returns:
        tuple: A tuple containing (agent, resume_cycle, resume_image_idx, resume_epoch).
               Returns defaults (agent, 1, 0, 0) if no checkpoint is found.
    """
    print("Attempting to find the latest checkpoint to resume from...")
    
    # Find all checkpoint files in all subdirectories
    all_checkpoints = glob.glob(os.path.join(base_save_dir, 'cycle_*', '*', 'checkpoint_*.pth'))

    if not all_checkpoints:
        print("No checkpoints found. Starting a fresh run.")
        return agent, 1, 0, 0

    # Find the most recently modified checkpoint file
    latest_ckpt_path = max(all_checkpoints, key=os.path.getmtime)
    print(f"Found latest checkpoint: {latest_ckpt_path}")

    try:
        # Use regex to parse cycle, image name, and epoch from the path
        # This is robust to different path structures
        match = re.search(r'cycle_(\d+)[\\/](.+?)[\\/]checkpoint_(\d+)\.pth', latest_ckpt_path)
        if not match:
            raise ValueError("Could not parse checkpoint path.")
            
        resume_cycle = int(match.group(1))
        image_name = match.group(2)
        resume_epoch = int(match.group(3))

        # Find the index of the image to resume from
        target_filenames = [os.path.splitext(os.path.basename(p))[0] for p in all_target_files]
        resume_image_idx = target_filenames.index(image_name)

        # Load the checkpoint data
        device = next(agent.parameters()).device
        ckpt = torch.load(latest_ckpt_path, map_location=device)
        
        agent.load_state_dict(ckpt['model'])
        agent.optimizer.load_state_dict(ckpt['optimizer'])

        print(f"Successfully loaded checkpoint. Resuming from:")
        print(f"  - Cycle: {resume_cycle}")
        print(f"  - Image: '{image_name}' (index {resume_image_idx})")
        print(f"  - Epoch: {resume_epoch + 1}")

        return agent, resume_cycle, resume_image_idx, resume_epoch

    except (ValueError, IndexError) as e:
        print(f"Error parsing checkpoint path or finding image index: {e}")
        print("Could not resume. Starting a fresh run.")
        return agent, 1, 0, 0