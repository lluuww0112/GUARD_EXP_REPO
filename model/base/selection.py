import cv2
import numpy as np
import torch


def uniform_sampling(
    video_path: str,
    num_frames: int = 8,
    max_side: int | None = 720,
) -> torch.Tensor:
    """_summary_

    Args:
        video_path (str): 프레임 추출할 영상 경로
        num_frames (int, optional): 균등 샘플링할 프레임 갯수
        max_side (int | None, optional): 장축 최대길이

    장축 최대길이를 초과하면 장축=max_side로 scale down(화면비는 유지됨)

    Returns:
        torch.Tensor: (T, H, W, C) 형태의 uint8 텐서
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError("Failed to read total frame count.")

    # uniform sampling
    if num_frames == 1:
        indices = [total_frames // 2]
    else:
        indices = np.linspace(0, total_frames - 1, num_frames).round().astype(int).tolist()

    target_set = set(indices)
    frames: list[np.ndarray] = []
    frame_idx = 0

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        if frame_idx in target_set:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            if max_side is not None:
                h, w = frame_rgb.shape[:2]
                scale = min(max_side / max(h, w), 1.0)
                if scale < 1.0:
                    new_w = max(1, int(round(w * scale)))
                    new_h = max(1, int(round(h * scale)))
                    frame_rgb = cv2.resize(
                        frame_rgb,
                        (new_w, new_h),
                        interpolation=cv2.INTER_AREA,
                    )

            frames.append(frame_rgb)

        frame_idx += 1

    cap.release()

    if len(frames) != len(indices):
        raise RuntimeError(
            f"Expected {len(indices)} sampled frames, but got {len(frames)}."
        )

    # resize all frame
    base_h, base_w = frames[0].shape[:2]
    normalized_frames = []
    for f in frames:
        if f.shape[:2] != (base_h, base_w):
            f = cv2.resize(f, (base_w, base_h), interpolation=cv2.INTER_AREA)
        normalized_frames.append(f)

    video_np = np.stack(normalized_frames, axis=0)  # (T, H, W, C)
    return torch.from_numpy(video_np)
