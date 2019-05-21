import numpy as np
import cv2

MEAN = (115.0, 115.0, 115.0)
STD = (55.0, 55.0, 55.0)


def get_frames(video_path, skip=1):
    frames = []
    metadata_list = []
    cap = cv2.VideoCapture(video_path)
    frame_no = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if len(frame.shape) == 3:
                frame = np.flip(frame, 2)  # to RGB
            frames.append(frame)
            metadata_list.append({'shape': frame.shape, 'type': frame.dtype, 'frame': frame_no})
        else:
            break
        frame_no += 1
    if not cap.isOpened():
        raise RuntimeError("Video file {} could not be opened, check configuration.".format(video_path))
    cap.release()
    if skip > 1:
        frames = frames[::skip]
        metadata_list = metadata_list[::skip]
    return frames, metadata_list


def normalize(x, mean=MEAN, std=STD):
    x = x.astype(np.float32)
    x[:, :, 0] = (x[:, :, 0] - mean[0]) / std[0]
    x[:, :, 1] = (x[:, :, 1] - mean[1]) / std[1]
    x[:, :, 2] = (x[:, :, 2] - mean[2]) / std[2]
    return x
