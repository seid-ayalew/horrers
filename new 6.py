def add_flicker(frame):
    if np.random.rand() > 0.9:  # 10% chance to flicker
        frame = cv2.addWeighted(frame, 0.8, np.zeros_like(frame), 0.2, 0)
    return frame