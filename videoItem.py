import os
import cv2
import time
import scenedetect
import glob
from pathlib import Path
import numpy as np

class VideoItem:
    def __init__(self, video_path):
        self.name = Path(video_path).stem
        self.video_path = video_path
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Не удалось открыть видео: {video_path}")
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0
        cap.release()

    def extract_frames(self, output_dir, fps=2):
        ts = time.time()
        os.makedirs(output_dir, exist_ok=True)
        self.frame_paths = []
        cap = cv2.VideoCapture(self.video_path)
        step = int(round(self.fps / fps))
        print(f"{self.frame_count} кадров всего")

        frame_idx = 0
        saved_idx = 0
        success, frame = cap.read()
        while success:
            if frame_idx % 1000 == 0:
                print(f"\r{frame_idx}/{self.frame_count} frames", end="", flush=True)
            if frame_idx % step == 0:
                out_path = os.path.join(output_dir, f"{saved_idx:05d}.jpg")
                self.frame_paths.append(out_path)
                cv2.imwrite(out_path, frame)
                saved_idx += 1
            success, frame = cap.read()
            frame_idx += 1

        cap.release()

        self.frames_path = output_dir
        print(f"\n✅ Done in {time.time() - ts:.2f} seconds.")

    def detect_scenes(self, threshold=30.0, show_progress=False):
        scene_list = scenedetect.detect(self.video_path, scenedetect.ContentDetector(threshold=threshold), show_progress=show_progress)
        scenes = [(scene[0].get_frames(), scene[1].get_frames()) for scene in scene_list]
        return scenes

    def extract_frames_by_scenes(self, output_dir, threshold=25.0, take_first_percent=0.5, brightness_threshold=2):
        ts = time.time()
        os.makedirs(output_dir, exist_ok=True)
        self.frame_paths = []
        scenes = self.detect_scenes(threshold=threshold, show_progress=True)

        cap = cv2.VideoCapture(self.video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i, (start_f, end_f) in enumerate(scenes):
            if (start_f / frame_count) > take_first_percent:
                continue
            middle_frame = (start_f + end_f) // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
            success, frame = cap.read()
            if success:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                brightness = np.mean(gray)
                if brightness < brightness_threshold:
                    continue  # пропускаем чёрный кадр
                
                out_path = os.path.join(output_dir, f"{middle_frame:07d}.jpg")
                self.frame_paths.append(out_path)
                cv2.imwrite(out_path, frame)
        cap.release()

        self.frames_path = output_dir
        print(f"\n{len(scenes)} сцен обработано за {time.time() - ts:.2f} сек.")

    def get_extracted_frames(self, frame_path):
        self.frame_path = frame_path
        self.frame_paths = sorted(glob.glob(os.path.join(frame_path, "*.jpg")))

    def set_emb_path(self, emb_path):
        self.emb_path = emb_path