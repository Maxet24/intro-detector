import os
import cv2
from PIL import Image

class VideoMatch:
    def __init__(self, video_a, video_b, frame_path_a, frame_path_b, similarity):
        self.video_a = video_a
        self.video_b = video_b
        self.frame_path_a = frame_path_a
        self.frame_path_b = frame_path_b
        self.similarity = similarity

    def __repr__(self):
        return (
            f"<Match {self.frame_path_a} and {self.frame_path_b} | "
            f"sim={self.similarity:.3f}>"
        )

    def expand(self, embedder, similarity_threshold=0.80, step_in_secs=0.5, max_steps=600):
        def index_from_path(path):
            return int(os.path.splitext(os.path.basename(path))[0])
    
        def read_frame(video_path, frame_index):
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = cap.read()
            cap.release()
            if not success:
                return None
            # BGR → RGB для PIL
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(frame)

        def format_time(seconds):
            total_seconds = int(round(seconds))
            hrs = total_seconds // 3600
            mins = (total_seconds % 3600) // 60
            secs = total_seconds % 60
            return f"{hrs:02d}:{mins:02d}:{secs:02d}"

        step_in_frames_a = int(step_in_secs * self.video_a.fps)
        step_in_frames_b = int(step_in_secs * self.video_b.fps)
        
        idx_a = index_from_path(self.frame_path_a)
        idx_b = index_from_path(self.frame_path_b)
    
        start_a, end_a = idx_a, idx_a
        start_b, end_b = idx_b, idx_b
    
        # Вправо
        for _ in range(max_steps):
            idx_a_next = end_a + step_in_frames_a
            idx_b_next = end_b + step_in_frames_b
            frame_a = read_frame(self.video_a.video_path, idx_a_next)
            frame_b = read_frame(self.video_b.video_path, idx_b_next)
            if frame_a is None or frame_b is None:
                break
            emb_a = embedder.embed_image(frame_a)
            emb_b = embedder.embed_image(frame_b)
            sim = 1 - embedder.distance(emb_a, emb_b, metric="cosine")
            if sim < similarity_threshold:
                break
            end_a = idx_a_next
            end_b = idx_b_next
    
        # Влево
        for _ in range(max_steps):
            idx_a_prev = start_a - step_in_frames_a
            idx_b_prev = start_b - step_in_frames_b
            if idx_a_prev < 0 or idx_b_prev < 0:
                break
            frame_a = read_frame(self.video_a.video_path, idx_a_prev)
            frame_b = read_frame(self.video_b.video_path, idx_b_prev)
            if frame_a is None or frame_b is None:
                break
            emb_a = embedder.embed_image(frame_a)
            emb_b = embedder.embed_image(frame_b)
            sim = 1 - embedder.distance(emb_a, emb_b, metric="cosine")
            if sim < similarity_threshold:
                break
            start_a = idx_a_prev
            start_b = idx_b_prev
    
        return {
            "start_frame_idx_a": start_a,
            "end_frame_idx_a": end_a,
            "start_frame_idx_b": start_b,
            "end_frame_idx_b": end_b,
            "length": (end_a - start_a + 1),
            "start_time_a": format_time(start_a / self.video_a.fps),
            "end_time_a": format_time(end_a / self.video_a.fps),
            "start_time_b": format_time(start_b / self.video_b.fps),
            "end_time_b": format_time(end_b / self.video_b.fps),
        }



