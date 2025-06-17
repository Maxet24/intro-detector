import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import faiss
import numpy as np
from tqdm import tqdm
from videoMatch import VideoMatch


class Matcher:
    def __init__(self, embedder, similarity_threshold=0.95, k=1, batch_size=1):
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.k = k
        self.batch_size = batch_size

    def find_matching_frames(self, video_a, video_b):
        paths_a = video_a.frame_paths
        paths_b = video_b.frame_paths

        # print(video_a.emb_path, video_b.emb_path)
        emb_a = np.load(video_a.emb_path)
        emb_b = np.load(video_b.emb_path)

        emb_a /= np.linalg.norm(emb_a, axis=1, keepdims=True)
        emb_b /= np.linalg.norm(emb_b, axis=1, keepdims=True)

        index = faiss.IndexFlatIP(emb_b.shape[1])
        index.add(emb_b)

        # TO-DO: убрать batch
        matches = []
        for i in tqdm(range(0, len(emb_a), self.batch_size), desc="🔍 Поиск"):
            chunk = emb_a[i:i + self.batch_size]
            D, I = index.search(chunk, k=self.k)
            for j, (d_row, i_row) in enumerate(zip(D, I)):
                idx_a = i + j
                for sim, idx_b in zip(d_row, i_row):
                    if sim >= self.similarity_threshold:
                        
                        matches.append(VideoMatch(
                            video_a,
                            video_b,
                            paths_a[idx_a],
                            paths_b[idx_b],
                            float(sim)
                        ))
        return matches

    def expand_unique_matches(self, matches, embedder):
        def index_from_path(path):
                return int(os.path.splitext(os.path.basename(path))[0])
            
        unique_segments = []
    
        for match in matches:
            is_overlap = False
            for segment in unique_segments:
                if index_from_path(match.frame_path_a) > segment['start_frame_idx_a'] and \
                    index_from_path(match.frame_path_a) < segment['end_frame_idx_a']:
                    is_overlap = True
                    break
                    
            if is_overlap:
                continue
            
            
            segment = match.expand(embedder)
            unique_segments.append(segment)
    
        return unique_segments

    def get_segments(self, video_a, video_b, embedder):
        matches = self.find_matching_frames(video_a, video_b)
        return self.expand_unique_matches(matches, embedder)
