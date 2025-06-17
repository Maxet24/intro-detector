from pathlib import Path
import shutil
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import faiss
import pickle
import json
from matcher import Matcher


class SeriesManager:

    def __init__(
        self,
        unpaired_dir: str | Path,
        series_dir: str | Path,
        json_file_name: str | Path,
        embedder,
        fps: int = 2,
        sim_thr: float = 0.92,
        run_len: int = 10,
        min_segment_frames: int = 10,
        min_max_segments_length: int = 150,
    ):
        self.unpaired_dir = Path(unpaired_dir)
        self.unpaired_dir.mkdir(parents=True, exist_ok=True)
        self.series_dir = Path(series_dir)
        self.series_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self.fps = fps
        self.sim_thr = sim_thr
        self.run_len = run_len
        self.matcher = Matcher(self.embedder)
        self.min_segment_frames = min_segment_frames
        self.json_data = {}
        self.json_file_name = series_dir / Path(json_file_name)
        self.min_max_segments_length = min_max_segments_length


    def _emb_path_for(self, video_stem: str) -> Path:
        return self.unpaired_dir / f"{video_stem}.npy"

    def cache_embeddings(self, video):
        emb_path = self._emb_path_for(video.name)
        video.set_emb_path(emb_path)
        video_path = emb_path.with_suffix(".pkl")

        if emb_path.exists() and video_path.exists():
            emb = np.load(emb_path)
            with open(video_path, "rb") as f:
                video = pickle.load(f)

            return emb, video
    
        # кадры только из первой части эпизода по сценам
        tmp_dir = self.unpaired_dir / f"_frames_{video.name}"
        video.extract_frames_by_scenes(output_dir=tmp_dir)
    
        emb = self.embedder.embed_images(video.frame_paths)
        np.save(emb_path, emb.astype("float32"))
    
        with open(video_path, "wb") as f:
            pickle.dump(video, f)

        # res
        res_emb_path = Path("res") / f"{video.name}.npy"
        res_video_path = res_emb_path.with_suffix(".pkl")
        np.save(res_emb_path, emb.astype("float32"))
        with open(res_video_path, "wb") as f:
            pickle.dump(video, f)
    
        shutil.rmtree(tmp_dir, ignore_errors=True)
    
        return emb, video

    

    def add_video(self, video):
        emb_new, new_video = self.cache_embeddings(video)

        segments = []
        videos = []
        # перебираем существующих одиночек
        for emb_file in self.unpaired_dir.glob("*.npy"):
            if emb_file.name == f"{new_video.name}.npy":
                continue  # это мы только что записали


            # Получить все сегменты
            emb_old = np.load(emb_file)
            video_path = emb_file.with_suffix(".pkl")
            with open(video_path, "rb") as f:
                video_old = pickle.load(f)
                videos.append(video_old)

            segment = self.matcher.get_segments(new_video, video_old, self.embedder)
            segments.append(segment)
            print(video_path, segment)

        # перебираем сериалы
        for emb_file in self.series_dir.glob("*.npy"):
            if emb_file.name == f"{new_video.name}.npy":
                continue  # это мы только что записали


            # Получить все сегменты
            emb_old = np.load(emb_file)
            video_path = emb_file.with_suffix(".pkl")
            with open(video_path, "rb") as f:
                video_old = pickle.load(f)
                videos.append(video_old)
            # print(video_path)
            segment = self.matcher.get_segments(new_video, video_old, self.embedder)
            segments.append(segment)
            print(video_path, segment)

        print(f'Сегменты: {segments}')
        
        # найти лучшее совпадение
        max_segments_length_video = None
        max_segments_length = 0
        max_segments_length_id = 0
        for vid_id in range(len(segments)):
            s = 0
            for segment in segments[vid_id]:
                s += segment['end_frame_idx_a'] - segment['start_frame_idx_a']
                
            if s > self.min_segment_frames and s > max_segments_length:
                max_segments_length = s
                max_segments_length_id = vid_id
                max_segments_length_video = videos[vid_id]

        print(f'segment_len: {max_segments_length}')
        
        if max_segments_length >= self.min_max_segments_length:
            # Пара найдена!
            old_npy_path = max_segments_length_video.emb_path
            old_pkl_path = old_npy_path.with_suffix(".pkl")
        
            dst_old_npy = self.series_dir / Path(max_segments_length_video.name).with_suffix(".npy")
            dst_old_pkl = self.series_dir / Path(max_segments_length_video.name).with_suffix(".pkl")

            print('PATHS')
            print(old_npy_path, old_pkl_path)
            print(dst_old_npy, dst_old_pkl)

            if old_npy_path != dst_old_npy:
                shutil.copy(old_npy_path, dst_old_npy)
                # shutil.copy(new_pkl_path, dst_new_pkl)
                with open(old_pkl_path, "rb") as f:
                    video_old = pickle.load(f)
                video_old.set_emb_path(dst_old_npy)
                with open(dst_old_pkl, "wb") as f:
                    pickle.dump(video_old, f)

            # удаляем старые
            if new_video.emb_path.exists():
                new_video.emb_path.unlink()
            pkl_path = new_video.emb_path.with_suffix(".pkl")
            if pkl_path.exists():
                pkl_path.unlink()

            if old_npy_path != dst_old_npy:
                if max_segments_length_video.emb_path.exists():
                    max_segments_length_video.emb_path.unlink()
                pkl_path = max_segments_length_video.emb_path.with_suffix(".pkl")
                if pkl_path.exists():
                    pkl_path.unlink()

            # upload data to json
            # create of not exists
            # TO-DO in class
            if not Path(self.json_file_name).exists():
                with open(self.json_file_name, "w", encoding="utf-8") as f:
                    json.dump({}, f, ensure_ascii=False, indent=4)
        
            with open(self.json_file_name, "r", encoding="utf-8") as f:
                json_data = json.load(f)

            # TO-DO IMPORTANT
            # i здесь это номер совпадающего сегмента
            if new_video.name not in json_data:
                json_segments = []
                for i in range(len(segments[max_segments_length_id])):
                    json_segments.append({
                        'start': segments[max_segments_length_id][i]['start_time_a'],
                        'end': segments[max_segments_length_id][i]['end_time_a'],
                    })

                json_data[new_video.name] = json_segments
                
            if videos[max_segments_length_id].name not in json_data:
                json_segments = []
                for i in range(len(segments[max_segments_length_id])):
                    json_segments.append({
                        'start': segments[max_segments_length_id][i]['start_time_b'],
                        'end': segments[max_segments_length_id][i]['end_time_b'],
                    })
                    
                json_data[videos[max_segments_length_id].name] = json_segments
            
            with open(self.json_file_name, "w", encoding="utf-8") as f:
                json.dump(json_data, f, ensure_ascii=False, indent=4)
        
            return str(dst_old_npy), str(dst_old_pkl)

        # пары нет -> emb и pickle уже в папке 
        return None

    def save_to_json():
        pass
