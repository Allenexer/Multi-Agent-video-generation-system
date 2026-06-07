"""
Video composition — frame extraction (OpenCV) + concatenation (moviepy).

No external dependencies beyond Python packages.
"""
import os
import cv2
import requests
from tools.base_tool import _tmpfile


class KeyframeExtractor:
    """Extract the last frame from a video file using OpenCV."""

    @staticmethod
    def extract_last_frame(video_path: str, output_path: str = None) -> str:
        if output_path is None:
            output_path = _tmpfile(suffix=".png")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            raise RuntimeError(f"No frames in: {video_path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, total - 1)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(f"Cannot read last frame: {video_path}")
        cv2.imwrite(output_path, frame)
        return output_path

    @staticmethod
    def download_video(url: str, output_path: str = None) -> str:
        if output_path is None:
            output_path = _tmpfile(suffix=".mp4")
        r = requests.get(url, stream=True, timeout=120)
        r.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        return output_path


class VideoConcatTool:
    """Concatenate video clips using moviepy."""

    def execute(self, clip_paths: list, output_path: str) -> dict:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        if not clip_paths:
            return {"status": "error", "message": "No clips to concat"}

        from moviepy import VideoFileClip, concatenate_videoclips
        clips = []
        for p in clip_paths:
            try:
                clips.append(VideoFileClip(p))
            except Exception as e:
                return {"status": "error", "message": f"Cannot open {p}: {e}"}
        if not clips:
            return {"status": "error", "message": "No valid clips"}
        final = concatenate_videoclips(clips, method="compose")
        final.write_videofile(output_path, codec="libx264", audio=False,
                              logger=None)
        for c in clips:
            c.close()
        final.close()
        return {"output_path": output_path, "clip_count": len(clip_paths)}
