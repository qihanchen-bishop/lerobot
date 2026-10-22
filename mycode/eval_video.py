"""Evaluation-only H.264 encoding with a bounded CPU budget."""
from io import BytesIO
from pathlib import Path
import os
import shutil
import subprocess
import sys


def encode_eval_video_in_subprocess(image_dir, video_path, fps):
    """Keep codec work outside the GUI interpreter and prefer desktop I/O."""
    command = [sys.executable, str(Path(__file__).resolve()),
               str(Path(image_dir).resolve()), str(Path(video_path).resolve()), str(fps)]
    ionice = shutil.which("ionice") if sys.platform.startswith("linux") else None
    if ionice is not None:
        command = [ionice, "-c", "2", "-n", "7", *command]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode:
        raise RuntimeError(f"Video encoding failed (exit {result.returncode}): {result.stdout[-4000:]}")


def encode_eval_video(image_dir, video_path, fps):
    import av
    from PIL import Image

    frames = sorted(Path(image_dir).glob("frame-*.png"))
    if not frames:
        raise FileNotFoundError(f"No recorded PNG frames in {image_dir}")
    with Image.open(frames[0]) as first:
        width, height = first.size
    with av.open(str(video_path), "w") as output:
        stream = output.add_stream("h264", rate=fps, options={"preset": "veryfast", "crf": "30", "g": "2"})
        stream.width, stream.height = width, height
        stream.pix_fmt = "yuv420p"
        stream.thread_count = 2
        last_png = None
        image = None
        for path in frames:
            png = path.read_bytes()
            # Held policy/live inputs repeat identical PNGs for many frames.
            # Decode them once while preserving every output frame and timestamp.
            if png != last_png:
                with Image.open(BytesIO(png)) as source:
                    image = source.convert("RGB")
                last_png = png
            frame = av.VideoFrame.from_image(image)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("video_path", type=Path)
    parser.add_argument("fps", type=int)
    args = parser.parse_args()
    # Only the disposable encoder process gets a lower CPU priority. Never
    # lower the priority of the GUI, robot control, or inference threads.
    if hasattr(os, "nice"):
        os.nice(10)
    encode_eval_video(args.image_dir, args.video_path, args.fps)
