"""Evaluation-only H.264 encoding with a bounded CPU budget."""
from io import BytesIO
from pathlib import Path


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
