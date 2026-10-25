"""Post-run image storage and H.264 encoding with bounded resource usage."""
from io import BytesIO
from pathlib import Path
import os
import shutil
import subprocess
import sys


def save_pending_episode_as_images(root, repo_id):
    """Embed staged PNGs in a portable image dataset, without video encoding.

    Only a small batch of PNG bytes is held in memory. Keep the original frames
    until both the replacement parquet and metadata are committed, so a failed
    save can be retried without losing the recording.
    """
    import copy
    import json
    import tempfile

    import pyarrow as pa
    import pyarrow.parquet as pq
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.datasets.utils import DEFAULT_IMAGE_PATH, get_hf_features_from_features

    root = Path(root)
    meta = LeRobotDatasetMetadata(repo_id, root=root)
    keys = list(meta.video_keys)
    if not keys:
        return  # Already saved as an image dataset; safe to retry the labels.
    if meta.total_episodes != 1:
        raise ValueError(f"Expected one pending episode in {root}; found {meta.total_episodes}.")
    info = copy.deepcopy(meta.info)
    for key in keys:
        info["features"][key]["dtype"] = "image"
        info["features"][key].pop("info", None)
    info["video_path"] = None
    schema = get_hf_features_from_features(info["features"]).arrow_schema
    data_path = root / meta.get_data_file_path(0)
    # Build the full replacement before touching the original dataset.
    with tempfile.TemporaryDirectory(prefix="eval_images_", dir=root) as temporary:
        replacement = Path(temporary) / "data.parquet"
        with pq.ParquetFile(data_path) as source, pq.ParquetWriter(replacement, schema) as writer:
            if source.metadata.num_rows != meta.total_frames:
                raise ValueError("Pending episode frame count does not match metadata.")
            for batch in source.iter_batches(batch_size=8):
                table = pa.Table.from_batches([batch])
                frame_indices = table["frame_index"].to_pylist()
                columns = []
                for field in schema:
                    if field.name in keys:
                        images = []
                        for frame_index in frame_indices:
                            path = root / DEFAULT_IMAGE_PATH.format(
                                image_key=field.name, episode_index=0, frame_index=int(frame_index),
                            )
                            images.append({"bytes": path.read_bytes(), "path": None})
                        columns.append(pa.array(images, type=field.type))
                    else:
                        columns.append(table[field.name].cast(field.type))
                writer.write_table(pa.Table.from_arrays(columns, schema=schema))
        info_path = Path(temporary) / "info.json"
        info_path.write_text(json.dumps(info, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")
        replacement.replace(data_path)
        info_path.replace(root / "meta" / "info.json")
    for key in keys:
        image_dir = (root / DEFAULT_IMAGE_PATH.format(image_key=key, episode_index=0, frame_index=0)).parent
        shutil.rmtree(image_dir)
    # A previous failed MP4 attempt may have produced partial videos. The image
    # dataset now contains every source frame, so those artifacts are redundant.
    if (root / "videos").is_dir():
        shutil.rmtree(root / "videos")


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
