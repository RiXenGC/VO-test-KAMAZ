#!/usr/bin/env python3

import argparse
from pathlib import Path

import cv2
from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageFilter, StorageOptions
from sensor_msgs.msg import Image


def main():
    parser = argparse.ArgumentParser(
        description="Export a ROS 2 image topic to lossless PNG files and header timestamps"
    )
    parser.add_argument("bag", type=Path, help="Directory containing the bag metadata.yaml")
    parser.add_argument("--output-dir", type=Path, default=Path("images"))
    parser.add_argument("--times", type=Path, default=Path("calib/times.txt"))
    parser.add_argument("--topic", default="/image_raw")
    parser.add_argument("--png-compression", type=int, default=1, choices=range(10))
    args = parser.parse_args()

    if args.times.exists():
        raise FileExistsError(f"Timestamp file already exists: {args.times}")
    if args.output_dir.exists() and any(args.output_dir.glob("*.png")):
        raise FileExistsError(f"PNG files already exist in: {args.output_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.times.parent.mkdir(parents=True, exist_ok=True)

    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=str(args.bag), storage_id="sqlite3"),
        ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )
    reader.set_filter(StorageFilter(topics=[args.topic]))

    bridge = CvBridge()
    timestamps_ns = []
    frame_size = None

    while reader.has_next():
        _, serialized_message, _ = reader.read_next()
        message = deserialize_message(serialized_message, Image)
        frame = bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")

        height, width = frame.shape[:2]
        if frame_size is None:
            frame_size = (width, height)
        elif (width, height) != frame_size:
            raise RuntimeError("Image dimensions changed during the recording")

        timestamp_ns = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
        if timestamp_ns == 0:
            raise RuntimeError(f"Frame {len(timestamps_ns)} has a zero header timestamp")

        output_path = args.output_dir / f"{len(timestamps_ns):06d}.png"
        written = cv2.imwrite(
            str(output_path),
            frame,
            [cv2.IMWRITE_PNG_COMPRESSION, args.png_compression],
        )
        if not written:
            raise RuntimeError(f"Could not write image: {output_path}")

        timestamps_ns.append(timestamp_ns)
        if len(timestamps_ns) % 100 == 0:
            print(f"Exported {len(timestamps_ns)} frames", flush=True)

    if not timestamps_ns:
        raise RuntimeError(f"No messages found on {args.topic}")

    with args.times.open("w", encoding="utf-8") as times_file:
        for timestamp_ns in timestamps_ns:
            times_file.write(f"{timestamp_ns / 1_000_000_000:.9f}\n")

    duration = (timestamps_ns[-1] - timestamps_ns[0]) / 1_000_000_000
    mean_fps = (len(timestamps_ns) - 1) / duration if duration > 0 else 0.0
    print(
        f"Exported {len(timestamps_ns)} frames at {frame_size[0]}x{frame_size[1]}\n"
        f"Images: {args.output_dir}\n"
        f"Timestamps: {args.times}\n"
        f"Duration: {duration:.6f} s, mean rate: {mean_fps:.3f} Hz"
    )


if __name__ == "__main__":
    main()
