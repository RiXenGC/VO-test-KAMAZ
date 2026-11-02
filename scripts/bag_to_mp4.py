import argparse
from pathlib import Path

import cv2
from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageFilter, StorageOptions
from sensor_msgs.msg import Image


def main():
    parser = argparse.ArgumentParser(description="Convert a ROS 2 image topic to MP4")
    parser.add_argument("bag", type=Path, help="Directory containing the bag metadata.yaml")
    parser.add_argument("output", type=Path, help="Output MP4 file")
    parser.add_argument("--topic", default="/image_raw")
    parser.add_argument("--fps", type=float, default=10.0)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Output file already exists: {args.output}")

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
    writer = None
    frame_size = None
    frame_count = 0

    try:
        while reader.has_next():
            _, serialized_message, _ = reader.read_next()
            message = deserialize_message(serialized_message, Image)
            frame = bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")

            height, width = frame.shape[:2]
            if writer is None:
                frame_size = (width, height)
                writer = cv2.VideoWriter(
                    str(args.output),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    args.fps,
                    frame_size,
                )
                if not writer.isOpened():
                    raise RuntimeError("Could not open the MP4 video writer")
            elif (width, height) != frame_size:
                raise RuntimeError("Image dimensions changed during the recording")

            writer.write(frame)
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Encoded {frame_count} frames", flush=True)
    finally:
        if writer is not None:
            writer.release()

    if frame_count == 0:
        raise RuntimeError(f"No messages found on {args.topic}")

    print(f"Saved {frame_count} frames to {args.output}")


if __name__ == "__main__":
    main()
