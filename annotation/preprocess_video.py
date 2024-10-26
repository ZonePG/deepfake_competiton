import argparse
import os
from glob import glob
from typing import Union

import cv2
import numpy as np
from deepface import DeepFace
from deepface.modules import preprocessing
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess video")
    parser.add_argument(
        "--preprocess_path",
        type=str,
        default="/data/zonepg/datasets/dataset",
        help="Path to preprocess",
    )
    parser.add_argument("--interval", type=int, default=1, help="Interval of frames")
    parser.add_argument("--img_size", type=int, default=224, help="Size of image")
    parser.add_argument(
        "--backend",
        type=str,
        default="yolov8",
        help="Detector backend",
        choices=[
            "opencv",
            "ssd",
            "dlib",
            "mtcnn",
            "fastmtcnn",
            "retinaface",
            "mediapipe",
            "yolov8",
            "yunet",
            "centerface",
        ],
    )
    return parser.parse_args()


class VideoProcessor:
    def __init__(self, video_path, args):
        self.video_path = video_path
        self.preprocess_path = args.preprocess_path
        self.interval = args.interval
        self.frames_path = os.path.join(
            self.preprocess_path, f"frames_{self.interval}s"
        )
        self.img_size = args.img_size
        self.backend = args.backend

    def extract_frames(self):
        video_files = sorted(
            glob(os.path.join(self.preprocess_path, self.video_path, "**.**"))
        )
        for _, video_file in enumerate(tqdm(video_files)):
            video_file_id = os.path.splitext(os.path.basename(video_file))[0]
            frames_path = os.path.join(self.frames_path, self.video_path, video_file_id)
            os.makedirs(frames_path, exist_ok=True)

            cap = cv2.VideoCapture(video_file)

            # Check if the video opened successfully
            if not cap.isOpened():
                print(f"Error: Could not open video. {video_file}")
                exit()

            fps = cap.get(cv2.CAP_PROP_FPS)

            frames = []
            try:
                while True:
                    # Read the next frame
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(frame)

            finally:
                image_count = 1
                i = 0
                while i < len(frames):
                    frame_path = os.path.join(
                        frames_path, f"{str(image_count).zfill(4)}.jpg"
                    )
                    frame = frames[int(i)]
                    cv2.imwrite(frame_path, frame)
                    image_count += 1
                    i += fps * self.interval
                cap.release()

    def detect_faces(self):
        image_files = sorted(
            glob(os.path.join(self.frames_path, self.video_path, "**", "**.jpg"))
        )
        saved_face_path = os.path.join(
            self.preprocess_path,
            f"faces_{self.interval}s_{self.backend}_{self.img_size}x{self.img_size}",
        )
        for _, image_file in enumerate(tqdm(image_files)):
            saved_image_path = image_file.replace(self.frames_path, saved_face_path)

            face_image = self.detect_face(
                image_file,
                (self.img_size, self.img_size),
                detector_backend=self.backend,
                enforce_detection=False,
            )
            face_image = face_image[0]

            save_image = Image.fromarray(np.uint8(face_image * 255)).convert("RGB")
            os.makedirs(os.path.dirname(saved_image_path), exist_ok=True)
            save_image.save(saved_image_path)

    @staticmethod
    def detect_face(
        img_path: Union[str, np.ndarray],
        target_size: tuple = (224, 224),
        detector_backend: str = "opencv",
        enforce_detection: bool = True,
        align: bool = True,
    ):
        face_objs = DeepFace.extract_faces(
            img_path=img_path,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            align=align,
            grayscale=False,
        )
        extracted_face = None
        if len(face_objs) == 1:
            extracted_face = face_objs[0]["face"]
        elif len(face_objs) > 1:
            print(
                f"{img_path} Detected multiple faces but only one face is allowed, taking the global picture"
            )
            extracted_face = cv2.imread(img_path)
            extracted_face = cv2.cvtColor(extracted_face, cv2.COLOR_BGR2RGB)
        extracted_face = preprocessing.resize_image(
            img=extracted_face, target_size=target_size
        )
        return extracted_face

    def get_video_durations(self):
        video_files = sorted(
            glob(os.path.join(self.preprocess_path, self.video_path, "**.**"))
        )
        video_durations = []
        for video_file in video_files:
            cap = cv2.VideoCapture(video_file)
            if not cap.isOpened():
                print(f"Error: Could not open video. {video_file}")
                continue

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_seconds = frame_count / fps if fps > 0 else 0
            video_name = os.path.basename(video_file)
            video_durations.append((video_name, duration_seconds))
            cap.release()

        # Sort videos by duration in descending order
        video_durations.sort(key=lambda x: x[1], reverse=True)

        # Print video names and durations
        for video_name, duration in video_durations:
            print(f"Video: {video_name}, Duration: {duration:.2f} seconds")


def main():
    args = parse_args()
    video_processor_train = VideoProcessor("train", args)
    video_processor_train.extract_frames()
    video_processor_train.detect_faces()
    # video_processor_train.get_video_durations()

    video_processor_pre_test = VideoProcessor("pre_test", args)
    video_processor_pre_test.extract_frames()
    video_processor_pre_test.detect_faces()
    # video_processor_pre_test.get_video_durations()


if __name__ == "__main__":
    main()
