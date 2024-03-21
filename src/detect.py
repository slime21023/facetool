import os
from typing import List
import cv2 as cv
import numpy as np
from src.ml.yunet import YuNet


_onnx_path = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "./onnx/face_detection_yunet_2023mar_int8.onnx"
    )
)
_model = YuNet(
    model_path=_onnx_path,
    input_size=(320, 320),
    conf_threshold=0.95,
    nms_threshold=0.8,
    top_k=100,
    backend_id=cv.dnn.DNN_BACKEND_OPENCV,
    target_id=cv.dnn.DNN_TARGET_CPU,
)


def set_config(
    conf_threshold: float = 0.95, nms_threshold: float = 0.8, top_k: int = 100
):
    _model = YuNet(
        model_path=_onnx_path,
        input_size=(320, 320),
        conf_threshold=conf_threshold,
        nms_threshold=nms_threshold,
        top_k=top_k,
        backend_id=cv.dnn.DNN_BACKEND_OPENCV,
        target_id=cv.dnn.DNN_TARGET_CPU,
    )


def infer(image: cv.typing.MatLike):
    h, w, _ = image.shape
    _model.set_input_size((w, h))
    results = _model.infer(image)
    return results


def visualize(
    image: cv.typing.MatLike,
    results: np.ndarray,
    box_color: tuple = (0, 255, 0),
    text_color: tuple = (0, 0, 255),
):
    output = image.copy()
    landmark_color = [
        (255, 0, 0),  # right eye
        (0, 0, 255),  # left eye
        (0, 255, 0),  # nose tip
        (255, 0, 255),  # right mouth corner
        (0, 255, 255),  # left mouth corner
    ]

    for det in results:
        bbox = det[0:4].astype(np.int32)
        cv.rectangle(
            output,
            (bbox[0], bbox[1]),
            (bbox[0] + bbox[2], bbox[1] + bbox[3]),
            box_color,
            2,
        )

        conf = det[-1]
        cv.putText(
            output,
            "{:.4f}".format(conf),
            (bbox[0], bbox[1] + 12),
            cv.FONT_HERSHEY_DUPLEX,
            0.5,
            text_color,
        )

        landmarks = det[4:14].astype(np.int32).reshape((5, 2))
        for idx, landmark in enumerate(landmarks):
            cv.circle(output, landmark, 2, landmark_color[idx], 2)

    return output


def bounding_box(image: cv.typing.MatLike, nums=1) -> List[tuple]:
    h, w, _ = image.shape
    _model.set_input_size((w, h))
    results = _model.infer(image)
    if len(results) == 0:
        return []

    bbox_points = []
    for det in results:
        bbox = det[0:4].astype(np.int32)
        point1 = (bbox[0], bbox[1])
        point2 = (bbox[0] + bbox[2], bbox[1] + bbox[3])
        bbox_points.append((point1, point2))

    return bbox_points[:nums]


def normalize_face_image(
    image: cv.typing.MatLike, size=(90, 120), nums=1
) -> List[tuple]:
    h, w, _ = image.shape
    _model.set_input_size((w, h))
    results = _model.infer(image)
    if len(results) == 0:
        return []

    normal_faces = []
    for det in results:
        bbox = det[0:4].astype(np.int32)
        pt1 = (bbox[0], bbox[1])
        pt2 = (bbox[0] + bbox[2], bbox[1] + bbox[3])

        bounding_img = image[pt1[1] : pt2[1], pt1[0] : pt2[0]].copy()
        img = cv.resize(bounding_img, size, interpolation=cv.INTER_AREA)
        normal_faces.append(img)

    return normal_faces[:nums]
