import numpy as np
import cv2 as cv
import os


class YuNet:
    def __init__(
        self,
        model_path: os.PathLike,
        input_size: tuple = (320, 320),
        conf_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
        backend_id: int = 0,
        target_id: int = 0,
    ):
        self._model_path = model_path
        self._input_size = input_size
        self._conf_threshold = conf_threshold
        self._nms_threshold = nms_threshold
        self._top_k = top_k
        self._backend_id = backend_id
        self._target_id = target_id

        self._model = cv.FaceDetectorYN.create(
            model=self._model_path,
            config="",
            input_size=self._input_size,
            score_threshold=self._nms_threshold,
            top_k=self._top_k,
            backend_id=self._backend_id,
            target_id=self._target_id,
        )

    def set_computes(self, backend_id, target_id):
        """
        Set the opencv backend and target by backend_id and target_id
        """
        self._backend_id = backend_id
        self._target_id = target_id
        self._model = cv.FaceDetectorYN.create(
            model=self._model_path,
            config="",
            input_size=self._input_size,
            score_threshold=self._nms_threshold,
            top_k=self._top_k,
            backend_id=self._backend_id,
            target_id=self._target_id,
        )

    def set_input_size(self, input_size: tuple):
        self._model.setInputSize(input_size)

    def infer(self, image):
        faces = self._model.detect(image)
        return np.array([]) if faces[1] is None else faces[1]
