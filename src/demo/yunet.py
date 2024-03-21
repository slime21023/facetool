import argparse
import os
import numpy as np
import cv2 as cv

from src.ml.yunet import YuNet

# Check OpenCV version
assert (
    cv.__version__ >= "4.9.0"
), "Please install latest opencv-python to try this demo: python3 -m pip install --upgrade opencv-python"

# Valid combinations of backends and targets
backend_target_pairs = [
    [cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_TARGET_CPU],
    [cv.dnn.DNN_BACKEND_CUDA, cv.dnn.DNN_TARGET_CUDA],
    [cv.dnn.DNN_BACKEND_CUDA, cv.dnn.DNN_TARGET_CUDA_FP16],
    [cv.dnn.DNN_BACKEND_TIMVX, cv.dnn.DNN_TARGET_NPU],
    [cv.dnn.DNN_BACKEND_CANN, cv.dnn.DNN_TARGET_NPU],
]

parser = argparse.ArgumentParser(
    description="YuNet: A Fast and Accurate CNN-based Face Detector (https://github.com/ShiqiYu/libfacedetection)."
)

parser.add_argument(
    "--input",
    "-i",
    type=str,
    help="Usage: Set input to a certain image, omit if using camera.",
)

onnx_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../onnx/face_detection_yunet_2023mar_int8.onnx'))
# print(f"path: {onnx_path}")
parser.add_argument(
    "--model",
    "-m",
    type=str,
    default=onnx_path,
    help="Usage: Set model type, defaults to 'face_detection_yunet_2023mar_int8.onnx'.",
)

parser.add_argument(
    "--backend_target",
    "-bt",
    type=int,
    default=0,
    help="""Choose one of the backend-target pair to run this demo:
                        {:d}: (default) OpenCV implementation + CPU,
                        {:d}: CUDA + GPU (CUDA),
                        {:d}: CUDA + GPU (CUDA FP16),
                        {:d}: TIM-VX + NPU,
                        {:d}: CANN + NPU
                    """.format(*[x for x in range(len(backend_target_pairs))]),
)

parser.add_argument(
    "--conf_threshold",
    type=float,
    default=0.95,
    help="Usage: Set the minimum needed confidence for the model to identify a face, defauts to 0.9. Smaller values may result in faster detection, but will limit accuracy. Filter out faces of confidence < conf_threshold.",
)

parser.add_argument(
    "--nms_threshold",
    type=float,
    default=0.7,
    help="Usage: Suppress bounding boxes of iou >= nms_threshold. Default = 0.3.",
)

parser.add_argument(
    "--top_k",
    type=int,
    default=1000,
    help="Usage: Keep top_k bounding boxes before NMS.",
)

parser.add_argument(
    "--save",
    "-s",
    action="store_true",
    help="Usage: Specify to save file with results (i.e. bounding box, confidence level). Invalid in case of camera input.",
)

parser.add_argument(
    "--vis",
    "-v",
    action="store_true",
    help="Usage: Specify to open a new window to show results. Invalid in case of camera input.",
)

args = parser.parse_args()


def visualize(image, results, box_color=(0, 255, 0), text_color=(0, 0, 255), fps=None):
    output = image.copy()
    landmark_color = [
        (255, 0, 0),  # right eye
        (0, 0, 255),  # left eye
        (0, 255, 0),  # nose tip
        (255, 0, 255),  # right mouth corner
        (0, 255, 255),  # left mouth corner
    ]

    if fps is not None:
        cv.putText(
            output,
            "FPS: {:.2f}".format(fps),
            (0, 15),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            text_color,
        )

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


if __name__ == "__main__":
    backend_id = backend_target_pairs[args.backend_target][0]
    target_id = backend_target_pairs[args.backend_target][1]

    # Initialize YuNet
    model = YuNet(
        model_path=args.model,
        input_size=(320, 320),
        conf_threshold=args.conf_threshold,
        nms_threshold=args.nms_threshold,
        top_k=args.top_k,
        backend_id=backend_id,
        target_id=target_id,
    )

    # If input is an image
    if args.input is not None:
        image = cv.imread(args.input)
        h, w, _ = image.shape

        # Inference
        model.set_input_size((w, h))
        results = model.infer(image)

        # Print results
        print(f"{results.shape[0]} faces detected.")
        for idx, det in enumerate(results):
            print(
                "{}: {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f}".format(
                    idx, *det[:-1]
                )
            )

        # Draw results on the input image
        image = visualize(image, results)

        # Save results if save is true
        if args.save:
            print("Resutls saved to result.jpg\n")
            cv.imwrite("result.jpg", image)

        # Visualize results in a new window
        if args.vis:
            cv.namedWindow(args.input, cv.WINDOW_AUTOSIZE)
            cv.imshow(args.input, image)
            cv.waitKey(0)
    else:  # Omit input to call default camera
        device_id = 0
        cap = cv.VideoCapture(device_id)
        w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        model.set_input_size((w, h))

        tm = cv.TickMeter()

        while cv.waitKey(1) < 0:
            hasFrame, frame = cap.read()
            if not hasFrame:
                print("No frames grabbed!")
                break

            # Inference
            tm.start()
            results = model.infer(frame)  # results is a tuple
            tm.stop()

            # Draw results on the input image
            frame = visualize(frame, results, fps=tm.getFPS())

            # Visualize results in a new Window
            cv.imshow("YuNet Demo", frame)

            tm.reset()
