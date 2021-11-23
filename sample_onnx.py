#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse

import cv2 as cv
import numpy as np
import onnxruntime


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--movie", type=str, default=None)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument(
        "--model",
        type=str,
        default='model/DM-Count_QNRF_640_360.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='640,360',
    )

    args = parser.parse_args()

    return args


def run_inference(onnx_session, input_size, image):
    # Pre process:Resize, BGR->RGB, Standardization, float32 cast, Transpose
    input_image = cv.resize(image, dsize=(input_size[0], input_size[1]))

    input_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    input_image = (input_image / 255 - mean) / std
    input_image = np.array(input_image, dtype=np.float32)
    input_image = input_image.transpose(2, 0, 1)
    input_image = input_image.reshape(-1, 3, input_size[1], input_size[0])

    # Inference
    input_name = onnx_session.get_inputs()[0].name
    result, _ = onnx_session.run(None, {input_name: input_image})

    # Post process
    result_map = np.array(result)

    return result_map, int(np.sum(result_map))


def main():
    args = get_args()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    if args.movie is not None:
        cap_device = args.movie
    image_path = args.image

    model_path = args.model
    input_size = [int(i) for i in args.input_size.split(',')]

    # Initialize video capture
    if image_path is None:
        cap = cv.VideoCapture(cap_device)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Load model
    onnx_session = onnxruntime.InferenceSession(model_path)
    if image_path is not None:
        image = cv.imread(image_path)
        debug_image = copy.deepcopy(image)

        start_time = time.time()

        # Inference execution
        result_map, peaple_count = run_inference(
            onnx_session,
            input_size,
            image,
        )

        elapsed_time = time.time() - start_time

        # Draw
        debug_image = draw_debug(
            debug_image,
            elapsed_time,
            result_map,
            peaple_count,
        )

        elapsed_time = time.time() - start_time

        # Draw
        debug_image = draw_debug(
            debug_image,
            elapsed_time,
            result_map,
            peaple_count,
        )

        cv.imshow('DM-Count Demo : Original Image', image)
        cv.imshow('DM-Count Demo : Activation Map', debug_image)
        cv.waitKey(0)
    else:
        while True:
            start_time = time.time()

            # Capture read
            ret, frame = cap.read()
            if not ret:
                break
            debug_image = copy.deepcopy(frame)

            # Inference execution
            result_map, peaple_count = run_inference(
                onnx_session,
                input_size,
                frame,
            )

            elapsed_time = time.time() - start_time

            # Draw
            debug_image = draw_debug(
                debug_image,
                elapsed_time,
                result_map,
                peaple_count,
            )

            key = cv.waitKey(1)
            if key == 27:  # ESC
                break
            cv.imshow('DM-Count Demo : Original Image', frame)
            cv.imshow('DM-Count Demo : Activation Map', debug_image)

        cap.release()
    cv.destroyAllWindows()


def draw_debug(image, elapsed_time, result_map, peaple_count):
    image_width, image_height = image.shape[1], image.shape[0]
    debug_image = copy.deepcopy(result_map[0, 0])

    # Apply ColorMap
    debug_image = (debug_image - debug_image.min()) / (
        debug_image.max() - debug_image.min() + 1e-5)
    debug_image = (debug_image * 255).astype(np.uint8)
    debug_image = cv.applyColorMap(debug_image, cv.COLORMAP_JET)

    debug_image = cv.resize(debug_image, dsize=(image_width, image_height))

    # addWeighted
    debug_image = cv.addWeighted(image, 0.35, debug_image, 0.65, 1.0)

    # Inference elapsed time
    cv.putText(debug_image,
               "Elapsed Time : " + '{:.1f}'.format(elapsed_time * 1000) + "ms",
               (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1,
               cv.LINE_AA)

    # Peaple Count
    cv.putText(debug_image, "People Count : " + str(peaple_count), (10, 60),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv.LINE_AA)

    return debug_image


if __name__ == '__main__':
    main()