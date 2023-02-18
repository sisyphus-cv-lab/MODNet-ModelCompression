import argparse

import openvino.runtime as ov
from time import time
import numpy as np
import cv2 as cv


def get_scale_factor(im_h, im_w, ref_size):
    if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
        if im_w >= im_h:
            im_rh = ref_size
            im_rw = int(im_w / im_h * ref_size)
        elif im_w < im_h:
            im_rw = ref_size
            im_rh = int(im_h / im_w * ref_size)
    else:
        im_rh = im_h
        im_rw = im_w

    im_rw = im_rw - im_rw % 32
    im_rh = im_rh - im_rh % 32

    x_scale_factor = im_rw / im_w
    y_scale_factor = im_rh / im_h

    return x_scale_factor, y_scale_factor


def preprocess(image):
    ref_size = 512
    im = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    # unify image channels to 3
    if len(im.shape) == 2:
        im = im[:, :, None]
    if im.shape[2] == 1:
        im = np.repeat(im, 3, axis=2)
    elif im.shape[2] == 4:
        im = im[:, :, 0:3]

    # normalize values to scale it between -1 to 1
    im = (im - 127.5) / 127.5

    im_h, im_w, im_c = im.shape
    x, y = get_scale_factor(im_h, im_w, ref_size)

    # resize image
    im = cv.resize(im, (ref_size, ref_size), fx=x, fy=y, interpolation=cv.INTER_AREA)

    # prepare input shape
    im = np.transpose(im)
    im = np.swapaxes(im, 1, 2)
    im = np.expand_dims(im, axis=0).astype('float32')

    return im, im_w, im_h


def inference_image(model_path, input_path, device_name):
    core = ov.Core()

    # Compile the Model
    compiled_model = core.compile_model(model_path, device_name)

    # Create an Inference Request
    infer_request = compiled_model.create_infer_request()

    # Load image and preprocessing
    im = cv.imread(input_path)
    im, im_w, im_h = preprocess(im)

    input_tensor = ov.Tensor(array=im)
    infer_request.set_input_tensor(input_tensor)

    # start inference
    start_time = time()
    infer_request.infer()
    end_time = time()
    print(f"Inference time: {end_time - start_time:.6f} ms")

    # get output
    output = infer_request.get_output_tensor()
    output_buffer = output.data

    matte = (np.squeeze(output_buffer[0]) * 255).astype('uint8')
    matte = cv.resize(matte, dsize=(im_w, im_h), interpolation=cv.INTER_AREA)
    matte = cv.resize(matte, None, fx=0.2, fy=0.2, interpolation=cv.INTER_AREA)

    cv.imshow('matte', matte)
    cv.waitKey()
    cv.destroyAllWindows()


def inference_video(model_path, video_path, device_name):
    core = ov.Core()

    # Compile the Model
    compiled_model = core.compile_model(model_path, device_name)

    # Create an Inference Request
    infer_request = compiled_model.create_infer_request()

    capture = cv.VideoCapture(video_path)

    while capture.isOpened():
        ret, frame = capture.read()

        if frame is None:
            break
        if ret:
            im, im_w, im_h = preprocess(frame)

            input_tensor = ov.Tensor(array=im)
            infer_request.set_input_tensor(input_tensor)

            start_time = time()
            infer_request.infer()
            end_time = time()
            print(f"fps: {1. / (end_time - start_time):.2f}")

            # get output
            output = infer_request.get_output_tensor()
            output_buffer = output.data

            matte = (np.squeeze(output_buffer[0]) * 255).astype('uint8')
            matte = cv.resize(matte, dsize=(im_w, im_h), interpolation=cv.INTER_AREA)

            cv.imshow("result", matte)

        if cv.waitKey(10) & 0xFF == 27:
            break


if __name__ == "__main__":
    # define cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True,
                        help='path of .bin path')
    parser.add_argument('--image-path', type=str, required=True,
                        help='path for testing image')
    parser.add_argument('--device', type=str, required=True,
                        help='inference device, including CPU、GPU、NSC2')
    parser.add_argument('--mode', type=str, default='image', required=False,
                        help='choose image or video inference')
    args = parser.parse_args()

    model_path = args.model_path
    image_path = args.image_path
    device = args.device
    mode = args.mode

    if mode == 'image':
        inference_image(model_path, image_path, device)
    elif mode == 'video':
        inference_video(model_path, image_path, device)
