"""Данный модуль необходим для получения изображения с камеры Intel Real Sense"""
import pyrealsense2 as rs
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time

def get_video_stream_from_camera():
    pipe = rs.pipeline()
    config = rs.config()
    profile = pipe.start(config)
    for _ in range(5):
        pipe.wait_for_frames()
    
    while True:
        frame = pipe.wait_for_frames()
        yield np.asanyarray(frame.get_color_frame().get_data())


def show_video_from_camera() -> None:
    """
    Позволяет получить изображение с камеры Intel Real Sense, подключенной к USB-порту.
    В конце работы выводит в консоль средний фреймрейт за время работы.
    """
    pipe = rs.pipeline()
    config = rs.config()
    profile = pipe.start(config)
    for _ in range(5):
        pipe.wait_for_frames()

    frame_counter = 0
    start = time.time()
    while True:
        frame = pipe.wait_for_frames()
        
        show_frame(frame)
        key = cv2.waitKey(1)
        frame_counter += 1
        if key == 27:
            break

    stop = time.time()
    work_time = stop - start
    print(f'Framerate: {frame_counter / work_time}')
    pipe.stop()
    cv2.destroyAllWindows()


def show_frame(frame: rs.composite_frame, show_depth_frame: bool=True) -> None:
    """
    Отображает кадр, полученный с камеры.
    Arguments:
    frame - Объект composite_frame для отображения.
    show_depth_frame - bool, обозначает, необходимо ли отображать фрейм глубины изображения.
    """
    color_frame = np.asanyarray(frame.get_color_frame().get_data())
    depth_frame = frame.get_depth_frame()
    colorizer = rs.colorizer()
    colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())
    cv2.imshow('Color frame',color_frame)
    if show_depth_frame:
        cv2.imshow('Depth Frame', colorized_depth)


if __name__ == '__main__':
    show_video_from_camera()