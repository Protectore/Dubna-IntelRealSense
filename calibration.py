"""Данный модуль необходим для автоматической калибровки камеры Intel Real Sense"""
import pyrealsense2 as rs2


def calibrate(image_width=256, image_height=144, frame_rate=90, calibration_settings='', timeout=15000):
    """
    Функция калибровки камеры.
    Для калибровки камеры необходимо создать условия, описанные в статье 
    Intel® RealSense™ Self-Calibration for D400 Series Depth Cameras
    https://dev.intelrealsense.com/docs/self-calibration-for-depth-cameras
    
    Returns:
    result - калибровочная таблица
    health - оценка работоспособности камеры, если health < 0.25 - камера откалибрована хорошо, \
    0.25 < health < 0.75 - результат может быть улучшен, health > 0.75 - требуется повторная калибровка
    """
    
    # Once started successfully, dev object can be casted to rs2.auto_calibrated_device by calling:
    pipeline = rs2.pipeline()
    config = rs2.config()

    config.enable_stream(rs2.stream.depth,
                         image_width, 
                         image_height,
                         rs2.format.z16,
                         frame_rate)
    device = pipeline.start(config).get_device()

    # Once started successfully, dev object can be casted to rs2.auto_calibrated_device by calling:
    calibrated_device = rs2.auto_calibrated_device(device)

    result, health = calibrated_device.run_on_chip_calibration(calibration_settings,
                                                                lambda _ : print('.'),
                                                                timeout)

    return result, health


if __name__ == '__main__':
    result, health = calibrate()
    print(health)