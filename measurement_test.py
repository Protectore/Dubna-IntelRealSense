import pyrealsense2 as rs
import cv2
import numpy as np


DEPTH_MIN, DEPTH_MAX = 0.4, 10  # В метрах
WIDTH, HEIGHT = 1280, 720
FPS = 30

SELECTION_COLOR = (0, 255, 0)
LINE_THICKNESS = 1
start, stop = None, None
depth_start, depth_stop = None, None
size = None


def mouse_events_handler(event, x, y, flags, param):
    # Объявляем глобальные переменные. Первой строчкой идут изображения, отображаемые
    # в окнах, во второй идут переменные для отрисовки и вычисления размера линии
    global depth_colormap_to_show, color_frame_to_show
    global start, stop, size
    global depth_start, depth_stop
    if event == cv2.EVENT_LBUTTONDOWN:
        start = (x, y)
        stop = start
        # Вычисляем координаты выбранной точки цветной картинки на карте глубины
        depth_start = rs.rs2_project_color_pixel_to_depth_pixel(
            frame.get_depth_frame().get_data(), depth_scale,
            DEPTH_MIN, DEPTH_MAX,
            depth_intrinsics, color_intrinsics, depth_to_color_extrinsics,
            color_to_depth_extrinsics, start)
        # Преобразуем к инту, opencv иначе ругается
        depth_start = [int(i) for i in depth_start]
        depth_stop = depth_start
        # На самом деле пока что это не размер, а координаты проекции начальной точки в пространстве
        size = rs.rs2_deproject_pixel_to_point(depth_intrinsics, depth_start[::-1], depth_frame[depth_start[1], depth_start[0]])
        print(f'start: x = {x}, y = {y}, coords = {size}')

    elif event == cv2.EVENT_MOUSEMOVE:
        if (start):
            depth_colormap_to_show = np.copy(depth_colormap)
            color_frame_to_show = np.copy(color_frame)
            stop = (x, y)
            depth_stop = rs.rs2_project_color_pixel_to_depth_pixel(
                frame.get_depth_frame().get_data(), depth_scale,
                DEPTH_MIN, DEPTH_MAX,
                depth_intrinsics, color_intrinsics, depth_to_color_extrinsics,
                color_to_depth_extrinsics, stop)
            depth_stop = [int(i) for i in depth_stop]
            # Тут иногда может возникать ошибка, наверное, нужно исправить?
            try:
                cv2.line(depth_colormap_to_show, depth_start, depth_stop, SELECTION_COLOR, LINE_THICKNESS)
                cv2.line(color_frame_to_show, start, stop, SELECTION_COLOR, LINE_THICKNESS)
            except Exception as exc:
                print('Ашипка')
                print(depth_stop)
                print(exc)

    elif event == cv2.EVENT_LBUTTONUP:
        stop = (x, y)
        depth_stop = rs.rs2_project_color_pixel_to_depth_pixel(
            frame.get_depth_frame().get_data(), depth_scale,
            DEPTH_MIN, DEPTH_MAX,
            depth_intrinsics, color_intrinsics, depth_to_color_extrinsics,
            color_to_depth_extrinsics, stop)
        depth_stop = [int(i) for i in depth_stop]
        cv2.line(depth_colormap_to_show, depth_start, depth_stop, SELECTION_COLOR, LINE_THICKNESS)
        cv2.line(color_frame_to_show, start, stop, SELECTION_COLOR, LINE_THICKNESS)
        end_coords = rs.rs2_deproject_pixel_to_point(depth_intrinsics, depth_stop[::-1], depth_frame[depth_stop[1], depth_stop[0]])
        print(f'stop: x = {x}, y = {y}, coords = {end_coords}')
        # Если начальная/конечная точки попали туда, где не вычисленна глубина, вычисления невозможно выполнить
        if (np.argmax(size) == 0 or np.argmax(end_coords) == 0):
            print('Eror, start or end depth isn\'t defined')
        else:
            size = np.abs(np.array(end_coords) - np.array(size))
            print('size:')
            print(f'width = {round(size[1], 2)}mm')
            print(f'height = {round(size[0], 2)}mm')
            print(f'depth = {round(size[2], 2)}mm')
            print(f'vector norm = {round(np.linalg.norm(size), 2)}mm')
        start, stop = None, None
        depth_start, depth_stop = None, None
        size = None
        print()


def take_photo(pipeline: rs.pipeline) -> (np.array, np.array, np.array):
    """
    !!!!!!!!!!!!!!!!!!!!!!
    Возвращает всякие штуки, напишите, пожалуйста, по человечески
    !!!!!!!!!!!!!!!!!!!!!!
    pipeline - Объект pipeline, обрабатвающий поток с камеры.
    """
    frame = pipeline.wait_for_frames()  # Получаем кадр

    color_frame = np.asanyarray(frame.get_color_frame().get_data())  # Достаём цветную картинку

    depth_frame = np.asanyarray(frame.get_depth_frame().get_data())  # Достаём глубину. Тут ещё .astype(float) был
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_frame, alpha=0.03), cv2.COLORMAP_JET)  # Делаем картинку на основе глубины
    # color_frame = cv2.resize(color_frame, (depth_colormap.shape[1], depth_colormap.shape[0]))
    return frame, color_frame, depth_frame, depth_colormap


pipeline = rs.pipeline()
config = rs.config()

# Настройка каналов потока
config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)

pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()

# Start streaming
pipeline.start(config)
# Религиозная традиция, пропускать первые 5 кадров
for _ in range(5):
    pipeline.wait_for_frames()

# Полуаем профиль потока
profile = pipeline.get_active_profile()

# Тут достаём всякие штуки для маппинга цветных координат с координатами глубины.
# !!!!!
depth_scale = device.first_depth_sensor().get_depth_scale()
depth_stream = profile.get_stream(rs.stream.depth)
color_stream = profile.get_stream(rs.stream.color)
depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()  # Эта штука нужна для первоначального кода
color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
depth_to_color_extrinsics = depth_stream.as_video_stream_profile().get_extrinsics_to(color_stream)
color_to_depth_extrinsics = color_stream.as_video_stream_profile().get_extrinsics_to(depth_stream)
# !!!!!

# Собираем необходимые данные при запуске
frame, color_frame, depth_frame, depth_colormap = take_photo(pipeline)
# Делаем копии изображений для отображения. Они нужны чтобы нарисованные линии
# не записывались на оригиналы
color_frame_to_show = np.copy(color_frame)
depth_colormap_to_show = np.copy(depth_colormap)

if depth_frame.shape != color_frame.shape:
    depth_frame.resize((color_frame.shape[0], color_frame.shape[1]))

# Создаём окна для отображения карты глубины и цветной картинки
cv2.namedWindow('depth colormap')
cv2.namedWindow('image')
# Вешаем обработчик событий мыши
# cv2.setMouseCallback('depth colormap', mouse_events_handler)
cv2.setMouseCallback('image', mouse_events_handler)
# Основной цикл
while(1):
    # Отображаем изображения
    cv2.imshow('depth colormap', depth_colormap_to_show)
    cv2.imshow('image', color_frame_to_show)
    key = cv2.waitKey(1)
    # При нажатии клавиши p (английская p, раскладка учитывается)
    if key == ord('p'):
        print('taking photo...')
        # Получаем новый кадр и делаем его копии
        frame, color_frame, depth_frame, depth_colormap = take_photo(pipeline)
        color_frame_to_show = np.copy(color_frame)
        depth_colormap_to_show = np.copy(depth_colormap)
        print('done\n')
    # При нажатии Esc
    elif key & 0xFF == 27:
        # Выход из программы
        break
