import pyrealsense2 as rs
import cv2
import numpy as np


SELECTION_COLOR = (0, 255, 0)
LINE_THICKNESS = 1
start, stop = None, None
size = None


def mouse_events_handler(event, x, y, flags, param):
    global depth_frame, depth_colormap, depth_intrinsics, depth_colormap_to_show, color_frame, color_frame_to_show, start, stop, size
    if event == cv2.EVENT_LBUTTONDOWN:
        start = (x, y)
        stop = start
        size = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [y, x], depth_frame[y, x])
        print(f'start: x = {x}, y = {y}, coords = {size}')
        cv2.line(depth_colormap_to_show, start, stop, SELECTION_COLOR, LINE_THICKNESS)
        cv2.line(color_frame_to_show, start, stop, SELECTION_COLOR, LINE_THICKNESS)

    elif event == cv2.EVENT_MOUSEMOVE:
        if (start):
            depth_colormap_to_show = np.copy(depth_colormap)
            color_frame_to_show = np.copy(color_frame)
            stop = (x, y)
            cv2.line(depth_colormap_to_show, start, stop, SELECTION_COLOR, LINE_THICKNESS)
            cv2.line(color_frame_to_show, start, stop, SELECTION_COLOR, LINE_THICKNESS)

    elif event == cv2.EVENT_LBUTTONUP:
        stop = (x, y)
        cv2.line(depth_colormap_to_show, start, stop, SELECTION_COLOR, LINE_THICKNESS)
        cv2.line(color_frame_to_show, start, stop, SELECTION_COLOR, LINE_THICKNESS)
        end_coords = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [y, x], depth_frame[y, x])
        print(f'stop: x = {x}, y = {y}, coords = {end_coords}')
        if (np.argmax(size) == 0 or np.argmax(end_coords) == 0):
            print('Eror, start or end depth isn\'t defined')
        else:
            size = np.abs(np.array(end_coords) - np.array(size))
            print('size:')
            print(f'width = {round(size[1], 2)}mm')
            print(f'height = {round(size[0], 2)}mm')
            print(f'depth = {round(size[2], 2)}mm')
            print(f'vector norm = {round(np.linalg.norm(size), 2)}mm')
        start = None
        stop = None
        size = None
        print()


def convert_depth_to_phys_coord_using_realsense(x, y, depth, cameraInfo):
    _intrinsics = rs.intrinsics()
    _intrinsics.width = cameraInfo.width
    _intrinsics.height = cameraInfo.height
    _intrinsics.ppx = cameraInfo.K[2]
    _intrinsics.ppy = cameraInfo.K[5]
    _intrinsics.fx = cameraInfo.K[0]
    _intrinsics.fy = cameraInfo.K[4]
    #_intrinsics.model = cameraInfo.distortion_model
    _intrinsics.model = rs.distortion.none
    _intrinsics.coeffs = [i for i in cameraInfo.D]
    result = rs.rs2_deproject_pixel_to_point(_intrinsics, [y, x], depth)
    # result[0]: right, result[1]: down, result[2]: forward
    return result[2], -result[0], -result[1]


def get_image(frame: rs.composite_frame) -> (np.array, np.array):
    """
    Возвращает цветное изображение и depth frame
    Arguments:
    frame - Объект composite_frame, содержащий данные для считывания.
    """
    color_frame = np.asanyarray(frame.get_color_frame().get_data())
    depth_frame = frame.get_depth_frame()
    depth_frame = np.asanyarray(depth_frame.get_data()).astype(float)
    return color_frame, depth_frame


def take_photo(pipeline: rs.pipeline) -> (([bool, rs.composite_frame]), np.array, np.array, np.array):
    """
    !!!!!!!!!!!!!!!!!!!!!!
    Возвращает всякие штуки, напишите, пожалуйста, по человечески
    !!!!!!!!!!!!!!!!!!!!!!
    pipeline - Объект pipeline, обрабатвающий поток с камеры.
    """
    frame = pipeline.wait_for_frames()
    color_frame, depth_frame = get_image(frame)
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_frame, alpha=0.03), cv2.COLORMAP_JET)
    color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
    color_frame = cv2.resize(color_frame, (depth_colormap.shape[1], depth_colormap.shape[0]))
    return frame, color_frame, depth_frame, depth_colormap


pipeline = rs.pipeline()
config = rs.config()

pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()

#config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
#config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# Start streaming
pipeline.start(config)
for _ in range(10):
    pipeline.wait_for_frames()

# Get stream profile and camera intrinsics
profile = pipeline.get_active_profile()
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()

frame, color_frame, depth_frame, depth_colormap = take_photo(pipeline)
color_frame_to_show = np.copy(color_frame)
depth_colormap_to_show = np.copy(depth_colormap)

if depth_frame.shape != color_frame.shape:
    depth_frame.resize((color_frame.shape[0], color_frame.shape[1]))

cv2.namedWindow('depth colormap')
cv2.namedWindow('image')
cv2.setMouseCallback('depth colormap', mouse_events_handler)
cv2.setMouseCallback('image', mouse_events_handler)
while(1):
    cv2.imshow('depth colormap', depth_colormap_to_show)
    cv2.imshow('image', color_frame_to_show)
    key = cv2.waitKey(1)
    if key == ord('p'):
        print('taking photo...')
        frame, color_frame, depth_frame, depth_colormap = take_photo(pipeline)
        color_frame_to_show = np.copy(color_frame)
        depth_colormap_to_show = np.copy(depth_colormap)
        print('done\n')
    elif key & 0xFF == 27:
        break
