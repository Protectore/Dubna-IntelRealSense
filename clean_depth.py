import cv2 
import pyrealsense2 as rs2
import numpy as np


def set_device_options(profile):
    # set high density preset to reduce missing depth pixels
    depth_sensor = profile.get_device().first_depth_sensor()

    # get high_accuracy_visual_preset
    #preset_range = depth_sensor.get_option_range(rs2.option.visual_preset)
    #for i in range(int(preset_range.max)):
    #    visual_preset = depth_sensor.get_option_value_description(rs2.option.visual_preset, i)
    #    print(f'{visual_preset} - {i}')

    depth_sensor.set_option(rs2.option.visual_preset, 4) # 4 = high dencity


def depth_processing(data : rs2.composite_frame) -> rs2.depth_frame : 
    ''' Applying filters to improve depth quality '''

    # use decimation filter to reduce the amount of data while preserving best samples
    decimation = rs2.decimation_filter()
    decimation.set_option(rs2.option.filter_magnitude, 2)
    # define transformations from and to disparity domain
    depth2disparity = rs2.disparity_transform()
    disparity2depth = rs2.disparity_transform(False)
    # define spatial filter (edge-preserving)
    spat = rs2.spatial_filter()
    #enable hole-filling
    spat.set_option(rs2.option.holes_fill, 5) # 5 = fill all the zero pixels
    # define temporal filter
    temp = rs2.temporal_filter()
    # spatially align all streams to depth viewport
    align_to = rs2.align(rs2.stream.depth)

    data = align_to.process(data)
    depth = data.get_depth_frame()
    depth = decimation.process(depth)
    depth = depth2disparity.process(depth)
    depth = spat.process(depth)
    depth = disparity2depth.process(depth)
    return depth


def colorize_depth(depth: rs2.depth_frame) -> rs2.video_frame: 
    ''' Colorize depth data for vizualization '''

     # use colorizer to visualize depth data
    color_map = rs2.colorizer()
    # use black to white color map
    color_map.set_option(rs2.option.color_scheme, 3)

    colorized = color_map.colorize(depth)
    return colorized
    #np.asanyarray(colorized.get_data())


def get_frame(pipe: rs2.pipeline):
    frames = pipe.wait_for_frames()
    if frames:
        depth = frames.get_depth_frame()
        color = frames.get_color_frame()
    return frames, depth, color


def to_image_representation(frames: rs2.frame = None, depth_frame: rs2.depth_frame = None,
 color_frame: rs2.video_frame = None):

    if not depth_frame:
        depth_frame = frames.get_depth_frame()
    if not color_frame:
        color_frame = frames.get_color_frame()

    depth_frame = colorize_depth(depth_frame)

    color_img = np.asanyarray(color_frame.get_data())
    depth_img = np.asanyarray(depth_frame.get_data())

    return depth_img, color_img
    



if __name__ == '__main__':

    COLOR_WINDOW = 'image'
    DEPTH_WINDOW = 'depth'

    # declare RealSense pipeline, encapsulating the actual device and sensors
    pipe = rs2.pipeline()

    # configure camera pipeline for depth + color streaming
    cfg = rs2.config()
    cfg.enable_stream(rs2.stream.depth) #enable default depth
    # set color stream format to RGBA 
    # to allow blending of the color frames on top of the depth frames
    cfg.enable_stream(rs2.stream.color, rs2.format.rgba8)

    # start pipeline 
    profile = pipe.start(cfg)

    set_device_options(profile)

    #depth_sensor = profile.get_device().first_depth_sensor()
    #depth_sensor.set_option(rs2.option.visual_preset, 3) # 4 = high dencity

    while True:
        frames, depth_frame, color_frame = get_frame(pipe)
        depth_frame = depth_processing(frames)
        depth_img, col_img = to_image_representation(depth_frame=depth_frame, color_frame=color_frame)
        cv2.imshow(COLOR_WINDOW, col_img)
        cv2.imshow(DEPTH_WINDOW, depth_img)
        if cv2.waitKey(1) == 27:
            break

    pipe.stop()
    cv2.destroyAllWindows()

    

    


    