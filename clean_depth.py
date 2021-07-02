import cv2 
import pyrealsense2 as rs2
import numpy as np

from frames import *

def set_device_options(profile):
    # set high density preset to reduce missing depth pixels
    depth_sensor = profile.get_device().first_depth_sensor()

    # get high_accuracy_visual_preset
    #preset_range = depth_sensor.get_option_range(rs2.option.visual_preset)
    #for i in range(int(preset_range.max)):
    #    visual_preset = depth_sensor.get_option_value_description(rs2.option.visual_preset, i)
    #    print(f'{visual_preset} - {i}')

    depth_sensor.set_option(rs2.option.visual_preset, 4) # 4 = high dencity



class DepthProcesser:

    def __init__(self):
         # use decimation filter to reduce the amount of data while preserving best samples
        self.decimation = rs2.decimation_filter()
        self.decimation.set_option(rs2.option.filter_magnitude, 4) # 2 or 4
        # define transformations from and to disparity domain
        self.depth2disparity = rs2.disparity_transform()
        self.disparity2depth = rs2.disparity_transform(False)
        # define spatial filter (edge-preserving)
        self.spat = rs2.spatial_filter()
        #enable hole-filling
        self.spat.set_option(rs2.option.holes_fill, 5) # 5 = fill all the zero pixels
        self.spat.set_option(rs2.option.filter_magnitude, 5)
        self.spat.set_option(rs2.option.filter_smooth_alpha, 1)
        self.spat.set_option(rs2.option.filter_smooth_delta, 50)
        # define temporal filter
        self.temp = rs2.temporal_filter()
        # spatially align all streams to depth viewport
        self.align_to = rs2.align(rs2.stream.depth)

    
    def process(self, data : rs2.composite_frame, align: bool = False) -> rs2.depth_frame:
        if align:
            data = self.align_to.process(data)
        depth = data.get_depth_frame()
        #print('shape before', np.asanyarray(depth.get_data()).shape)
        #depth = self.decimation.process(depth)
        #print('shape after decimation', np.asanyarray(depth.get_data()).shape)
        depth = self.depth2disparity.process(depth)
        #print('shape after depth2disparsity', np.asanyarray(depth.get_data()).shape)
        depth = self.spat.process(depth)
        #print('shape after spatial filter', np.asanyarray(depth.get_data()).shape)
        depth = self.temp.process(depth)
        #print('shape after temp', np.asanyarray(depth.get_data()).shape)
        depth = self.disparity2depth.process(depth)
        #print('shape after disparsity2depth', np.asanyarray(depth.get_data()).shape)
        return depth





def depth_processing(data : rs2.composite_frame) -> rs2.depth_frame : 
    ''' Applying filters to improve depth quality '''

    # use decimation filter to reduce the amount of data while preserving best samples
    decimation = rs2.decimation_filter()
    decimation.set_option(rs2.option.filter_magnitude, 2) # 2 or 4
    # define transformations from and to disparity domain
    depth2disparity = rs2.disparity_transform()
    disparity2depth = rs2.disparity_transform(False)
    # define spatial filter (edge-preserving)
    spat = rs2.spatial_filter()
    #enable hole-filling
    spat.set_option(rs2.option.holes_fill, 5) # 5 = fill all the zero pixels
    spat.set_option(rs2.option.filter_magnitude, 5)
    spat.set_option(rs2.option.filter_smooth_alpha, 1)
    spat.set_option(rs2.option.filter_smooth_delta, 50)
    # define temporal filter
    temp = rs2.temporal_filter()
    # spatially align all streams to depth viewport
    align_to = rs2.align(rs2.stream.depth)

    data = align_to.process(data)
    depth = data.get_depth_frame()
    depth = decimation.process(depth)
    depth = depth2disparity.process(depth)
    depth = spat.process(depth)
    depth = temp.process(depth)
    depth = disparity2depth.process(depth)
    return depth
    

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

    depth_processer = DepthProcesser()

    while cv2.waitKey(1) != 27:

        frames, depth_frame, color_frame = get_frame(pipe)
        #depth_frame = depth_processing(frames)
        depth_frame = depth_processer.process(frames, align=False)
        depth_img, col_img = to_image_representation(depth_frame=depth_frame, color_frame=color_frame)
        cv2.imshow(COLOR_WINDOW, col_img)
        cv2.imshow(DEPTH_WINDOW, depth_img)
    cv2.destroyAllWindows()
    pipe.stop()
    

    

    


    