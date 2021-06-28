import cv2 
import pyrealsense2 as rs2
import numpy as np


if __name__ == '__main__':

    # use colorizer to visualize depth data
    color_map = rs2.colorizer()
    # use black to white color map
    color_map.set_option(rs2.RS2_OPTION_COLOR_SCHEME, 3)
    # use decimation filter to reduce the amount of data while preserving best samples
    decimation = rs2.decimation_filter()
    decimation.set_option(rs2.RS2_OPTION_FILTER_MAGNITUDE, 2)
    # define transformations from and to disparity domain
    depth2disparity = rs2.disparity_transform()
    disparity2depth = rs2.disparity_transform(False)
    # define spatial filter (edge-preserving)
    spat = rs2.spatial_filter()
    #enable hole-filling
    spat.set_option(rs2.RS2_OPTION_HOLES_FILL, 5) # 5 = fill all the zero pixels
    # define temporal filter
    temp = rs2.temporal_filter()
    # spatially align all streams to depth viewport
    align_to = rs2.align(rs2.RS2_STREAM_DEPTH)

    