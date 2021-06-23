import pyrealsense2 as rs
import cv2
import numpy as np

def callback(event, x, y, flags, param):
	global depth_frame, depth_intrinsics
	if event == cv2.EVENT_LBUTTONDOWN:
		#print('type:', depth_frame.shape)
		#x, y = float(x), float(y)
		result = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [y, x], depth_frame[y, x])
		print(result)




def convert_depth_to_phys_coord_using_realsense(x, y, depth, cameraInfo):
  _intrinsics = rs.intrinsics()
  _intrinsics.width = cameraInfo.width
  _intrinsics.height = cameraInfo.height
  _intrinsics.ppx = cameraInfo.K[2]
  _intrinsics.ppy = cameraInfo.K[5]
  _intrinsics.fx = cameraInfo.K[0]
  _intrinsics.fy = cameraInfo.K[4]
  #_intrinsics.model = cameraInfo.distortion_model
  _intrinsics.model  = rs.distortion.none
  _intrinsics.coeffs = [i for i in cameraInfo.D]
  result = rs.rs2_deproject_pixel_to_point(_intrinsics, [y, x], depth)
  #result[0]: right, result[1]: down, result[2]: forward
  return result[2], -result[0], -result[1]


def show_frame(frame: rs.composite_frame, show_depth_frame: bool=True) -> None:
	"""
	Отображает кадр, полученный с камеры.
	Arguments:
	frame - Объект composite_frame для отображения.
	show_depth_frame - bool, обозначает, необходимо ли отображать фрейм глубины изображения.
	"""
	color_frame = np.asanyarray(frame.get_color_frame().get_data())
	depth_frame = frame.get_depth_frame()
	depth_frame = np.asanyarray(depth_frame.get_data()).astype(float)
	return color_frame, depth_frame



pipeline = rs.pipeline()
config = rs.config()

pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# Start streaming
pipeline.start(config)

# Get stream profile and camera intrinsics
profile = pipeline.get_active_profile()
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()

frame = pipeline.wait_for_frames()
color_frame, depth_frame = show_frame(frame, True)

depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_frame, alpha=0.03), cv2.COLORMAP_JET)

if depth_frame.shape != color_frame.shape:
	depth_frame.resize((color_frame.shape[0], color_frame.shape[1]))

cv2.namedWindow('image')
cv2.setMouseCallback('image', callback)
while(1):
	cv2.imshow('image', depth_colormap)
	if cv2.waitKey(20) & 0xFF == 27:
		break

	


