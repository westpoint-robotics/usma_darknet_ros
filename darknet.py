from ctypes import *
import math
import random, time

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
bridge = CvBridge()

import cv2
import matplotlib.pyplot as plt

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

"""
class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]
"""
class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int),
                ("uc", POINTER(c_float)),
                ("points", c_int),
                ("embeddings", POINTER(c_float)),
                ("embedding_size", c_int),
                ("sim", c_float),
                ("track_id", c_int)]

class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

    

#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
lib = CDLL("/home/ros/catkin_ws/src/usma_darknet_ros/libdarknet.so", RTLD_GLOBAL)
#lib = CDLL("/home//darknet/libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

ndarray_image = lib.ndarray_to_image
ndarray_image.argtypes = [POINTER(c_ubyte), POINTER(c_long), POINTER(c_long)]
ndarray_image.restype = IMAGE

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def nparray_to_image(img):

    data = img.ctypes.data_as(POINTER(c_ubyte))
    image = ndarray_image(data, img.ctypes.shape, img.ctypes.strides)

    return image

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    #im = load_image(image, 0, 0)
    #im = array_to_image(image)
    im = nparray_to_image(image)
    num = c_int(0)
    pnum = pointer(num)
    #print("Predicting...")
    predict_image(net, im)
    #print("Prediction complete!")
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res

def draw(img, boxes):
    colors = [(255,0,0), (0,255,0), (0,0,255)]
    for i, box in enumerate(boxes):
        item, confidence, (x, y, w, h) = box
        color = colors[i%len(colors)]
        
        cv2.rectangle(img, (int(x-w//2), int(y-h//2)), (int(x+w//2), int(y+h//2)), color, 2) #top-left, bottom-right
        cv2.putText(img, "{}: {:.2f}%".format(item, confidence*100), (int(y), int(x)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    

def callback(img):
    plt.clf()
    prev = time.time()
    img = bridge.imgmsg_to_cv2(img, desired_encoding="bgr8")
    boxes = detect(net, meta, img)
    #print(boxes)
    draw(img, boxes)
    print("Delay: {:.2f}".format(time.time()-prev))
    #print(type(img))
    #cv2.imshow("Result", img)
    plt.imshow(img, cmap='gray')
    #plt.show()
    plt.pause(0.001)
    #cv2.waitKey(1)
    #cv2.destroyAllWindows()

def listener():
    rospy.init_node("object_detection", anonymous=True)
    rate = rospy.Rate(10)
    #topic = "/camera_fm/camera_fm/image_raw"
    # identify topic name with the command "rostopic list"
    topic = "/usb_cam/image_raw"
    #topic = "/camera/rgb/image"
    rospy.Subscriber(topic, Image, callback)
    while not rospy.is_shutdown():
        rate.sleep()
    
if __name__ == "__main__":
    #r = classify(net, meta, im)
    #print r[:10]
    # change directories based on your file structure
    net = load_net("yolov5.cfg", "yolov5.weights", 0)
    meta = load_meta("cfg/coco.data")
    # use gpu id 0, if you're not using gpu then keep the next line commented
    #set_gpu(0)
    try:
        listener()
    except KeyboardInterrupt:
        print("-----BYE-----")

    

