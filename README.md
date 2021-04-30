# Darknet #
Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.
(original source: https://github.com/pjreddie/darknet)
(modified by CDT Maeng 21')

## Step by Step guide

* **configure Makefile** Skip this step if you do NOT want to use GPU. Change line 1 in Makefile to 'GPU=1'

* **make** Run 'make' to create libdarknet.so

* **locate libdarknet.so** Change line 70 in darknet.py to 'lib = CDLL("YOUR PATH/libdarknet.so", RTLD_GLOBAL)'

* **locate yolov#.cfg and yolov#.weights** Change line 222 in darknet.py to 'net = load_net("YOUR PATH/yolov5.cfg", "YOUR PATH/yolov5.weights", 0)'

* **change topic name** Change line 213 in darknet.py to 'topic = "YOUR IMAGE TOPIC"'

* **Run darknet** Run 'python darknet.py' in your command prompt

* **download weight files** Visit [Darknet project website](http://pjreddie.com/darknet) to download the most recent weight file

## More information

For more information see the [Darknet project website](http://pjreddie.com/darknet).

For questions or issues please use the [Google Group](https://groups.google.com/forum/#!forum/darknet).



