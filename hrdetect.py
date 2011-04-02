#!/usr/bin/python
"""
This program is demonstration for face and object detection using haar-like features.
The program finds faces in a camera image or video stream and displays a red box around them.

Original C implementation by:  ?
Python implementation by: Roman Stanchak, James Bowman
"""
import sys
import cv
import math
import numpy
import mdp
import array
import matplotlib.pyplot as plt
from pprint import pprint
from optparse import OptionParser
import my_functions

# Parameters for haar detection
# From the API:
# The default parameters (scale_factor=2, min_neighbors=3, flags=0) are tuned 
# for accurate yet slow object detection. For a faster operation on real video 
# images the settings are: 
# scale_factor=1.2, min_neighbors=2, flags=CV_HAAR_DO_CANNY_PRUNING, 
# min_size=<minimum possible face size

min_size = (20, 20)
image_scale = 2
haar_scale = 1.2
min_neighbors = 2
haar_flags = 0

frame_no = 30 #also the number of samples
time_point = 0

min_bps = 0.75
max_bps = 4
fs = 0 

input_data = []
local_haar_detect = cv.HaarDetectObjects

temp_color_data = []

last_f = -1

def calc_heart_rate_1(x, fs):
    '''
        x is a discrete function, list of values. 
        Returns a DFT of these samples.
        fs is the sampling frequency        
        N is the total number of samples
    '''
    global min_bps
    global max_bps
    N = len(x)
    #use discrete frequency transform for the samplings
    dft_x = numpy.fft.fft(x)
    amp_max = 0
    tar_freq = 0

    tmp_list = [(fs*i/float(N),math.sqrt(dft_x[i].real**2 + dft_x[i].imag**2)) for i in range(N) if (fs*i/float(N))>min_bps and (fs*i/float(N))<max_bps]
    return tmp_list
    #global for_drawing
    #for_drawing = tmp_list[:]
    #for i in range(1, N):
    #    tmp = dft_x[i]
    #    f = fs * i / float(N)
    #    if f<min_bps or f>max_bps: continue
    #    amp = math.sqrt(tmp.real**2 + tmp.imag**2)
    #    if amp_max < amp:
    #        amp_max = amp
    #        tar_freq = f
    #return tar_freq

def convert():
    global min_bps
    global max_bps
    global input_data

    normalized_input_data = normalize_samples(input_data)
    #g_modes = ['pow3', 'tanh', 'gaus', 'skew']
    g = 'gaus'
    xs = []
    ys = []
    def my_compare(x,y):
        return cmp(y[1], x[1])
    #after_ica = mdp.fastica(numpy.array(normalized_input_data))
    tmp_input_data = numpy.array(normalized_input_data)
    jade_node = mdp.nodes.JADENode(white_parm={'svd':True})
    try:
        jade_node.train(tmp_input_data)
        after_ica = jade_node.get_projmatrix()
    except mdp.NodeException, e:
        print e
    print '-----------------'
    pprint(normalized_input_data)
    pprint(after_ica)
    pprint(jade_node.get_recmatrix())
    a = numpy.matrix(after_ica)
    b = numpy.matrix(jade_node.get_recmatrix())
    pprint(a*b)
    
    before_fft = numpy.transpose(after_ica) 
    after_fft = numpy.fft.fft(before_fft)
    pprint(after_fft)
    #x = after_fft 
    #ys = []
    #for i in x:
    #    tmp = [math.sqrt(j.real**2+j.imag**2) for j in i]
    #    ys.append(tmp)
    #xs = range(len(x[0]))
    #xs = [fs*i*60/len(x[0]) for i in xs]
    #print ys 
    #plt.plot(xs, ys[0], color='red')
    #plt.plot(xs, ys[1], color='green')    
    #plt.plot(xs, ys[2], color='blue')
    #plt.show()

def detect_and_draw(img, cascade):

    global time_point
    global frame_no
    global input_data
    global fs
    global max_bps
    
    global last_f

    # allocate temporary images
    gray = cv.CreateImage((img.width,img.height), 8, 1)
    small_img = cv.CreateImage((cv.Round(img.width / image_scale), 
                                cv.Round (img.height / image_scale)), 8, 1)

    # convert color input image to grayscale
    cv.CvtColor(img, gray, cv.CV_BGR2GRAY)

    # scale input image for faster processing
    cv.Resize(gray, small_img, cv.CV_INTER_LINEAR)

    cv.EqualizeHist(small_img, small_img)

    window = cv.CreateImage((cv.Round(img.width),
                             cv.Round (img.height)), 8, 3)
    if(cascade):
        faces = local_haar_detect(small_img, cascade, cv.CreateMemStorage(0),
                                 haar_scale, min_neighbors, haar_flags, min_size)

        channels = None
        if faces:
            for ((x, y, w, h), n) in faces:
                # the input to cv.HaarDetectObjects was resized, so scale the 
                # bounding box of each face and convert it to two CvPoints
                pt1 = (cv.Round((x + w*.2) * image_scale), cv.Round(y * image_scale))
                pt2 = (cv.Round((x + w*.8) * image_scale), cv.Round((y + h) * image_scale))
                
                window = cv.CreateImage((cv.Round(w * .6) * image_scale, 
                                         cv.Round(h) * image_scale), 8, 3)
                #cv.Smooth(window, window, cv.CV_GAUSSIAN, 3, 3)
                channels = [cv.CreateImage((cv.Round(w * .6) * image_scale, 
                                            cv.Round(h) * image_scale), 8, 1),
                            cv.CreateImage((cv.Round(w * .6) * image_scale, 
                                            cv.Round(h) * image_scale), 8, 1),
                            cv.CreateImage((cv.Round(w * .6) * image_scale, 
                                            cv.Round(h) * image_scale), 8, 1)]

                cv.GetRectSubPix(img, window, (cv.Round((pt1[0] + pt2[0]) / 2.0), 
                                               cv.Round((pt1[1] + pt2[1]) / 2.0)))

                cv.Rectangle(img, pt1, pt2, cv.RGB(255, 0, 0), 3, 8, 0)
                cv.Split(window, channels[0], channels[1], channels[2], None)
                input_data.append([cv.Avg(channels[0])[0], 
                                   cv.Avg(channels[1])[0], 
                                   cv.Avg(channels[2])[0]])

                #measure the sampling frequency
                now_point = cv.GetTickCount()

                if float(fs) / 2 < max_bps and fs != 0:
                    max_bps = float(fs) / 2

                if len(input_data) > frame_no:
                    fs = cv.GetTickFrequency() * 1000000. / (now_point - time_point)                    
                    input_data.pop(0)

                    #print my_functions.calc_heart_rate(input_data)
                    final_data = my_functions.calc_heart_rate(input_data)
                    tmp_last_f = my_functions.plot_diagrams(final_data, fs, last_f)
                    last_f = tmp_last_f
                    print last_f

                time_point = now_point
        else:
            print "Can not detect face"

         
    cv.ShowImage("result", img)

if __name__ == '__main__':

    parser = OptionParser(usage = "usage: %prog [options] [filename|camera_index]")
    parser.add_option("-c", "--cascade", 
                      action="store", dest="cascade", 
                      type="str", help="Haar cascade file, default %default", 
                      default = "/usr/local/share/opencv/haarcascades/haarcascade_frontalface_alt.xml")
    (options, args) = parser.parse_args()

    cascade = cv.Load(options.cascade)
    
    if len(args) != 1:
        parser.print_help()
        sys.exit(1)

    input_name = args[0]
    if input_name.isdigit():
        capture = cv.CreateCameraCapture(int(input_name))
    else:
        capture = None

    cv.NamedWindow("result", 1)

    if capture:
        frame_copy = None
        while True:
            frame = cv.QueryFrame(capture)
            if not frame:
                cv.WaitKey(0)
                break
            if not frame_copy:
                frame_copy = cv.CreateImage((frame.width,frame.height),
                                            cv.IPL_DEPTH_8U, frame.nChannels)
            if frame.origin == cv.IPL_ORIGIN_TL:
                cv.Copy(frame, frame_copy)
            else:
                cv.Flip(frame, frame_copy, 0)
            
            time_point = cv.GetTickCount()
            detect_and_draw(frame_copy, cascade)

            if cv.WaitKey(10) >= 0:
                break
    else:
        image = cv.LoadImage(input_name, 1)
        detect_and_draw(image, cascade)
        cv.WaitKey(0)

    #for i in input_data:
        #print ",".join(["%s" % k for k in i])
            
    cv.DestroyWindow("result")
