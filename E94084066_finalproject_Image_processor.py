#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np                                                 ## import numpt and named it as numpy
from tkinter import *                                              ## import all tkinter module
import tkinter.filedialog                                          ## aim to acccess computer file system
from PIL import Image, ImageFilter, ImageTk                        ## import PIL for image processing
import os                                                          ## for the file path
import tkinter.ttk                                                 ## for the user entry
import math                                                        ## import math for function sketch to use
import cv2                                                         ## import cv2 for image processing


# In[3]:


def load_image():
    """The function load_img is used to get 
    the image choose by user on their computer
    """
    global path                                                    ## global variable path used to keep the image's filepath 
    global Width                                                   ## global variable Width usec to keep the image's width  
    global Height                                                  ## global variable Height usec to keep the image's height  
    path = tkinter.filedialog.askopenfilename()                    ## access the file system
    file_label.config(text=f'原始檔案路徑  {path}')                 ## this tkinter label is used to show the filepath of our selected image 
    img = Image.open(path)                       
    Width=img.size[0]                                              ## get image width
    Height=img.size[1]                                             ## get image height
    w.set(Width)                                                   ## the string variable w and h for output image size
    h.set(Height)            
    x.set(Width)                                                   ## the string variable x and y used to show input image size
    y.set(Height)
    img = img.resize((int(Width*0.5), int(Height*0.5)),Image.ANTIALIAS) ## resise the input image size to avoid image distortion
    global img_origin                                              ## load the orginal image on the tkinter window
    img_origin = ImageTk.PhotoImage(img)
    global label_img
    label_img.configure(image=img_origin)
    label_img.pack()


# In[4]:


def output():
    """function output used to keep our output image
    in the same path of the computer, after the save operation
    is done, the tkinter window will show a message to tell the user
    the output image's file path(output file set to be .jpg file)
    """
    global output_file, output_path                                       ## global variable output_file, output_path used to keep where the output file will be saved
    output_file.save(output_path)
    tkinter.messagebox.showinfo('提示','儲存成功\n檔案路徑:' + output_path) ## show message after the save operation is done


# In[5]:


def detect_circle(img):
    """The function detect_circle is used for user to 
    detect circles in the input picture and circle it out,
    the output image will mark the circles in the picture.
    """
    img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)          ## convert the input image from type PIL.Image to type OpenCv
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                   ## convert the image's color to gray
    gaussian = cv2.GaussianBlur(gray, (3, 3), 0)                   ## use GaussianBlur to do noise reduction on the image
    circles1 = cv2.HoughCircles(gaussian, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=30, minRadius=15, maxRadius=80) ## use Hough Circle transform to do circle detection
    circles = circles1[0, :, :]                                    ## get the circles
    circles = np.uint16(np.around(circles))
    for c in circles[:]:                                           ## show the circles and the center of circles
        cv2.circle(img, (c[0], c[1]), c[2], (0, 255, 0), 3)
        cv2.circle(img, (c[0], c[1]), 2, (255, 0, 255), 10)        
    pil_img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)) ## convert the output image from type OpenCv to type PIL.Image
    return pil_img  


# In[6]:


def rebuild(u, v, sigma, percent): 
    """The function rebuild will accepts 4 parameters, u, sigma, 
    v is the content required for the reconstruction matrix, 
    and percent is the ratio of the eigenvalues used
    """
    M = len(u)
    N = len(v)
    A = np.zeros((M, N))
    count = (int)(sum(sigma))
    curSum = 0
    k = 0
    while curSum <= count * percent:         ## get singular values according to the percent entered by user
        uk = u[:, k].reshape(M, 1)           ## if the percent is more close to 1, the output image will not get large distortion
        vk = v[k].reshape(1, N)              ## ,else the percent is more close to 0, the the ourput image will get more distortion compare with input image
        A += sigma[k] * np.dot(uk, vk)
        curSum += sigma[k]
        k += 1
    A[A < 0] = 0
    A[A > 255] = 255                       
    return np.rint(A).astype("uint8")        ## round the array a's values to their closest integer value and return it   
def svd_compression(percent, img):
    """The function svd_compression will accepts 2 parameters, percent
    is the ratio of the eigenvalues used, it is a value between 0 ~ 1
    it can be enter by the user, the function getval() as we defined below will
    convert the user's input number between 0 ~ 1, and img is the input image,
    so by changing the scale percent, we can see the effect of using the eigenvalue 
    scale on the picture
    """
    a = np.array(img)                        ## the image array
    u, sigma, v = np.linalg.svd(a[:, :, 0])  ## get the singular value SVD, and return new RGB data according to the percent after processing
    R = rebuild(u, v, sigma, percent)
    u, sigma, v = np.linalg.svd(a[:, :, 1])
    G = rebuild(u, v, sigma, percent)
    u, sigma, v = np.linalg.svd(a[:, :, 2])
    B = rebuild(u, v, sigma, percent)        ## overlay the new RGB to the input image
    I = np.stack((R, G, B), 2)
    img = Image.fromarray(I)                 ## convert the output image to the type PIL.Image
    return img


# In[7]:


def getangle():
    """The function getangle is for the image 
    rotation, user can enter any angle as he(she)
    want to rotate the picture
    """
    angle = angle_Text.get(1.0, tkinter.END + "-1c")  ## get the user's input
    angle = int(angle) % 360                          ## can rotate any positive angle beacuse we will constraint it between 0 ~ 360
    return int(angle)                                 ## convert the type of angle from sring to int and return it


# In[8]:


def getval():
    """The function getval is used to convert the svd_compression
    percent enter by user to type int, and constraint the
    value between 0 and 1"""
    val = val_entry.get()                             ## get the input value
    val = int(val)                                    ## convert the type to int
    x = float(val)                                    ## x is val convert the type to float
    counter = 1
    while(val // 10 != 0):                            ## count the number of digits, then we can easily convert the input value to float type by divide the corresponding value
        val /= 10
        counter += 1
    divisor = 10 ** counter
    x = x / divisor
    return float(x)                                   ## return the percent that used to decompose the image


# In[9]:


def decay():
    """The function decay is used for image special effects by convert the
    hue of image and do some ImageFilter function on it  to get the ourput
    image look cool and has feel of nostalgia
    """
    tkinter.messagebox.showinfo('提示','圖片轉換中，請稍後!')                   ## show the image converting message
    file_path = os.path.dirname(path)                                         ## keep the file_path and filename for output used
    filename = os.path.basename(path) 
    filename = filename.split('.')[0]
    img_n = Image.open(path)                                                  ## img_n is the image for output, initially it is the input file
    img_n = img_n.resize((180, 180), Image.ANTIALIAS)                         ## resize for faster converting
    img_n = cv2.cvtColor(np.asarray(img_n),cv2.COLOR_RGB2BGR)                 ## convert the img_n from type PIL.Image to type OpenCv
    rows, cols = img_n.shape[:2]                                              ## get the image's number of row and column 
    img_aft = np.zeros((rows, cols, 3), dtype = 'uint8')                      ## img_aft is image after converting
    for i in range(rows):
        for j in range(cols):
            r = img_n[i,j][2]                                                 ## orginal proportion of red color on input image
            g = img_n[i,j][1]                                                 ## orginal proportion of green color on input image
            b = img_n[i,j][0]                                                 ## orginal proportion of blue color on input image
            B = int(0.3 * r + 0.98 * g + 0.57 * b)                            ## self define picture tint for the feel of nostalgia
            G = int(0.34 * r + 0.40 * g + 0.80 * b)
            R = int(0.72 * r + 0.13 * g + 0.24 * b)
            if B>255:
                B = 255
            if G>255:
                G = 255
            if R>255:
                R = 255
            img_aft[i,j] = np.uint8((B, G, R))                                ## convert the img_aft's type to uint8
    img_n = Image.fromarray(cv2.cvtColor(img_aft,cv2.COLOR_BGR2RGB))          ## convert the output image img_n from type OpenCv to type PIL.Image
    img_n = img_n.filter(ImageFilter.SMOOTH)                                  ## smooth the image
    img_n = img_n.filter(ImageFilter.DETAIL)                                  ## enhance the details of the image 
    img_n = img_n.filter(ImageFilter.EDGE_ENHANCE)                            ## enhance the edges of the image
    if (int(w.get()) * int(h.get()) == 0):                                    ## the output picture size can't be zero
        tkinter.messagebox.showerror('Error', "Image size can't be zero")     ## show the error message 
        w.set(x.get())                                                        ## reset to orginal size
        h.set(y.get())                                                        ## reset to output size
    img_n = img_n.resize((int(w.get()), int(h.get())), Image.ANTIALIAS)       ## resize the ouptut size user entered, if user is not change the field output image saved as the orginal size 
    
    global img_p
    img_p = img_n.resize((int(int(w.get())*0.5), int(int(h.get())*0.5)), Image.ANTIALIAS)  ## result image show on tkinter window      
    img_p = ImageTk.PhotoImage(img_p)
    label_img2.configure(image=img_p)
    label_img2.pack()
    global  output_path,output_file                                           
    output_path = file_path + '/' + filename +'_new'+ '.' + 'jpg'              
    output_file = img_n                                                       


# In[10]:


def sketch():
    """The function sketch is used for image special effects by used some
    ImageFilter function, to make the output image feel like a sketch version
    of input image
    """
    tkinter.messagebox.showinfo('提示','圖片轉換中，請稍後!')                    ## show the image converting message
    file_path = os.path.dirname(path)                                          ## keep the file_path and filename for output used
    filename = os.path.basename(path)
    filename = filename.split('.')[0]
    img_n = Image.open(path)                                                   ## img_n is the image for output, initially it is the input file
    img_n = img_n.filter(ImageFilter.DETAIL)                                   ## show the datails of the image
    img_n = img_n.filter(ImageFilter.CONTOUR)                                  ## detect the contours of image
    img_n = img_n.filter((ImageFilter.EDGE_ENHANCE))                           ## enhance the edges of image    
    if (int(w.get()) * int(h.get()) == 0):                                     ## the output picture size can't be zero
        tkinter.messagebox.showerror('Error', "Image size can't be zero")      ## show the error message 
        w.set(x.get())                                                         ## reset to orginal size
        h.set(y.get())                                                         ## reset to output size
    img_n = img_n.resize((int(w.get()), int(h.get())), Image.ANTIALIAS)        ## resize the ouptut size user entered, if user is not change the field output image saved as the orginal size 
    img_n = img_n.convert('RGB')
    global img_p
    img_p = img_n.resize((int(int(w.get())*0.5), int(int(h.get())*0.5)), Image.ANTIALIAS)   ## result image show on tkinter window  
    img_p = ImageTk.PhotoImage(img_p)
    label_img2.configure(image=img_p)
    label_img2.pack()
    global  output_path,output_file
    output_path = file_path + '/' + filename +'_new'+ '.' + 'jpg'
    output_file = img_n


# In[11]:


def animation():
    """The function animation is used for the image special effects by 
    do serial of processes to make the output image has animation feel
    """
    tkinter.messagebox.showinfo('提示','圖片轉換中，請稍後!')                     ## show the image converting message
    file_path = os.path.dirname(path)                                           ## keep the file_path and filename for output used
    filename = os.path.basename(path)
    filename = filename.split('.')[0]
    img_n = Image.open(path)                                                    ## img_n is the image for output, initially it is the input file
    img_n = cv2.cvtColor(np.asarray(img_n),cv2.COLOR_RGB2BGR)                   ## convert the img_n from type PIL.Image to type OpenCv
    color = img_n
    for i in range(10):                                                         ## perform bilateral filtering 10 times on the original image
        color = cv2.bilateralFilter(color, d = 9, sigmaColor = 9, sigmaSpace = 7)
    gray = cv2.cvtColor(img_n, cv2.COLOR_RGB2GRAY)                              ## convert the img_n's color to gray and keep it in gray
    blur = cv2.medianBlur(gray, 7)                                              ## image gray do median filter processing
    edge = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=2) ##  do edge detection and adaptive thresholding
    edge = cv2.cvtColor(edge, cv2.COLOR_GRAY2RGB)                               ## convert the result image edge to RGB image
    img_n = cv2.bitwise_and(color, edge)                                        ## processing bitwise_and function get the goal image
    img_n = Image.fromarray(cv2.cvtColor(img_n,cv2.COLOR_BGR2RGB))              ## convert the output image img_n from type OpenCv to type PIL
    if (int(w.get()) * int(h.get()) == 0):                                      ## the output picture size can't be zero
        tkinter.messagebox.showerror('Error', "Image size can't be zero")       ## show the error message 
        w.set(x.get())                                                          ## reset to orginal size
        h.set(y.get())                                                          ## reset to output size
    img_n = img_n.resize((int(w.get()), int(h.get())), Image.ANTIALIAS)         ## resize the ouptut size user entered, if user is not change the field output image saved as the orginal size 
    global img_p
    img_p = img_n.resize((int(int(w.get())*0.5), int(int(h.get())*0.5)), Image.ANTIALIAS)    ## result image show on tkinter window 
    img_p = ImageTk.PhotoImage(img_p)
    label_img2.configure(image=img_p)
    label_img2.pack()
    global  output_path,output_file
    output_path = file_path + '/' + filename +'_new'+ '.' + 'jpg'
    output_file = img_n


# In[12]:


path = ''                                                                           ## file path initailize as null string                 
feature_list = []                                                                   ## feature_list is for checkbutton used initialize as null list 
Width = 0                                                                           ## image's width                                            
Height = 0                                                                          ## image's height
output_path = None                                                                  ## initailize output_path as None
output_file = None                                                                  ## initailize output_file as None
window = Tk()                                                                       ## window is the tkinter window
label_img = None                                                                    ## label_img is for showing the input image
w = tkinter.StringVar()                                                             ## w is a tkinter.StringVar() to get user entered output image's width, when input image loaded, dafault value of w is the orginal image width
h = tkinter.StringVar()                                                             ## h is a tkinter.StringVar() to get user entered output image's height, when input image loaded, dafault value of h is the orginal image height
w.set(f'width')                                                                     ## before the input image loaded, show message to inform user this is a column for width 
h.set(f'height')                                                                    ## before the input image loaded, show message to inform user this is a column for height
window.geometry("950x650")                                                          ## set the window's size
window.title('懶人神器一鍵修圖app')                                                  ## define the title 


def check_function(img):
    """function check_function is to check if any checkbutton is checked, 
    then do the corresponding image convert process on input image img
    """
    if (0 in func_vec):
        img = img.transpose(Image.FLIP_LEFT_RIGHT)                                   ## left-right flip the image 
    if (1 in func_vec):                                    
        img = img.transpose(Image.FLIP_TOP_BOTTOM)                                   ## top-bottom flip thr image
    if (2 in func_vec):
        angle = getangle()                                                           ## get the user entered angle
        img = img.resize((int(int(w.get())*0.5), int(int(h.get())*0.5)),Image.ANTIALIAS)  ## resize the image that will show on window  
        img = img.rotate(angle)                                                      ## rotate the image
    if (3 in func_vec):                                                              
        img = img.filter(ImageFilter.BLUR)                                           ## blur the image
    if (4 in func_vec):
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        img = cv2.blur(img, (15, 15))                                                ## blur the image by used GaussianBlur
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))                  ## convert the output image img_n from type OpenCv to type PIL
    if (5 in func_vec):
        value = getval()                                                             ## get the image compact percentage
        img = svd_compression(value, img)                                            ## call svd_compression to do the decomposition
    if (6 in func_vec):
        img = img.filter(ImageFilter.EMBOSS)                                         ## emboss the image
    if (7 in func_vec): 
        img = img.filter(ImageFilter.CONTOUR)                                        ## detect the contours of the image 
    if (8 in func_vec):
        img = img.filter((ImageFilter.EDGE_ENHANCE))                                 ## enhance the edges of the image
    if (9 in func_vec):
        img = img.filter(ImageFilter.FIND_EDGES)                                     ## find edges in the image
    if (10 in func_vec):
        img = detect_circle(img)                                                     ## detect circles in the image
    return img                                                                       ## return the output image

def convert_img(path, W = Width, H = Height):
    """The function convert_img will accepts 3 parameters, path is the image's file path
    W is the input image's width, H is the input image's height, then it will check if
    any checkbutton is checked to do the corresponding image convert process, and renew
    the output image's file path
    """
    tkinter.messagebox.showinfo('提示','圖片轉換中，請稍後!')                         ## show the image converting message
    W = int(W)                                                                      ## input image width
    H = int(H)                                                                      ## input image height
    file_path = os.path.dirname(path)                                               ## keep the file_path and filename for output used
    filename = os.path.basename(path)
    filename = filename.split('.')[0]
    img = Image.open(path)
    img = check_function(img)                                                       ## check the state of all checkbutton by calling chcek_function
    if (int(w.get()) * int(h.get()) == 0):                                          ## the output picture size can't be zero
        tkinter.messagebox.showerror('Error', "Image size can't be zero")           ## show the error message 
        W = int(x.get())                                                            ## reset to orginal size
        H = int(y.get())                                                            ## reset to output size
        w.set(x.get())                                                              ## show the orginal size
        h.set(y.get())                                                              ## show the output size
    img = img.resize((W, H), Image.ANTIALIAS)                                       ## resize the ouptut size user entered, if user is not change the field output image saved as the orginal size 
    img = img.convert('RGB')
    img_n = img.resize((int(W * 0.5), int(H * 0.5)), Image.ANTIALIAS)               ## result image show on tkinter window 
    global img_new                                                              
    img_new = ImageTk.PhotoImage(img_n)
    label_img2.configure(image = img_new)
    label_img2.pack()
    global  output_path,output_file
    output_path = file_path + '/' + filename +'_new'+ '.' + 'jpg'
    output_file = img
                    
fm0 = Frame(window)                                                                 ## create a new frame for welcome message
fm0.pack(side = TOP, fill = BOTH, expand = YES)                                     ## TOP, BOTH, YES defined in tkinter
Welcome = Label(fm0,text = 'Please enjoy for the service', font=("Times New Roman", 30), bg = 'black',fg = 'white')
Welcome.pack(fill = BOTH)                                                           ## show the label message and fill the label

fm1 = Frame(window)                                                                 ## create a new frame for file loading button and the label that show the file path 
fm1.pack(side = TOP, fill = BOTH, expand = YES)
btn1 = Button(fm1, text = "請選擇圖片", height = 1, width = 10, bg = '#5CADAD', fg = 'white', font = ("Arial", 10), command = load_image) ## the file loading button
btn1.pack()
file_label = Label(fm1, text = '檔案路徑顯示位置', bg = '#FFF4C1')                   ## show the file path
file_label.pack()


fm2 = Frame(window)                                                                 ## create a new frame for the message that tell users to choose their effects on input image
lb2 = Label(fm2 ,text = '以下選擇圖片調整功能(可複選)', width = 25, height = 2, font = ("Times New Roman", 12), bg = "#D0D0D0")
lb2.pack(side = LEFT, fill = Y)
fm2.pack(side = TOP, fill = BOTH, expand = YES)
fm3 = Frame(window)                                                                 ## create a new frame for users to choose their effects on input image
fm3.pack(side = TOP, fill = BOTH, expand = YES)
label_check1 = tkinter.Label(fm3, text = '圖片轉向 : ',width = 15,font = ("Arial", 10), bg = '#ACD6FF')        ## first type contains effects for image rotate and direction change  
label_check1.pack(side = LEFT, fill = BOTH, expand = NO)

fm4 = Frame(window)                                                                                       
fm4.pack(side = TOP, fill = BOTH, expand = YES)
label_check2 = tkinter.Label(fm4, text = '圖片壓縮及模糊 : ', width = 15, font = ("Arial", 10), bg = '#D2E9FF') ## second type contains effects for image compaction and blur i
label_check2.pack(side=LEFT, fill=BOTH, expand = NO)

fm5 = Frame(window)
fm5.pack(side = TOP, fill = BOTH, expand = YES)
label_check3 = tkinter.Label(fm5, text = '圖片邊緣處理 : ', width = 15, font = ("Arial", 10), bg = '#ACD6FF')   ## third type contains effects for image edge processing
label_check3.pack(side = LEFT, fill = BOTH, expand = NO)

fm6 = Frame(window)
fm6.pack(side = TOP, fill = BOTH, expand = YES)
label_check4 = tkinter.Label(fm6, text = '懶人專區  \n一鍵特效 : ', width = 15, font = ("Arial", 10), bg = '#D2E9FF') ## last type for lazy people, just a buttom click, apply the special effects
label_check4.pack(side = LEFT, fill = BOTH, expand = NO)
btn2 = Button(fm6 , text = "冷色系懷舊", bg = '#ECF5FF', command = decay)           ## the button for special effects decay
btn2.pack(side = LEFT, expand = YES)
btn3 = Button(fm6 , text = "寫實素描風", bg = '#ECF5FF', command = sketch)          ## the button for special effects sketch
btn3.pack(side = LEFT, expand = YES)
btn4 = Button(fm6 , text = "虛擬動畫風", bg = '#ECF5FF', command = animation)       ## the button for special effects animation
btn4.pack(side = LEFT, expand = YES)


fm7 = Frame(window)                                                                ## create a new frame for users to convert the image size and apply the effects of checkbuttons, then save the output image 
fm7.pack(side = TOP, fill = BOTH, expand = YES)
btn5 = Button(fm7 ,text = "圖片轉換", bg = '#D1E9E9',command = lambda:convert_img(path, w.get(), h.get()))  ## button to convert the image by calling the function convert_img
btn5.pack(side = LEFT, expand = YES)
btn6 = Button(fm7,text="確認並儲存", bg = '#D1E9E9', command = output)              ## button to save the image by calling the function output  
btn6.pack(side = LEFT, expand = YES)

func =[]                                                                           ## check-or-not list
for i in range(11):
    func.append(IntVar())                                                          ## initailize all elements as IntVar()
func_vec = []                                                                      ## checked function list                                              
def check_feature():
    """The function check_feature is use to appand the
    correspond value into func_vec if the correspond checkbutton 
    is checked"""
    func_vec.clear()                                                               ## clear the func_vec each time
    for i in range(11):                                                            ## do the iteration to check states of every checkbuttons 
        if func[i].get():
            func_vec.append(i)                                                     ## if the checkbutton is checked, append its number into func_vec
            
## all elements in checkbutton list            
feature_list.append(tkinter.Checkbutton(fm3, text = "水平鏡像", command = check_feature, variable = func[0], onvalue = 1, offvalue = 0, padx = 40))
feature_list.append(tkinter.Checkbutton(fm3, text = "垂直鏡像", command = check_feature, variable = func[1], onvalue = 1, offvalue = 0, padx = 40))
feature_list.append(tkinter.Checkbutton(fm3, text = "旋轉", command = check_feature, variable = func[2], onvalue = 1, offvalue = 0,  padx = 20))
feature_list.append(tkinter.Checkbutton(fm4, text = "模糊效果", command = check_feature, variable = func[3], onvalue = 1, offvalue = 0 , padx = 40))
feature_list.append(tkinter.Checkbutton(fm4, text = "霧化效果", command = check_feature, variable = func[4], onvalue = 1, offvalue = 0, padx = 40))
feature_list.append(tkinter.Checkbutton(fm4, text = "分解壓縮", command = check_feature, variable = func[5], onvalue = 1, offvalue = 0, padx = 20))
feature_list.append(tkinter.Checkbutton(fm5, text = "模擬浮雕", command = check_feature, variable = func[6], onvalue = 1, offvalue = 0, padx = 40))
feature_list.append(tkinter.Checkbutton(fm5, text = "顯示輪廓", command = check_feature, variable = func[7], onvalue = 1, offvalue = 0, padx = 40))
feature_list.append(tkinter.Checkbutton(fm5, text = "邊緣增強", command = check_feature, variable = func[8], onvalue = 1, offvalue = 0, padx = 20))
feature_list.append(tkinter.Checkbutton(fm5, text = "邊緣檢測", command = check_feature, variable = func[9], onvalue = 1, offvalue = 0, padx = 40))
feature_list.append(tkinter.Checkbutton(fm5, text = "圓形檢測", command = check_feature, variable = func[10], onvalue = 1, offvalue = 0, padx = 16))

for check_bt in feature_list:                                                          ## pack all of the checkbuttons
    check_bt.pack(side=LEFT, expand=NO)

angle_label = tkinter.Label(fm3, text = '↻ 輸入旋轉角度', font=("Times New Roman", 10)) ## angle_label for tell user enter the angle he(she) want to rotate, if the checkbutton rotate is checked 
angle_label.pack(side = LEFT, expand = NO)
angle_Text = tkinter.Text(fm3, width = 3, height = 1)                                  ## get the angle from user
angle_Text.pack(side = LEFT, expand = NO)

val_label = tkinter.Label(fm4, text = '(輸入值0.', font = ("Times New Roman", 10))     ## val_label for tell user enter the compaction percentage if the svd compact checkbutton is check
val_label.pack(side = LEFT, expand = NO)
val_entry = tkinter.Entry(fm4, width = 3)                                             ## get the percent value from user
val_entry.pack(side = LEFT, expand = NO)
val_label = tkinter.Label(fm4, text = ')',font = ("Times New Roman", 10))              
val_label.pack(side = LEFT, expand = NO)


fm_size_org = Frame(window)                                                           ## create a new frame for image size control
fm_size_org.pack(side = TOP, fill = BOTH, expand = YES)
size_org = tkinter.Label(fm_size_org, text = 'orginal size :', padx = 10)             ## show the orginal size
size_org.pack(side = LEFT)
x = tkinter.StringVar()
y = tkinter.StringVar()
x.set(f'width')
y.set(f'height')
w_org = tkinter.ttk.Entry(fm_size_org, width=10, textvariable = x)                   ## show the orginal width
w_org.configure(state = 'disabled')                                                  ## read only, can't change
w_org.pack(side = LEFT, padx = 20)
multi1 = tkinter.Label(fm_size_org,text = 'x', padx = 0.5)                           ## show multiple sign x
multi1.pack(side = LEFT)
h_org = tkinter.ttk.Entry(fm_size_org, width=10, textvariable = y)                   ## show the orginal height
h_org.configure(state = 'disabled')                                                  ## read only, can't change
h_org.pack(side = LEFT, padx = 20)

size_out = tkinter.Label(fm_size_org, text = '\t\t\toutput size :')                  ## show the output size
size_out.pack(side = LEFT)
def only_uint(P):
    """The function only_uint for the tkinter.ttk.Entry
    that only accept uint value"""
    if str.isdigit(P) or P == '':                                                    ## isdigit() used to determine whether the text is a number
        return True
    else:
        return False
valid1 = (fm_size_org.register(only_uint), '%P')                                     ## for validatecommand
w_out = tkinter.ttk.Entry(fm_size_org, validate = 'key', validatecommand = valid1, width=10, textvariable = w) ## can change the output width(defalut as input width), user can only enter the unsigned int value 
w_out.pack(side = LEFT, padx = 20)
multi2 = tkinter.Label(fm_size_org,text = 'x', padx = 0.5)
multi2.pack(side = LEFT)
valid2 = (fm_size_org.register(only_uint), '%P')                                     ## for validatecommand
h_out = tkinter.ttk.Entry(fm_size_org, validate = 'key', validatecommand = valid2, width=10, textvariable = h) ## can change the output height(defalut as input height), user can only enter the unsigned int value
h_out.pack(side = LEFT, padx = 20)



fm_orginal=Frame(window)                                                             ## frame for the orginal image related                       
fm_output=Frame(window)                                                              ## frame for the output image related
fm_orginal.pack(side=LEFT, fill=BOTH, expand=YES)
fm_output.pack(side=LEFT, fill=BOTH, expand=YES)

label_img0= tkinter.Label(fm_orginal, text='選擇的原始圖片', font=("Times New Roman", 12), bg = '#FFF4C1') 
label_img0.pack(fill=BOTH)
label_img1 = tkinter.Label(fm_output, text='轉換後的圖片',  font=("Times New Roman", 12), bg = '#FFDAC8')
label_img1.pack(fill=BOTH)

label_img = tkinter.Label(fm_orginal)                                                ## show the orginal image
label_img.pack()
label_img2 = tkinter.Label(fm_output)                                                ## show the output image
label_img2.pack()

window.mainloop()                                                                    ## run the window


# In[ ]:




