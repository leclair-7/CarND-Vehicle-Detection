## Writeup

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/car-hog.jpg
[image3]: ./examples/non-car-hog.jpg
[image4]: ./examples/boundingboxes.jpg
[image5]: ./examples/pipeline-processed.jpg
[image6]: ./examples/bboxes_and_heat.png
[image7]: ./examples/labels_map.png
[image8]: ./examples/output_bboxes.png
[video1]: ./results.mp4

#### Here I will consider the [Rubric](https://review.udacity.com/#!/rubrics/513/view)  points individually and describe how I addressed each point in my implementation.  
---

### Histogram of Oriented Gradients (HOG)

#### 1. How HOG features were extracted from the training images.

I extracted the HOG features using the skimage.feature function, hog. In my code, the function, get_hog_features() which can be found in the vehicle_detection_utilities.py file calls it. To find the HOG features, I first wrote a program to try various combinations of parameters including 'pix_per_cell','cell_per_block' , and 'hog_channel'. I then wrote the classification accuracy to a log file. The HOG parameter with "ALL" always obtained a higher accuracy than 0,1, or 2.

Below are examples of outputs of the hog function with a car and a non-car image with the following parameters:

- color space: `YCrCb`  
- HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]
![alt text][image3]

#### 2. Explain how you settled on your final choice of HOG parameters.

The HOG parameter with "ALL" always obtained a higher accuracy than 0,1, or 2 if the rest of the parameters are fixed. Given the rate of false positives, it was a good choice to spend time finding good parameters.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a Linear SVM by using sklearn's,  train_test_split, which splits and shuffles the datapoints to training and test sets. The datapoints were then scaled then svc.fit(X_train, y_train) was called, which trained the LinearSVC object. Below are the parameters used.   

Parameters:
1. 'color_space' : 'YCrCb',
2. 'spatial_size' : (32,32),
3. 'orient' : 9,
4. 'pix_per_cell' : 8,
5. 'cell_per_block' : 2,
6. 'hog_channel' : "ALL",
7. 'spatial_feat':True,
8. 'hist_feat':True,
9. 'hog_feat':True

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Given the sliding window search's runtime, I used the Hog Sub-sampling Window Search from the lectures. This function extracts the HOG features of a region of the image (chosen by function parameters), and is sub-sampled based on the scaling size. Then samples are processed with a loop.

I began by calling the find_cars function on the periphery y = 400 to y = 496 with a scale of 1.5. I then used a function that prints out 4 images in various places of the test video to see if the parameters called in find_car made new false positives or if they found the cars. I then added calls to find_car until the ystop value was 696 which is approximately the highest y value where a car can appear. Each call had a different scale corresponding to the region in the image. Far away cars had a scale of 1 to 1.5 and nearby regions had a scale of around 2.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

---

### Video Implementation

#### 1. Link to the video output.  
Here's a [link to my video result](./result.mp4)

#### 2. Filter implementation for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap, thresholded that map, then I added it to a queue which kept track of the past 5 heatmaps. I then averaged the heatmaps and thresholded that map and returned the resulting heatmap.   `scipy.ndimage.measurements.label()` was then called to identify individual blobs in the heatmap. This function also combines overlapping bounding boxes as a feature.  I then assumed each blob corresponded to a vehicle.  Lastly I constructed bounding boxes to cover the area of each blob detected.  


#### Here are 4 frames of images after the bounding boxes were calculated:
![alt text][image4]
#### Here are 4 frames of images after the pipeline was completed:
![alt text][image5]

---

### Discussion

#### 1. Problems / Issues / Improvements:

Issues faced were mainly false positives especially on the shadow areas. It looks like the shadow areas had similar features to the black car. Aggressive thresholding and averaging fixed this. As a result, my model would not work in dark or rainy conditions.

I think the main limitations of my implementation was the data used and the preprocessing done. To make it more robust, rotating the images, doing a random brightness change, and slight perspective changes would probably help. A completely different approach, using YOLOv3 would be even better because there are downloadable weights from a pretrained network on the web.  
