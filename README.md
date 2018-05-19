## **Vehicle Detection Project**
The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # "Image References"
[classes]: ./img/classes.png
[hog]: ./img/hog.png
[heatmap]: ./img/heatmap.png
[heatmap1]: ./img/heatmap1.png
[heatmap2]: ./img/heatmap2.png
[heatmap3]: ./img/heatmap3.png
[windows]: ./img/windows.png
[negs]: ./img/negatives.png
[video]: ./project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

## Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of eight of each of the `vehicle` and `non-vehicle` classes:

![alt text][classes]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the YUV color space and HOG parameters of `orientations=11`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)`:


![alt text][hog]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried a few combinations of color spaces between RGB, HSV, YUV and YCrCb and various values for HOG parameters. RGB and HSV lower accuracies in general (closer to 95%), while YUV and YCrCb seemed to be about the same (around 98%). YUV consistently was a little faster and produced slightly better training results for the SVM. The pixel per cell parameter was set to 16 as opposed to 8, and that improved the training speed without much change in accuracy. I tried a range of number of orientations all the way up to 32, but eventually settled on 11 after comparing some different HOG images. The 11 orientations images seemed less noisy than some of the others.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

In the sixth block of the notebook, I trained an RBF based SVM using the HOG vector shown above and along with the color histogram and image vectors. The accuracy was above 99% on both the training and test sets. The sklearn Robust scaler was used in order to protect the model from outliers.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The `find_cars` function (in code block 7) scans across a predefined area (areas shown below) To avoid searching the entire image areas were chosen based on the cars' locations in of a handful of test images. Multiple areas were chosen to overlap to cover vehicles moving through the image vertically. Initallity I began with just 3 areas to scan, but that was clearly not enough. After adding more areas and confirming everything worked on the test images, I took a random sample of frames from the video. This allowed me to see exactly what locations the cars could end up in where I missed. I settled on the following areas. You can see the relative scale based on the height of the area.

![alt text][windows]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I searched on four scales using YUV 3-channel HOG features without spatially binned color or histograms of color in the feature vector. Color features did not improve the accuracy or even appear to help with overfitting much at all. Because of that and that they added quite a bit time, they were removed.  See the next section for images along with their heatmaps.

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here is a [link to my video result](./project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

![alt text][heatmap1]

![alt text][heatmap2]

![alt text][heatmap3]

![alt text][heatmap]

The boxes continue to contribute to the heatmap for half a second before they are discarded. This helped greatly calm the boxes' jerkiness. Originally I tried keeping the boxes for less time, but that didn't have as great of a smoothing effect.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The main issue I dealt with was deciding with features to use for the classifier. First I started simple, using just the Y channel of a YUV image to generate a HOG vector worked better than I expected. It had a lot of false positives however. I worked my way up to including all channels and the color histogram and spatial components as well. That proved to really be just noise, and only tripled the time needed to produce a video. Hard negative mining was the solution. Here are three example images. I ended up using more than 100, all the lane lines and trees.

![][negs]

The pipeline will likely fail on types vehicles it hasn't seen such as motorcycles. There is also a good chance that it would identify a picture of a car for example on a billboard or the side of a truck. Different lighting conditions also would pose a problem. At night it would be difficult to see darker colored cars, especially with headlights and taillights being much brighter. The pipeline also clearly has trouble with oncoming traffic.

A CNN might save from having to implement the sliding search and likely be faster to run instead of scanning the same pieces of the the image multiple times. Also trying to understand the speed (and a separate speed for oncoming traffic) could be used to vary the time boxes continue to contribute to the heatmap. Keeping boxes for half a second might be way too short for city driving. In general the boxes fitting could be improved with more training data. Another improvement could be remembering cars from frame to frame.