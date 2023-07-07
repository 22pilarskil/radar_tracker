Process for establishing heading and location

1. Researched basic template matching methods. None of these worked, none were resistant to size or orientation variance
   
2. Decided to first locate symbols using YOLOv5. Trained model after creating data generator, left with bounding boxes to analyze further
   
3. Used cv2's Canny method to generate edges for each bounding box's contents, used cv2 HoughLines method to extrapolate these edges into lines. Used FSRCNN to increase resolution of each bounding box prior to undergoing line detection to counter the pixelated nature of bounding box contents. Eventually, removed usage of FSRCNN due to computational complexity (time complexity of O(n) per frame with respect to number of detections), simply resized image to 16 times original size and played around with thresholds for Canny edge detection
   
4. Applied same transformations to reference tracker image resulting in two images with lines outlining general shape of bounding box contents, tried to compare line outputs directly using various algorithms (ORB to discern orientation, custom algorithms, etc.)
   
5. None of these worked, tried reducing lines down to a set number to make problem easier. For the tracker that I trained on, I decided that it could be defined by four lines very well. Experimented with various algorithms like MeanShift (no control over how many lines are output) and DBSCAN (hard to control in general), eventually determined that a simple clustering algorithm that combined lines within certain thresholds of each other was sufficient. 
   
6. Attempted to perform analysis with the angles of these lines. Compared angles of the four lines generated from the reference image with the four lines generated from the input image. Added a single degree to the angles array from 1 to 360 to determine which rotation of the four input image angles most closely matched the angles of the reference image. Wrote a function that for each rotation, first rearranged the elements of the input image's rotated angles array to minimize the absolute difference between itself and the angles from the reference image
   
7. This did not work well. Transitioned to the same process but performed on intersection points between the four lines. In order to effectively compare them, normalized the points around the center of the bounding box, presumed to also be the center of the track. This worked much better, presumably because points represent location and relative angle data whereas angles represent only angle data.

8. Intersection points still had some issues, tried assigning ids to each bounding box and tracking angle displacement of each new bounding box from frame directly preceding it. Between ZQPei's DeepSort and ABewley's Sort, Sort was much easier to use. Due to noise associated with bounding boxes, this did not work that well. Would work better if I could select bounding boxes further apart in time so that the noise is dampened by the distance between the two boxes, will experiment with this more if all else fails

9. Noticed that the images that fail are often those that capture edges associated with words or small features in the image, or small isolated regions. Experimented with thinning out cv2.Canny output to remove short or isolated edges while preserving longer, connected edges. This did not work for any methods I attempted, too hard to distinguish between noise and important features

10. Revisited point 5 because my method of reducing lines was highly dependent on the order of the lines input. Whichever lines were first in the array were the ones that were being compared to for the rest of the algorithm's runtime. Used KMeans clustering instead, worked much better, able to select lines that more accurately reflected the overall trend of the lines