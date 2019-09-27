import cv2	#conda install -c menpo opencv

original = cv2.imread("images/flower.jpg")
image_to_compare = cv2.imread("images/flower_hue.jpg")

# 1) Check if 2 images are equals
if original.shape == image_to_compare.shape:
    #print("The images have same size and channels")
    difference = cv2.subtract(original, image_to_compare)
    b, g, r = cv2.split(difference)
    if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
        print("The images are completely equal")
    # 2) Check for similarities between the 2 images
    else:
        print("Images are not equal")
        sift = cv2.xfeatures2d.SIFT_create()
        kp_1, desc_1 = sift.detectAndCompute(original, None)
        kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)
        
        index_params = dict(algorithm=0, trees=5)
        search_params = dict()
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(desc_1, desc_2, k=2)
        
        good_points = []
        #bad_points = []
        ratio = 0.6
        for m, n in matches:
            if m.distance < ratio*n.distance:
                good_points.append(m)
        print(len(good_points))    
        if len(good_points)>50:
            print("Images are similar")
        
        result = cv2.drawMatches(original, kp_1, image_to_compare, kp_2, good_points, None)
        #print("done")
        #cv2.imshow("result", cv2.resize(result, None, fx=0.4, fy=0.4))

cv2.imshow("Original",cv2.resize(original, None, fx=0.4, fy=0.4))
cv2.imshow("image_to_compare", cv2.resize(image_to_compare, None, fx=0.4, fy=0.4))
cv2.waitKey(0)
cv2.destroyAllWindows()
