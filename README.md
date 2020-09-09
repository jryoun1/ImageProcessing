# ImageProcessing
### course in Brno University of Technology (FIT) 

## Implement Canny's Edge Detector 

There are 4 steps in Canny Edge Detector algorithm. <br>
1. Delete noise by apply Gaussian Blur
2. Find Sobel Edge using Sobel Edge Detector
3. Apply Non-maximum Suppression
4. Hysteresis Thresholding <br>

Implement each step and watch variations when giving different parameters, Gaussian deviation and upper and lower hysteresis thresholds. <br>

## How to Implement Project

### Implement in MacOS & Xcode

1. Download Homebrew <br>
```
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```
2. Download openCV using homebrew <br>
``` 
brew install opencv
```
3. Download pkg-config  <br>
```
# pkg-config download
brew install pkg-config
```
4. Extract Linker Flags <br>
```
# Linker Flags extraction
pkg-config --cflags --libs opencv4.2.0
```
After command 4, the result will be printed. Just keep the terminal and move to next step <br>

5. Start Xcode and make project in c++ language <br>
6. Go to **Build Setting** <br>

6-1. **Header Search Paths** setting <br>
```
/usr/local/Cellar/opencv/4.2.0_1/include/opencv4
```
6-2. **Library Search Paths** setting <br>
```
/usr/local/Cellar/opencv/4.2.0_1/lib
```
6-3. **Other Linker Flags** setting <br>
```
copy and pasted the result of command 4 (contents in terminal)
```

7. Write input image filePath in the main function. <br>
8. Enter parameter of standard deviation for Gaussian Filter (e.g 0.5/1/3/5) <br>
9. Enter parameter of low&high threshold value (e.g 20 40/20 80/40 60/40 80) <br>


## Result 
Result is Original image, Gaussian Blurred image, Sobel Filter applied image, Non-Maximum Suppression applied image and Final Image <br>
![Example1](https://i.imgur.com/xxsN7c2.png) <br>
Input : Standard Deviation = 1, Low & High threshold value = 20 40 <br>
![Example2](https://i.imgur.com/iH5QN8g.png) <br>
Input : Standard Deviation = 1, Low & High threshold value = 20 80 <br>

