#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <vector>

using namespace std;
using namespace cv;

class canny {
private:
    Mat img; //Original Image
    Mat gaussianFiltered; // Gradient
    Mat sobelFiltered; //Sobel Filtered
    Mat anglemap; //Angle Map
    Mat nonMax; // Non-maxima supp.
    Mat thres; //Double threshold and edge tracking by hysteresis
public:
    canny(String,double,double,double); //Constructor
    vector<vector<double>> GaussianFilter(int, int, double);
    Mat applyFilter(Mat, vector<vector<double>>); //Apply gaussian filter
    Mat sobel(); //Sobel filtering
    Mat nonMaxSupp(); //Non-maxima suppression
    Mat threshold(Mat, int, int); //Double threshold and finalize picture
};
