#include <iostream>
#include <cmath>
#include <vector>
#include "cannyEdgeDetector.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

int main(){
    //Filepath of input image
    cv::String filePath ="/Users/yeon/Desktop/ImageProcessing/Sample Images/budapest.jpeg";
    
    //value for parameters standard deviation, low&high threshold
    double stdv=0, low=0, high=0;
    
    cout<< "Enter standard deviation for Gaussian Filter : " ;
    cin >> stdv;
    cout << "Enter low and high threshold value : ";
    cin >> low >> high ;
    
    canny cannyEdgeDetector(filePath, stdv, low, high);
    return 0;
}

canny::canny(String filename, double stdv, double low, double high){
    img = imread(filename);
    if (!img.data) {// Check for invalid input
        cout << "Could not open or find the image" << std::endl;
    }
    else{
    vector<vector<double>> filter = GaussianFilter(5, 5, stdv);
        
    //Print filter
    cout<< "Print Gaussian filter" << endl;
    for (int i = 0; i<filter.size(); i++){
        for (int j = 0; j<filter[i].size(); j++){
            cout << filter[i][j] << " ";
        }
        cout<<endl;
    }
   
    cv::Mat image_gray;
    cv::cvtColor(img,image_gray,COLOR_RGB2GRAY); //Grayscale the image
    gaussianFiltered = Mat(applyFilter(image_gray, filter)); //Applied Gaussian Filter
    sobelFiltered = Mat(sobel()); //Applied Sobel Filter
    nonMax = Mat(nonMaxSupp()); //Applied Non-Maxima Suppression
    thres = Mat(threshold(nonMax, low, high)); //Double Threshold and edge tracking by hysteresis
    
    namedWindow("Original");
    namedWindow("Gaussian Blur");
    namedWindow("Sobel Filtered");
    namedWindow("Non-Maxima Supp.");
    namedWindow("Final");

    imshow("Original", img);
    imshow("Gaussian Blur", gaussianFiltered);
    imshow("Sobel Filtered", sobelFiltered);
    imshow("Non-Maxima Supp.", nonMax);
    imshow("Final", thres);
        
    waitKey(0);
    }
}

vector<vector<double>> canny::GaussianFilter(int row, int column, double stdv){
    vector<vector<double>> GaussianFilter;
    
    //initiate 5x5 kernel with 0
    for (int i = 0; i < row; i++){
        vector<double> col;
        for (int j = 0; j < column; j++){
            col.push_back(0);
        }
        GaussianFilter.push_back(col);
    }

    double r, s = 2.0 * stdv * stdv; 
    double sum = 0.0; // sum is for normalization

    //generating 5x5 gaussian kernel
    for (int x = - row/2; x <= row/2; x++){
        for (int y = -column/2; y <= column/2; y++){
            r = sqrt(x*x + y*y);
            GaussianFilter[x + row/2][y + column/2] = (exp(-(r*r) / s)) / (M_PI * s);
            sum += GaussianFilter[x + row/2][y + column/2];
        }
    }
    
    double checkSum = 0; // checkSum is for checking the sum of all cols&rows
    
    //Loop to normalize the filter
    for (int i = 0; i < row; i++)
        for (int j = 0; j < column; j++)
            GaussianFilter[i][j] /= sum;

    //Printout the sum of all kernel to check if the sum of all cols & rows is 1
    for (int i =0; i< GaussianFilter.size();i++){
        for(int j = 0; j<GaussianFilter[i].size();j++){
            checkSum +=GaussianFilter[i][j];
        }
    }
    cout << "Sum of all cols and rows : "<<checkSum <<endl;
    return GaussianFilter;
}

Mat canny::applyFilter(Mat img_in, vector<vector<double>> filter_in){
    int size = (int)filter_in.size()/2;
    
    Mat appliedImg = Mat(img_in.rows - 2*size, img_in.cols - 2*size, CV_8UC1);
    
    //apply gaussian filter(filter_in) to source img(img_in)
    for (int i = size; i < img_in.rows - size; i++){
        for (int j = size; j < img_in.cols - size; j++){
            double sum = 0;
            
            for (int x = 0; x < filter_in.size(); x++)
                for (int y = 0; y < filter_in.size(); y++){
                    sum += filter_in[x][y] * (double)(img_in.at<uchar>(i + x - size, j + y - size));
                }
            appliedImg.at<uchar>(i-size, j-size) = sum;
        }
    }
    return appliedImg;
}

Mat canny::sobel(){
    //Sobel X Filter
    vector<vector<double>> xFilter;
    vector<double> x1 ={-1.0, 0, 1.0};
    vector<double> x2 ={-2.0, 0, 2.0};
    vector<double> x3 ={-1.0, 0, 1.0};
    xFilter.push_back(x1);
    xFilter.push_back(x2);
    xFilter.push_back(x3);
    
    //Sobel Y Filter
    vector<vector<double>> yFilter;
    vector<double> y1 = {1.0, 2.0, 1.0};
    vector<double> y2 = {0, 0, 0};
    vector<double> y3 = {-1.0, -2.0, -1.0};
    yFilter.push_back(y1);
    yFilter.push_back(y2);
    yFilter.push_back(y3);
    
    //Limit Size
    int size = (int)xFilter.size()/2;

    Mat appliedImg = Mat(gaussianFiltered.rows - 2*size, gaussianFiltered.cols - 2*size, CV_8UC1);
    anglemap = Mat(gaussianFiltered.rows - 2*size, gaussianFiltered.cols - 2*size, CV_32FC1); 
    
    //apply sobel mask & get direction and norm
    for (int i = size; i < gaussianFiltered.rows - size; i++){
        for (int j = size; j < gaussianFiltered.cols - size; j++){
            double x_sum = 0;
            double y_sum = 0;
            
            for (int x = 0; x < xFilter.size(); x++)
                for (int y = 0; y < xFilter.size(); y++){
                    //Sobel_X Filter Value
                    x_sum += xFilter[x][y] * (double)(gaussianFiltered.at<uchar>(i + x - size, j + y - size));
                    //Sobel_Y Filter Value
                    y_sum += yFilter[x][y] * (double)(gaussianFiltered.at<uchar>(i + x - size, j + y - size));
                }
            double x_sumsq = x_sum * x_sum;
            double y_sumsq = y_sum * y_sum;

            double result = sqrt(x_sumsq + y_sumsq); //calculate gradient strength
            
            if(result > 255) //Unsigned Char Fix
                result =255;
            appliedImg.at<uchar>(i-size, j-size) = result;
 
            if(x_sum==0) //Arctan Fix
                anglemap.at<float>(i-size, j-size) = 90;
            anglemap.at<float>(i-size, j-size) = atan(y_sum/x_sum);
        }
    }
    return appliedImg;
}

Mat canny::nonMaxSupp(){
    Mat nonMaxSupped = Mat(sobelFiltered.rows-2, sobelFiltered.cols-2, CV_8UC1);
    for (int i=1; i< sobelFiltered.rows - 1; i++) {
        for (int j=1; j<sobelFiltered.cols - 1; j++) {
            float Tangent = anglemap.at<float>(i,j);

            nonMaxSupped.at<uchar>(i-1, j-1) = sobelFiltered.at<uchar>(i,j);
            //blue section
            if (((-22.5 < Tangent) && (Tangent <= 22.5)) || ((157.5 < Tangent) && (Tangent <= -157.5))){
                if ((sobelFiltered.at<uchar>(i,j) < sobelFiltered.at<uchar>(i,j+1)) || (sobelFiltered.at<uchar>(i,j) < sobelFiltered.at<uchar>(i,j-1)))
                    nonMaxSupped.at<uchar>(i-1, j-1) = 0;
            }
            //orange section
            if (((-157.5 < Tangent) && (Tangent <= -112.5)) || ((22.5 < Tangent) && (Tangent <= 67.5))){
                if ((sobelFiltered.at<uchar>(i,j) < sobelFiltered.at<uchar>(i+1,j+1)) || (sobelFiltered.at<uchar>(i,j) < sobelFiltered.at<uchar>(i-1,j-1)))
                    nonMaxSupped.at<uchar>(i-1, j-1) = 0;
            }
            //red section
            if (((-112.5 < Tangent) && (Tangent <= -67.5)) || ((67.5 < Tangent) && (Tangent <= 112.5))){
                if ((sobelFiltered.at<uchar>(i,j) < sobelFiltered.at<uchar>(i+1,j)) || (sobelFiltered.at<uchar>(i,j) < sobelFiltered.at<uchar>(i-1,j)))
                    nonMaxSupped.at<uchar>(i-1, j-1) = 0;
            }
            //green section
            if (((-67.5 < Tangent) && (Tangent <= -22.5)) || ((112.5 < Tangent) && (Tangent <= 157.5))){
                if ((sobelFiltered.at<uchar>(i,j) < sobelFiltered.at<uchar>(i-1,j+1)) || (sobelFiltered.at<uchar>(i,j) < sobelFiltered.at<uchar>(i+1,j-1)))
                    nonMaxSupped.at<uchar>(i-1, j-1) = 0;
            }
        }
    }
    return nonMaxSupped;
}

Mat canny::threshold(Mat imgin,int low, int high){
    //limitation for low& high threshold
    if(low > 255) low = 255;
    if(high > 255) high = 255;
    
    Mat EdgeMat = Mat(imgin.rows, imgin.cols, imgin.type());
    
    for (int i=0; i<imgin.rows; i++){
        for (int j = 0; j<imgin.cols; j++){
            EdgeMat.at<uchar>(i,j) = imgin.at<uchar>(i,j);
            if(EdgeMat.at<uchar>(i,j) > high) //if edge pixel is higher than high threshold => strong edge
                EdgeMat.at<uchar>(i,j) = 255;
            else if(EdgeMat.at<uchar>(i,j) < low) //if edge pixel is lower than low threshold => noise
                EdgeMat.at<uchar>(i,j) = 0;
            else{                               //edge pixel is between high and low threshold => weak edge
                bool anyHigh = false;
                bool anyBetween = false;
                for (int x=i-1; x < i+2; x++){
                    for (int y = j-1; y<j+2; y++){
                        if(x <= 0 || y <= 0 || EdgeMat.rows || y > EdgeMat.cols) //except out of bounds
                            continue;
                        else{ // edge tracking by hysteresis
                            if(EdgeMat.at<uchar>(x,y) > high){
                                EdgeMat.at<uchar>(i,j) = 255;
                                anyHigh = true;
                                break;
                            }
                            else if(EdgeMat.at<uchar>(x,y) <= high && EdgeMat.at<uchar>(x,y) >= low)
                                anyBetween = true;
                        }
                    }
                    if(anyHigh)
                        break;
                }
                if(!anyHigh && anyBetween)
                    for (int x=i-2; x < i+3; x++){
                        for (int y = j-1; y<j+3; y++){
                            if(x < 0 || y < 0 || x > EdgeMat.rows || y > EdgeMat.cols) //Out of bounds
                                continue;
                            else{ //check if there's any associativity with eight-way pixels
                                if(EdgeMat.at<uchar>(x,y) > high){
                                    EdgeMat.at<uchar>(i,j) = 255;
                                    anyHigh = true;
                                    break;
                                }
                            }
                        }
                        if(anyHigh) //associativity with eight-way pixels => edge
                            break;
                    }
                if(!anyHigh) //No associativity with eight-way pixels => noise
                    EdgeMat.at<uchar>(i,j) = 0;
            }
        }
    }
    return EdgeMat;
}
