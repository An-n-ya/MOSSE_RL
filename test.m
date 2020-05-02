clc
clear
img1 = imread('cat.jpg');
img2 = imread('cat0.jpg');

im1 = double(rgb2gray(img1));
im2 = double(rgb2gray(img2));

im1 = im1(1:2:size(im1,1),1:2:size(im1,2))-mean(mean(im1));
im1 = im1(1:2:size(im1,1),1:2:size(im1,2))-mean(mean(im1));

t = imfilter(im1,im2,'corr');
mesh(t)