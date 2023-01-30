clear;
close all;
clc;

tic;
%addpath('hw3data');
%addpath('hw3data/training set');
[trainImgs,trainLabels] = readMNIST('train-images-idx3-ubyte','train-labels-idx1-ubyte',20000,0);
%addpath('hw3data/test set');
[testImgs,testLabels] = readMNIST('t10k-images-idx3-ubyte','t10k-labels-idx1-ubyte',10000,0);
toc;