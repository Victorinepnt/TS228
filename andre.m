clear;
close all;
clc;

% Ouvrez le fichier de données d'images
fid = fopen('train-images-idx3-ubyte','r');

% Lisez les en-têtes du fichier
magic = fread(fid,1,'int32',0,'ieee-be');
numImages = fread(fid,1,'int32',0,'ieee-be');
numRows = fread(fid,1,'int32',0,'ieee-be');
numCols = fread(fid,1,'int32',0,'ieee-be');

% Lisez les données d'images
images = fread(fid,inf,'unsigned char');
images = reshape(images,numCols,numRows,numImages);
images = permute(images,[2 1 3]);

% Fermez le fichier
fclose(fid);
