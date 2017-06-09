
close all, clear;clc;
addpath(genpath('/home/sepideh/Documents/illuminChngeLrning/caffe_8dec16'));
caffe.reset_all();
caffe.set_mode_gpu();
caffe.set_device(0);

% model
cnnmodel = './deploy.prototxt';
cnnweights = './train_iter_100000.caffemodel';

net = caffe.Net(cnnmodel, cnnweights, 'test'); % create net and load weights

imagePath = '/home/sepideh/Documents/illuminChngeLrning/data/shadow/final_test';
imageType = '*.jpg';
images = dir([imagePath '/' imageType]);
nImages = length(images);

for i=1:nImages
    
    img = imread([imagePath '/' images(i).name]);
%     figure(1), imshow(img)
    imgSize = [size(img,2), size(img,1)];
    mean_data = [];
    mean_data(:,:,3) = repmat(112.63,imgSize); % B
    mean_data(:,:,2) = repmat(123.21,imgSize); % G
    mean_data(:,:,1) = repmat(126.61,imgSize); % R
    mean_data = single(mean_data);
    
    im_data = img(:, :, [3, 2, 1]);  % permute channels from RGB to BGR
    im_data = permute(im_data, [2, 1, 3]);  % flip width and height
    im_data = single(im_data);  % convert from uint8 to single
    im_data = imresize(im_data, imgSize, 'bilinear');  % resize im_data
    im_data = im_data - mean_data;  % subtract mean_data (already in W x H x C, BGR)

    net.blobs('data').reshape([imgSize 3 1]); % reshape blob 'data'
    net.reshape();
    
    % forward
    scores = net.forward({im_data});  
    scores = scores{1};
    
    %% binary map
    res = abs(scores(:,:,2)) + abs(scores(:,:,4)) + abs(scores(:,:,6));
    res1= abs(scores(:,:,1)) + abs(scores(:,:,3)) + abs(scores(:,:,5));
    predict = zeros(size(res));
    predict(res1(:,:)<res(:,:))=1;
    figure(2), subplot(1,2,1),imshow(predict');
   
    %%
    res = sum(scores, 3);
    predict = mat2gray(res);
    figure(2), subplot(1,2,2), imshow(mat2gray(predict'))
    %%
%     for ll=1:21
%         figure(20), subplot(4,6,ll), imagesc(scores(:,:,ll)');
%     end
%     imagesc(predict)
%     imwrite(predict', ['/home/sepideh/Documents/illuminChngeLrning/data/shadow/predictedLabels_final_test/', images(i).name(1:end-4), '_predict.png']);  
%     imagesc(scores(:,:,1)')
%     name = strcat(images(i).name, '_scores.mat');
%     save(name, 'scores')
%     imwrite(img, images(i).name);

end

