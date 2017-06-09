

imagePath = '/home/sepideh/Documents/illuminChngeLrning/data/shadow/final_test';
imageType = '*.jpg';
images = dir([imagePath '/' imageType]);
nImages = length(images);

for i=1:nImages
    
    img = imread([imagePath '/' images(i).name]);
    if size(img, 1) > 1000 | size(img, 2) > 1000
        img = imresize(img, 0.5);
        imwrite(img, ['/home/sepideh/Documents/illuminChngeLrning/data/shadow/final_test/', images(i).name]);  
    end
    
end