clear
data=load('res_girl.mat','-mat');
masks=data.segmentation;
for n = 1:21
    K = mat2gray(masks{n,1});
    baseFileName = sprintf('Image0%d.png', n);
    imwrite(K,baseFileName);
end