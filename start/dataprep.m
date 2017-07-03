% reads the images and puts them in train and test set with paired
% and unpaired images.
% variable img has 3 parts:- im_a, im_b, index value to keep hold.

%%TO-DO:
% convert to rgb.
% .....

img = struct();
cd c:/Stuff/work/ANU/PersonID/data/VIPeR/cam_a/;
filelist_a = dir('*.bmp');
cd c:/Stuff/work/ANU/PersonID/data/VIPeR/cam_a/;
filelist_b = dir('*.bmp');
img = struct('img_a', imread(filelist_a(1).name), 'img_b', ...
             imread(filelist_b(1).name),'labels',1, 'index', 1)
val = randperm( 632,400);
%matching pairs
for i =  val
    %    disp(i)
    vindex = img(end).index+1;
    vstruct = struct('img_a', imread(filelist_a(i).name), 'img_b', ...
                     imread(filelist_b(i).name), 'labels',1,'index', vindex)
    img = [img, vstruct]
    vindex = vindex + 1;
    disp(vindex)
end


val = randperm( 632,400);
%non-matching pairs
for u =  val  
    vindex = img(end).index + 1;
    vunpaired = randperm(632,1);
    if(u == vunpaired)
        vunpaired = vunpaired + 1;
    end
    vstruct = struct('img_a', imread(filelist_a(u).name), 'img_b', ...
                     imread(filelist_b(vunpaired).name),'labels',-1, 'index', vindex)
    img = [img, vstruct]
    vindex = vindex + 1;
end


filename = img
save('vdata.mat','filename')