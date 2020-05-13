function [InitialObservation, LoggedSignal] = CF_Reset()
% Reset function to place custom cart-pole environment into a random
% initial state.
base_path  = 'D:/data_seq';
%seqname = {'Bird1','faceocc1','faceocc2','Bird2','Bolt2','Box','Car1','Car2','Car24',};
seqname = {'Bird1','Bird2','Bolt2','Box','Car1','Car2','Car24','ClifBar','Coupon','Crowds',...
    'Dancer','Dancer2','Diving','Dog','DragonBaby','Girl2','Gym','Human2','Human3',...
    'Human4','Human5','Human6','Human7','Human8','Human9','Jump','KiteSurf','Man',...
    'Panda','RedTeam','Rubik','Skater','Skater2','Skating1','Skating2','Surfer',...
    'Toy','Trans','Twinnings','Vase','soccer','matrix','deer','skating1',...
    'shaking','singer1','singer2','carDark','car4','david','david2','sylvester',...
    'trellis','fish','mhyang','coke','bolt','boy','dudek','crossing','couple',...
    'football1','doll','girl','walking2','walking',...
    'fleetFace','freeman1','freeman3','freeman4','david3','jumping','carScale',...
    'skiing','dog1','suv','motorRolling','mountainBike','lemming','liquor',...
    'woman','faceocc1','faceocc2','basketball','football','subway','tiger1',...
    'tiger2','BlurCar1','BlurCar2','BlurCar3','BlurCar4','BlurBody','BlurFace',...
    'Board','BlurOwl'};
[~,sz] = size(seqname);
choose_seq = seqname{ceil(rand * sz)};
video_path = [base_path '/' choose_seq];
[seq, ground_truth] = load_video_info(video_path,choose_seq);
if seq.len > 100
    seq.startFrame = ceil(rand*(seq.len-100));
    seq.endFrame = seq.startFrame+100;
else
    seq.startFrame = 1;
    seq.endFrame = seq.len;
end
seq.ground_truth=ground_truth;
seq.init_rect = ground_truth(seq.startFrame,:);

s_frames = seq.s_frames;

%rect_anno = dlmread(['./anno/' seq.name '.txt']);

% select target from first frame
im = imread([video_path '/img/' s_frames{seq.startFrame}]);
center = [seq.init_rect(2)+seq.init_rect(4)/2 seq.init_rect(1)+seq.init_rect(3)/2];

% plot gaussian
sigma = 100;
gsize = size(im);%获取图像尺寸
[R,C] = ndgrid(1:gsize(1), 1:gsize(2));
%两个矩阵R和C，都是m行n列
g = gaussC(R,C, sigma, center);%通过R和C产生size等同于im的高斯滤波函数
g = mat2gray(g);

% randomly warp original image to create training set
if (size(im,3) == 3) 
    img = rgb2gray(im); 
else
    img = im;
end
wsize    = [seq.init_rect(1,4), seq.init_rect(1,3)];
init_pos = [seq.init_rect(1,2), seq.init_rect(1,1)] + floor(wsize/2);
tsz = [200,200];
img = get_pixels(img,init_pos,round(tsz),tsz);
g = get_pixels(g,init_pos,round(tsz),tsz);
% img = imcrop(img, seq.init_rect);
% g = imcrop(g, seq.init_rect);
G = fft2(g);
%将高斯滤波函数变换到频域
height = size(g,1);
width = size(g,2);
fi = preprocess(img);%imresize(img, [height width])将图片调整成滤波器的大小
Ai = (G.*conj(fft2(fi)));
Bi = (fft2(fi).*conj(fft2(fi)));
N = 128;
for i = 1:N
    fi = preprocess(rand_warp(img));
    Ai = Ai + (G.*conj(fft2(fi)));
    Bi = Bi + (fft2(fi).*conj(fft2(fi)));
end
% Return initial environment state variables as logged signals.
LoggedSignal.State = {seq.startFrame,Ai,Bi,seq,seq.init_rect,G};
InitialObservation = double(g);
end