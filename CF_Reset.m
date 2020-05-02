function [InitialObservation, LoggedSignal] = CF_Reset()
% Reset function to place custom cart-pole environment into a random
% initial state.
base_path  = 'D:/data_seq';
seqname = {'faceocc1','faceocc2'};
[~,sz] = size(seqname);
choose_seq = seqname{ceil(rand * sz)};
video_path = [base_path '/' choose_seq];
[seq, ground_truth] = load_video_info(video_path,choose_seq);
seq.startFrame = 1;
seq.endFrame = seq.len;
seq.ground_truth=ground_truth;

s_frames = seq.s_frames;
params.no_fram  = seq.endFrame - seq.startFrame + 1;
params.seq_st_frame = seq.startFrame;
params.seq_en_frame = seq.endFrame;
params.ground_truth=seq.ground_truth;

%rect_anno = dlmread(['./anno/' seq.name '.txt']);

% select target from first frame
im = imread([video_path '/img/' s_frames{1}]);
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
img = imcrop(img, seq.init_rect);
g = imcrop(g, seq.init_rect);
G = fft2(g);
%将高斯滤波函数变换到频域
height = size(g,1);
width = size(g,2);
fi = preprocess(imresize(img, [height width]));%imresize(img, [height width])将图片调整成滤波器的大小
Ai = (G.*conj(fft2(fi)));
Bi = (fft2(fi).*conj(fft2(fi)));
N = 128;
for i = 1:N
    fi = preprocess(rand_warp(img));
    Ai = Ai + (G.*conj(fft2(fi)));
    Bi = Bi + (fft2(fi).*conj(fft2(fi)));
end
% Return initial environment state variables as logged signals.
LoggedSignal.State = {1,Ai,Bi,seq,seq.init_rect,G};
InitialObservation = double(g);
end