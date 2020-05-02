%get images from source directory

%�˴��������ڵõ�ͼƬ�������ڵ�ַ
datadir = 'D:/data_seq/';
dataset = 'faceocc1';
path = [datadir dataset];
img_path = [path '/img/'];
D = dir([img_path, '*.jpg']);%��img_path�µ�����jpg��׺�ļ��ĵ�ַ����D��

seq_len = length(D(not([D.isdir])));%�õ�ͼƬ����
if exist([img_path num2str(1, '%04i.jpg')], 'file')
    img_files = num2str((1:seq_len)', [img_path '%04i.jpg']);
else
    error('No image files found in the directory.');
end

% select target from first frame
im = imread(img_files(1,:));
f = figure('Name', 'Select object to track'); 
imshow(im);
rect = getrect;
close(f); clear f;
center = [rect(2)+rect(4)/2 rect(1)+rect(3)/2];

% plot gaussian
sigma = 100;
gsize = size(im);%��ȡͼ��ߴ�
[R,C] = ndgrid(1:gsize(1), 1:gsize(2));
%��������R��C������m��n��
g = gaussC(R,C, sigma, center);%ͨ��R��C����size��ͬ��im�ĸ�˹�˲�����
g = mat2gray(g);

% randomly warp original image to create training set
if (size(im,3) == 3) 
    img = rgb2gray(im); 
end
img = imcrop(img, rect);
g = imcrop(g, rect);
G = fft2(g);
%����˹�˲������任��Ƶ��
height = size(g,1);
width = size(g,2);
fi = preprocess(imresize(img, [height width]));%imresize(img, [height width])��ͼƬ�������˲����Ĵ�С
Ai = (G.*conj(fft2(fi)));
Bi = (fft2(fi).*conj(fft2(fi)));
N = 128;

for i = 1:N
    fi = preprocess(rand_warp(img));
    Ai = Ai + (G.*conj(fft2(fi)));
    Bi = Bi + (fft2(fi).*conj(fft2(fi)));
end

% MOSSE online training regimen
eta = 0.25;
fig = figure('Name', 'MOSSE');
t = figure;
mkdir(['results_' dataset]);
for i = 1:size(img_files, 1)
    img = imread(img_files(i,:));
    im = img;
    if (size(img,3) == 3)
        img = rgb2gray(img);
    end
    if (i == 1)
        Ai = eta.*Ai;
        Bi = eta.*Bi;
    else
        Hi = Ai./Bi;
        fi = imcrop(img, rect);            
        fi = preprocess(imresize(fi, [height width])); 
        gi = uint8(255*mat2gray(ifft2(Hi.*fft2(fi))));
        maxval = max(gi(:));
        [P, Q] = find(gi == maxval);
        dx = mean(P)-height/2;
        dy = mean(Q)-width/2;
      
        rect = [rect(1)+dy rect(2)+dx width height];
        fi = imcrop(img, rect); 
        fi = preprocess(imresize(fi, [height width]));
        Ai = eta.*(G.*conj(fft2(fi))) + (1-eta).*Ai;
        Bi = eta.*(fft2(fi).*conj(fft2(fi))) + (1-eta).*Bi;
    end
    % visualization
    text_str = ['Frame: ' num2str(i)];
    box_color = 'green';
    position=[1 1];
    result = insertText(im, position,text_str,'FontSize',15,'BoxColor',...
                     box_color,'BoxOpacity',0.4,'TextColor','white');
    result = insertShape(result, 'Rectangle', rect, 'LineWidth', 3);
    imwrite(result, ['results_' dataset num2str(i, '/%04i.jpg')]);
	imshow(result);
    drawnow;
    rect
end
