function [NextObs,Reward,IsDone,LoggedSignals] = CF_Step(Action,LoggedSignals)
% Custom step function to construct cart-pole environment for the function
% name case.
%
% This function applies the given action to the environment and evaluates
% the system dynamics for one simulation step.
% Define the environment constants.
% Acceleration due to gravity in m/s^2
%get images from source directory


frame = LoggedSignals.State{1} + 1;
Ai = LoggedSignals.State{2};
Bi = LoggedSignals.State{3};
seq = LoggedSignals.State{4};
rect = LoggedSignals.State{5};
G = LoggedSignals.State{6};

no_fram  = seq.endFrame - seq.startFrame;

base_path  = 'D:/data_seq';
video_path = [base_path '/' seq.name];

s_frames = seq.s_frames;
% MOSSE online training regimen
%eta = 0.25;
    img =  imread([video_path '/img/' s_frames{frame}]);
    im = img;
    if (size(img,3) == 3)
        img = rgb2gray(img);
    end
        Hi = Ai./Bi;
        wsize    = [seq.init_rect(1,4), seq.init_rect(1,3)];
        pos = [rect(1,2), rect(1,1)] + floor(wsize/2);
        tsz = [200,200];
        fi = get_pixels(img,pos,round(tsz),tsz);
%         fi = imcrop(img, rect);     
        %[(rect(4)+1), (rect(3)+1)] = size(fi);
        %fi = preprocess(imresize(fi, [(rect(4)+1) (rect(3)+1)])); 
        fi = preprocess(fi); 
        try
            gi = uint8(255*mat2gray(ifft2(Hi.*fft2(fi))));
        catch 
            disp(int2str(size(Hi)))
            disp(int2str(size(fi)))
        end
        maxval = max(gi(:));
        [P, Q] = find(gi == maxval);
        dx = mean(P)-tsz(2)/2;
        dy = mean(Q)-tsz(1)/2;
      
        rect(1) = rect(1)+dy;
        rect(2) = rect(2)+dx;
        pos = [rect(1,2), rect(1,1)] + floor(wsize/2);
        fi = get_pixels(img,pos,round(tsz),tsz);
%         fi = imcrop(img, rect); 
        fi = preprocess(fi);
        Ai = Action.*(G.*conj(fft2(fi))) + (1-Action).*Ai;
        Bi = Action.*(fft2(fi).*conj(fft2(fi))) + (1-Action).*Bi;
        NextObs = double(gi);
        
        rect_anno = dlmread(['./anno/' seq.name '.txt']);
        score = calcRectInt(rect,rect_anno(frame,:));
        if score < 0.2
            Reward = -100;
        else
            Reward = 1;
        end
        
        if score < 0.2 
            IsDone = true;
        elseif frame == no_fram
            IsDone = true;
            Reward = 100;
        else
            IsDone = false;
        end
            
%     % visualization
%     text_str = ['Frame: ' num2str(frame)];
%     box_color = 'green';
%     position=[1 1];
%     result = insertText(im, position,text_str,'FontSize',15,'BoxColor',...
%                      box_color,'BoxOpacity',0.4,'TextColor','white');
%     result = insertShape(result, 'Rectangle', rect, 'LineWidth', 3);
%     %imwrite(result, ['results_' dataset num2str(frame, '/%04i.jpg')]);
% 	imshow(result);
%     drawnow;
%     %rect
%     

    
    LoggedSignals.State = {frame,Ai,Bi,seq,rect,G};

end

