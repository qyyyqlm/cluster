clc;
clear;
for TTT = 1:30
train_set_size = TTT;
descriptor_data = [];
clustering_size = 500;
knn_size = 3;
edge_thresh = 50;
%all_information = [];
%%

% the index of motor class is 1
motor_train = cell(1,train_set_size);
motor_d = cell(4,train_set_size);
for i = 1:train_set_size
  m = i; %randperm(500);
  n = m(1);
  if n < 10
    motor_train{1,i} = imread(['./motorbikes_train/img00' num2str(n) '.jpg']);
  else if n >= 10 & n < 100
          motor_train{1,i} = imread(['./motorbikes_train/img0' num2str(n) '.jpg']);
      else
          motor_train{1,i} = imread(['./motorbikes_train/img' num2str(n) '.jpg']);
      end
  end
  [f,d] = vl_sift(single(rgb2gray(motor_train{1,i})));
  descriptor_data = [descriptor_data d];
  motor_d{1,i} = d;
  [row column] = size(d);
  motor_d{2,i} = column;
  motor_d{4,i} = 1;
end

faces_train = cell(1,train_set_size);
faces_d = cell(4,train_set_size);
for i = 1:train_set_size  % for each image in the train set
    m = randperm(400); 
    n = m(1);    % get a random # [1,400]
    if n < 10
    faces_train{1,i} = imread(['./faces_train/img00' num2str(n) '.jpg']);
  else if n >= 10 & n < 100
          faces_train{1,i} = imread(['./faces_train/img0' num2str(n) '.jpg']);
      else
          faces_train{1,i} = imread(['./faces_train/img' num2str(n) '.jpg']);
      end
    end  % get the face train images(train_set_size#) randonly from 
  [f,d] = vl_sift(single(rgb2gray(faces_train{1,i}))); % grayscale and single precision 
  descriptor_data = [descriptor_data d]; 
  faces_d{1,i} = d;
  [row column] = size(d);
  faces_d{2,i} = column;
  faces_d{4,i} = 2;
end

cars_train = cell(1,train_set_size);
cars_d = cell(4,train_set_size);
for i = 1:train_set_size
  m = randperm(310);
  n = m(1);
  if n < 10
    cars_train{1,i} = imread(['./cars_train/img00' num2str(n) '.jpg']);
  else if n >= 10 & n < 100
          cars_train{1,i} = imread(['./cars_train/img0' num2str(n) '.jpg']);
      else
          cars_train{1,i} = imread(['./cars_train/img' num2str(n) '.jpg']);
      end
  end
  [f,d] = vl_sift(single(rgb2gray(cars_train{1,i})));
  descriptor_data = [descriptor_data d];
  cars_d{1,i} = d;
  [row column] = size(d);
  cars_d{2,i} = column;
  cars_d{4,i} = 3;
end

airplanes_train = cell(1,train_set_size);
airplanes_d = cell(4,train_set_size);
for i = 1:train_set_size
    m = randperm(500);
    n = m(1);
    if n < 10
    airplanes_train{1,i} = imread(['./airplanes_train/img00' num2str(n) '.jpg']);
  else if n >= 10 & n < 100
          airplanes_train{1,i} = imread(['./airplanes_train/img0' num2str(n) '.jpg']);
      else
          airplanes_train{1,i} = imread(['./airplanes_train/img' num2str(n) '.jpg']);
      end
  end
  [f,d] = vl_sift(single(rgb2gray(airplanes_train{1,i})));
  descriptor_data = [descriptor_data d];
  airplanes_d{1,i} = d;
  [row column] = size(d);
  airplanes_d{2,i} = column;
  airplanes_d{4,i} = 4;
end
% clear motor_train;
% clear faces_train;
% clear airplanes_train;
% clear cars_train;
all = cat(2,motor_d,faces_d,cars_d,airplanes_d);

%% K-means clustering
[centers, assignments] = vl_kmeans(single(descriptor_data), clustering_size);

%Get Bag-of-words histograms
counter_des = 1;
for i = 1:train_set_size*4
    col = all{2,i};
    counter_des_final = col + counter_des - 1;
    assign = assignments(counter_des:counter_des_final);
    xcenter = 1:clustering_size;
    H = hist(double(assign),xcenter);
    all{3,i} = H ./ col;
    counter_des = counter_des_final+1;
end

% for i = 1:train_set_size*4
%     col = all{2,i};
%     
% end

%%Loading Query Image

for images = 1:100
if images <= 25
     I_q = imread(['./motorbikes_test/img0' num2str(images + 10) '.jpg']);
  else if images >= 26 & images <= 50
          I_q = imread(['./faces_test/img0' num2str(images - 25 + 10) '.jpg']);
      else if images >= 51 & images <= 75
              I_q = imread(['./cars_test/img0' num2str(images - 50 + 10) '.jpg']);
          else
              I_q = imread(['./airplanes_test/img0' num2str(images- 75 + 10) '.jpg']);
          end
      end
  end
[f_q,d_q] = vl_sift(single(rgb2gray(I_q)));
[row,col] = size(d_q);
assign = zeros(1,col);
%assign2 = zeros(1,col);
for i = 1:col
%      for j = 1:clustering_size
%          dis(j) = norm(centers(:,j) - single(d_q(:,i)),2);
%      end
%      [v,pos] = min(dis);
%      assign(i) = pos;
     [~,k] = min(vl_alldist(single(d_q(:,i)), centers));
     assign(i) = k;
end
 xcenter = 1:clustering_size;
H = hist(double(assign),xcenter)./ col;

for i = 1:train_set_size*4
    dist(1,i) = sum((H - all{3,i}).^2 ./ (H + all{3,i}+ 0.0001) ); %norm(H-all{3,i});  %%
    dist(2,i) = all{4,i};
end

check = sortrows(dist',1);
Output = check(1:knn_size,2);

vote = zeros(1,4);
for i = 1:knn_size
    switch Output(i);
        case 1
            vote(1) = vote(1) + 1;
        case 2
            vote(2) = vote(2) + 1;
        case 3
            vote(3) = vote(3) + 1;
        case 4
            vote(4) = vote(4) + 1;
    end
end

[val,cl] = max(vote);
class(images) = cl
end

false = 0;
for i = 1:100
    if (i <= 25)
        if class(i) ~= 1
            false = false + 1;
        end
    else if (i > 25 & i <= 50)
            if class(i) ~=2
                false = false + 1;
            end
        else if (i > 50 & i <= 75)
                if class(i) ~= 3
                    false = false + 1;
                end
            else 
                if class(i) ~= 4
                    false = false + 1;
                end
            end
        end
    end
end

accuracy(TTT) = 1 - (false / 100);
end