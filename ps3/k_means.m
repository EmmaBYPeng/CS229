% Load images
A = double(imread('mandrill-small.tiff'));
A_large = double(imread('mandrill-large.tiff'));
width = size(A,1); % 128
widthL = size(A_large); % 512

K = 16;
threshold = 1e-6;
centroidDiff = 1;

% Randomly initialize k centroids
xRand = randi([1 width],1,K);
yRand = randi([1 width],1,K);

centroids(1:K,1:3) = 0;
for i = 1:K
    centroids(i,:) = reshape(A(xRand(i),yRand(i),:),1,3);
end

% Run k means
while centroidDiff > threshold
    % E step, assign each point to the closest centroid
    labels(1:width,1:width) = 0;
    for i = 1:width
        for j = 1:width
            dist(1:K) = 0;
            for k = 1:K
                diff = reshape(A(i,j,:),1,3) - centroids(k,:);
                dist(k) = diff * diff';
            end
            [minDist,labels(i,j)] = min(dist);
        end
    end
    
    % M step, move each centroid to the mean of the points assigned to it
    oldCentroids = centroids;
    centroids(1:K,1:3) = 0;
    for k = 1:K
       count = 0;
       for i = 1:width
           for j = 1:width
               if labels(i,j) == k 
                    centroids(k,:) = centroids(k,:) + reshape(A(i,j,:),1,3);
                    count = count + 1;
               end
           end
       end
       centroids(k,:) = centroids(k,:) / count;
    end
    
    centroidDiff = sum(sum(abs(oldCentroids - centroids)));
end

% Create new image
A2(1:widthL,1:widthL,1:3) = 0;
for i = 1:widthL
    for j = 1:widthL
        dist(1:K) = 0;
        for k = 1:K
            diff = reshape(A_large(i,j,:),1,3) - centroids(k,:);
            dist(k) = diff * diff';
        end
        [minDist,label] = min(dist);
        A2(i,j,:) = reshape(centroids(label,:),1,1,3);
    end
end

imshow(uint8(round(A2)));

