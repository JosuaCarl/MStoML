function [mz_y,mz_all,CMZ,PR,peaks] = cluster_window(mz,int)
%Classical Peak Picking
HeightFilter = 1000;  % 500
PromFilter = 1000;    % 500
%Binning/Clustering
cluster_cutoff = 0.007^2;  % 0.0055^2

%peak picking
for k = 1:size(int,1)
    [pks,locs] = findpeaks(int(k,:),'MinPeakHeight',HeightFilter,'MinPeakProminence',PromFilter); % Picks peaks according to Minimal Height and Prominence
    pmz        = mz(locs);              % extracts m/z values
    peaks(k,1) = {[pmz',pks']};         % Assignment to the out_variable of
end

peak_selection = [];
for k = 1:size(int,1)
    peak1 = [peaks{k,1}, ones(size(peaks{k,1}, 1), 1)*k] ;     % value in peaks[k, 1], and 1d list filled with k (group assignment)
    peak_selection = vertcat(peak_selection, peak1);           % mz values, intensities, group of peaks as lists in list
end

if(size(peak_selection, 1) <=1)       % Define arrays, if peak selection doesn't contain anything
    mz_y = [];
    mz_all = [];
    CMZ = [];
    PR = [];
else
    %Binning/Clustering     - Define distance function between two points
    distfun = @(x,y) (x(:,1)-y(:,1)).^2  + (x(:,2)==y(:,2))*10^6;       % Squares value of 1st col (mz value) 
                                                                        % adds 10^6, if the entry in the second column is in the same group (weight for "peak picking" step)
    
    %Binning/ clustering
    distance_matrix = pdist( [peak_selection(:,1), peak_selection(:,3)] , distfun);  % Pairwise distance between points in matrix, according to distance function
    %% Changes
    % original code
    % tree = linkage(distance_matrix);
    % Niklas Code
    tree = linkage(distance_matrix,'complete');                                     % Make farthest distance ("complete") linked tree from distance matrix
    clusters = cluster(tree,'CUTOFF',cluster_cutoff,'CRITERION','Distance');        % Define clusters according to cutoff value 
                                                                                    % (List of cluster number to which position is assigned)
    %% Changes 
    % Original code
    % CMZ = accumarray(clusters,prod(peak_selection(:,1:2),2))./accumarray(clusters,peak_selection(:,2));
    % From Niklas
    CMZ = accumarray(clusters, (peak_selection(:,1)) .* (peak_selection(:,2)).^5) ./ accumarray(clusters, peak_selection(:,2).^5);  % sum of mz * int^5 / sum int^5
                                                                                                                                    % --> weighting of intensity
    PR =  accumarray(clusters, peak_selection(:,2),[],@max);                                                                        % retain sum of maximal intensities
    
    [CMZ,h] = sort(CMZ);            % ascending sorting of m/z values  h: Permutation indices
    PR = PR(h);                     % Change order of maximal intensities, according to ascending m/z
    
    %% from Niklas
    cx = unique(clusters);
    mz_y = nan(length(peaks),length(cx));
    mz_all = nan(length(peaks),length(cx));
    
    for i = 1:length(cx)            % Iterate over unique clusters
        peakid_new = [];
        pheight_new = [];
        pmz_new = [];

        id = clusters == cx(h(i));
        idn = find(id);             % indices of id, that are not zero -> extracts unique clusters from clusters

        % assign same clusters to varibales
        peakpos = peak_selection(id,1);
        int_cx = peak_selection(id,2);
        peakid = peak_selection(id,3);

        % Select unique peak IDs 
        peakid_unique = unique(peakid);
        for kxy = 1:length(peakid_unique)
            pid_n = peakid_unique(kxy);         % All peak ids that are equal
            posn = pid_n == peakid;             % Positions of all peak ids that are equal to the current peak ID
            
            peak_heights = int_cx(posn);
            peak_mz      = peakpos(posn);
            [max_pheight,idk]  = max(peak_heights);     % Retain the maximum values in the cluster when comparing each peak
            peakid_new(1,kxy) = pid_n;
            pheight_new(1,kxy) = max_pheight;
            pmz_new(1,kxy)   = peak_mz(idk);
        end
        
        mz_y(peakid_new,i) = pheight_new;
        mz_all(peakid_new,i) = pmz_new;
        
    end
end
