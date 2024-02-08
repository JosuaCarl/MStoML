function [mz_y,mz_all,CMZ,PR,peaks] = cluster_window(mz,int)
%Classical Peak Picking
HeightFilter = 1000;  % 500
PromFilter = 1000;    % 500
%Binning/Clustering
cluster_cutoff = 0.007^2;  % 0.0055^2

%peak picking
for k = 1:size(int,1)
    [pks,locs] = findpeaks(int(k,:),'MinPeakHeight',HeightFilter,'MinPeakProminence',PromFilter);
    pmz        = mz(locs);
    peaks(k,1) = {[pmz',pks']};
end

peak_selection = [];
for k = 1:size(int,1)
    peak1 = [peaks{k,1},ones(size(peaks{k,1},1),1)*k] ;
    peak_selection = vertcat(peak_selection,peak1); %mz, int, group ~ file
end

if(size(peak_selection,1)<=1)
    mz_y = [];
    mz_all = [];
    CMZ = [];
    PR = [];
else
    %Binning/Clustering
    distfun = @(x,y) (x(:,1)-y(:,1)).^2  + (x(:,2)==y(:,2))*10^6;
    
    %Binning/ clustering
    xx = pdist([peak_selection(:,1),peak_selection(:,3)],distfun);
    %% Changes
    % original code
    % tree = linkage(xx);
    % Niklas Code
    tree = linkage(xx,'complete');
    clusters = cluster(tree,'CUTOFF',cluster_cutoff,'CRITERION','Distance');
    %% Changes 
    % Original code
    % CMZ = accumarray(clusters,prod(peak_selection(:,1:2),2))./accumarray(clusters,peak_selection(:,2));
    % From Niklas
    CMZ = accumarray(clusters,(peak_selection(:,1)).*(peak_selection(:,2)).^5)./accumarray(clusters,peak_selection(:,2).^5);
    
    PR =  accumarray(clusters,peak_selection(:,2),[],@max);
    
    [CMZ,h] = sort(CMZ);
    PR = PR(h);
    
    %% from Niklas
    cx = unique(clusters);
    mz_y = nan(length(peaks),length(cx));
    mz_all = nan(length(peaks),length(cx));
    
    for i = 1:length(cx)
        peakid_new = [];
        pheight_new = [];
        pmz_new = [];
        id = clusters == cx(h(i));
        idn = find(id);
        int_cx = peak_selection(id,2);
        peakid = peak_selection(id,3);
        peakpos = peak_selection(id,1);
        peakid_unique = unique(peakid);
        for kxy = 1:length(peakid_unique)
            pid_n = peakid_unique(kxy);
            posn = pid_n == peakid;
            
            peak_heights = int_cx(posn);
            peak_mz      = peakpos(posn);
            [max_pheight,idk]  = max(peak_heights);
            peakid_new(1,kxy) = pid_n;
            pheight_new(1,kxy) = max_pheight;
            pmz_new(1,kxy)   = peak_mz(idk);
        end
        
        mz_y(peakid_new,i) = pheight_new;
        mz_all(peakid_new,i) = pmz_new;
        
    end
end
