function cluster_full

global fia_data

cnt1 = [1:2000:2*10^6];         % Array 1, 2001, ... 2*10^6 - 1999
cnt2 = [2000:2000:2*10^6];      % Array 2000, 4000, .... 2*10^6

for k = 1:2
    FA ={};
    FMZ = [];
    FY = [];
    Fcmz= [];    
    parfor j = 1:1000           % iterate through Number of bins (1-> 2Mio in 2000 steps --> 1000 bins)
        mz1  = fia_data(k).mz(1,[cnt1(j):cnt2(j)]);     % Selecting mz values for clustering window
        int1 = fia_data(k).int(:,[cnt1(j):cnt2(j)]);    % Selecting intensity values for clustering window 
        
        %% CLUSTERING
        [mz_y,mz_all,cmz,pr,peaks] = cluster_window(mz1,int1);      % Clustering on 2000 units window in data
        
        FY = [FY; mz_y'];           % Appending mz_y data
        FMZ = [FMZ;mz_all'];        % Appending mz_all data
        Fcmz = [Fcmz;cmz];          % Appending clustered m/z data
    end
    fia_data(k).FY = FY;            % Assignment to variables
    fia_data(k).FMZ = FMZ;
    fia_data(k).CMZ = Fcmz; 
end

