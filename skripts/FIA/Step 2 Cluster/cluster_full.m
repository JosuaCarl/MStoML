function cluster_full

global fia_data

cnt1 = [1:2000:2*10^6];
cnt2 = [2000:2000:2*10^6];

for k = 1:2
    FA ={};
    FMZ = [];
    FY = [];
    Fcmz= [];    
    parfor j = 1:1000 
        mz1  = fia_data(k).mz(1,[cnt1(j):cnt2(j)]);
        int1 =fia_data(k).int(:,[cnt1(j):cnt2(j)]);
        
        %% CLUSTERING
        [mz_y,mz_all,cmz,pr,peaks] = cluster_window(mz1,int1);
        
        FY = [FY; mz_y'];
        FMZ = [FMZ;mz_all'];
        Fcmz = [Fcmz;cmz];
    end
    fia_data(k).FY = FY;
    fia_data(k).FMZ = FMZ;
    fia_data(k).CMZ = Fcmz; 
end

