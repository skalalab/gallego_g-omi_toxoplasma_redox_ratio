clear all;
initial = zeros(1000,1);
rr_out = initial;
N_tm_out = initial;
N_t1_out = initial;
N_t2_out = initial;
N_a1_out = initial;
N_chi_out = initial;
F_tm_out = initial;
F_t1_out = initial;
F_t2_out = initial;
F_a1_out = initial;
F_chi_out = initial;
Cellnum_out = initial;
Imnum_out = initial;
has_toxo_out = initial;

trr_out = initial;
tN_tm_out = initial;
tN_t1_out = initial;
tN_t2_out = initial;
tN_a1_out = initial;
tN_chi_out = initial;
tF_tm_out = initial;
tF_t1_out = initial;
tF_t2_out = initial;
tF_a1_out = initial;
tF_chi_out = initial;
tCellnum_out = initial;
tImnum_out = initial;
in_cell_out = initial;


filefront = 'C:\Users\csiamof\Documents\03182019\Cells-';
ffilefront = 'C:\Users\csiamof\Documents\03182019\Cells-';
filefrontm = 'C:\Users\csiamof\Documents\03182019\masks\Cells-';

%change these to match your data
im_num = [1 5 11 13];
fim_num = [2 6 8 12];
tim_num = [3 4 9 10];

%%
for a = 1:length(im_num)
    mask_image = (imread(strcat(filefrontm,num2str(im_num(a), '%03.f'),'_photons _cells.tiff')));
    iscell_image = (imread(strcat(filefrontm,num2str(im_num(a), '%03.f'),'_photons _cells.tiff')));
    toxo_image = (imread(strcat(filefrontm,num2str(tim_num(a), '%03.f'),'_Cycle00001_Ch1_000001.ome_toxo.tiff')));
    N_photons = imread(strcat(filefront,num2str(im_num(a), '%03.f'),'_photons.tiff'));
    F_photons = imread(strcat(ffilefront,num2str(fim_num(a),'%03.f'),'_photons.tiff'));
    N_t1 = imread(strcat(filefront,num2str(im_num(a), '%03.f'),'_t1.tiff'));
    N_t2 = imread(strcat(filefront,num2str(im_num(a), '%03.f'),'_t2.tiff'));
    N_a1 = imread(strcat(filefront,num2str(im_num(a), '%03.f'),'_a1[%].tiff'));
    N_a2 = imread(strcat(filefront,num2str(im_num(a), '%03.f'),'_a2[%].tiff'));
    N_chi = imread(strcat(filefront,num2str(im_num(a), '%03.f'),'_chi.tiff'));
    F_t1 = imread(strcat(ffilefront,num2str(fim_num(a),'%03.f'),'_t1.tiff'));
    F_t2 = imread(strcat(ffilefront,num2str(fim_num(a),'%03.f'),'_t2.tiff'));
    F_a1 = imread(strcat(ffilefront,num2str(fim_num(a),'%03.f'),'_a1[%].tiff'));
    F_a2 = imread(strcat(ffilefront,num2str(fim_num(a),'%03.f'),'_a2[%].tiff'));
    F_chi = imread(strcat(ffilefront,num2str(fim_num(a),'%03.f'),'_chi.tiff'));
    rr_image = N_photons./(N_photons+F_photons);
    N_tm_image = N_t1.*N_a1./100+N_t2.*N_a2./100;
    F_tm_image = F_t1.*F_a1./100+F_t2.*F_a2./100;
    
    
    maskmax = zeros(1,(max(max(mask_image))));
    toxomaskmax = zeros(1,(max(max(toxo_image))));
    
    rr_cell = maskmax;
    N_tm_cell = maskmax;
    N_t1_cell = maskmax;
    N_t2_cell = maskmax;
    N_a1_cell = maskmax;
    F_tm_cell = maskmax;
    F_t1_cell = maskmax;
    F_t2_cell = maskmax;
    F_a1_cell = maskmax;
    F_chi_cell = maskmax;
    N_chi_cell = maskmax;
    Cellpix_cell = maskmax;
    Cytopix_cell = maskmax;
    Cellnum_cell = maskmax;
    Imnum_cell = maskmax;
    has_toxo_cell = maskmax;

    trr_out = toxomaskmax;
tN_tm_cell = toxomaskmax;
tN_t1_cell = toxomaskmax;
tN_t2_cell = toxomaskmax;
tN_a1_cell = toxomaskmax;
tN_chi_cell = toxomaskmax;
tF_tm_cell = toxomaskmax;
tF_t1_cell = toxomaskmax;
tF_t2_cell = toxomaskmax;
tF_a1_cell = toxomaskmax;
tF_chi_cell = toxomaskmax;
tCellnum_cell = toxomaskmax;
tImnum_cell = toxomaskmax;
in_cell_cell = toxomaskmax;
    
    for i = 1:max(max(mask_image));
        
        [x y] = find(mask_image == i);
        if length(x)>0;
        xleninitial = zeros(1,length(x));
        rr_pix = xleninitial;
        N_tm_pix = xleninitial;
        N_t1_pix = xleninitial;
        N_t2_pix = xleninitial;
        N_a1_pix = xleninitial;
        F_tm_pix = xleninitial;
        F_t1_pix = xleninitial;
        F_t2_pix = xleninitial;
        F_a1_pix = xleninitial;
        F_chi_pix = xleninitial;
        N_chi_pix = xleninitial;
%         
        for j = 1:length(x);
%            
            rr_pix(j) = rr_image(x(j),y(j));
            N_tm_pix(j) = N_tm_image(x(j),y(j));
            N_t1_pix(j) = N_t1(x(j),y(j));
            N_t2_pix(j) = N_t2(x(j),y(j));
            N_a1_pix(j) = N_a1(x(j),y(j));
            F_tm_pix(j) = F_tm_image(x(j),y(j));
            F_t1_pix(j) = F_t1(x(j),y(j));
            F_t2_pix(j) = F_t2(x(j),y(j));
            F_a1_pix(j) = F_a1(x(j),y(j));
            N_chi_pix(j) = N_chi(x(j),y(j));
            F_chi_pix(j) = F_chi(x(j),y(j));
          
%           
        end
      
        if mean(N_tm_pix(N_tm_pix>00))<200;
            
        elseif mean(F_tm_pix(F_tm_pix>00))<10;
        elseif isnan(mean(F_tm_pix(F_tm_pix>0)))==1;
        elseif mean(N_tm_pix)==0;
        elseif mean(F_tm_pix)==0;

        else
        rr_cell(i)= mean(rr_pix(rr_pix>0));
        N_tm_cell(i) = mean(N_tm_pix(N_tm_pix>0));
        N_t1_cell(i) = mean(N_t1_pix(N_t1_pix>0));
        N_t2_cell(i) = mean(N_t2_pix(N_t2_pix>0));
        N_a1_cell(i) = mean(N_a1_pix(N_a1_pix>0));
        F_tm_cell(i) = mean(F_tm_pix(F_tm_pix>0));
        F_t1_cell(i) = mean(F_t1_pix(F_t1_pix>0));
        F_t2_cell(i) = mean(F_t2_pix(F_t2_pix>0));
        F_a1_cell(i) = mean(F_a1_pix(F_a1_pix>4));
        Cellnum_cell(i) = i;
        Imnum_cell(i)=im_num(a);
       F_chi_cell(i) = mean(F_chi_pix(F_chi_pix<10));
        N_chi_cell(i) = mean(N_chi_pix(N_chi_pix<10));
        
    
        index = i;
        toxo_check = toxo_image > 0;
        cell2check = mask_image == index;
        check_area = toxo_check.*(cell2check);
        check_mask = check_area >0;
        limit = 5;
        area = sum(check_mask(:));
        if area >= limit
            has_toxo_cell(i) = 1;
        else
            has_toxo_cell(i) = 2;
        end
     %two is non infected, 1 is infected
%         

        end
        end
    end
    %% TOXO
      for i = 1:max(max(toxo_image));
        
        [x y] = find(toxo_image == i);
        if length(x)>0;
        xleninitial = zeros(1,length(x));
        trr_pix = xleninitial;
        tN_tm_pix = xleninitial;
        tN_t1_pix = xleninitial;
        tN_t2_pix = xleninitial;
        tN_a1_pix = xleninitial;
        tF_tm_pix = xleninitial;
        tF_t1_pix = xleninitial;
        tF_t2_pix = xleninitial;
        tF_a1_pix = xleninitial;
        tF_chi_pix = xleninitial;
        tN_chi_pix = xleninitial;
%         
        for j = 1:length(x);
%            
            trr_pix(j) = rr_image(x(j),y(j));
            tN_tm_pix(j) = N_tm_image(x(j),y(j));
            tN_t1_pix(j) = N_t1(x(j),y(j));
            tN_t2_pix(j) = N_t2(x(j),y(j));
            tN_a1_pix(j) = N_a1(x(j),y(j));
            tF_tm_pix(j) = F_tm_image(x(j),y(j));
            tF_t1_pix(j) = F_t1(x(j),y(j));
            tF_t2_pix(j) = F_t2(x(j),y(j));
            tF_a1_pix(j) = F_a1(x(j),y(j));
            tN_chi_pix(j) = N_chi(x(j),y(j));
            tF_chi_pix(j) = F_chi(x(j),y(j));
%           
        end
      
        if mean(tN_tm_pix(tN_tm_pix>00))<200;
            
        elseif mean(tF_tm_pix(tF_tm_pix>00))<10;
        elseif isnan(mean(tF_tm_pix(tF_tm_pix>0)))==1;
        elseif mean(tN_tm_pix)==0;
        elseif mean(tF_tm_pix)==0;

        else
        trr_cell(i)= mean(trr_pix(trr_pix>0));
        tN_tm_cell(i) = mean(tN_tm_pix(tN_tm_pix>0));
        tN_t1_cell(i) = mean(tN_t1_pix(tN_t1_pix>0));
        tN_t2_cell(i) = mean(tN_t2_pix(tN_t2_pix>0));
        tN_a1_cell(i) = mean(tN_a1_pix(tN_a1_pix>0));
        tF_tm_cell(i) = mean(tF_tm_pix(tF_tm_pix>0));
        tF_t1_cell(i) = mean(tF_t1_pix(tF_t1_pix>0));
        tF_t2_cell(i) = mean(tF_t2_pix(tF_t2_pix>0));
        tF_a1_cell(i) = mean(tF_a1_pix(tF_a1_pix>4));
        
%         [cell_x cell_y] = find(cell_image==i);
%         Cellpix_cell(i) = length(cell_x);
%         Cytopix_cell(i) = length(x);
        tCellnum_cell(i) = i;
        tImnum_cell(i)=a;
       tF_chi_cell(i) = mean(tF_chi_pix(tF_chi_pix<10));
        tN_chi_cell(i) = mean(tN_chi_pix(tN_chi_pix<10));
        
        index = i;
        toxo2check = toxo_image == index;
        is_cell = iscell_image > 0;
        check_cellarea = is_cell.*(toxo2check);
        check_iscell = check_cellarea >0;
        limit = 5;
        carea = sum(check_mask(:));
        if carea >= limit
            in_cell_cell(i) = 1;
        else
            in_cell_cell(i) = 2;
        end
%   1 is in cell, 2 is out of cell      

        end
        end
    end
    %%
    if a == 1;
        ind1 = length(rr_out(rr_out>0))+1;
    ind2 = length(rr_out(rr_out>0))+length(rr_cell);
    else
        ind1 = max(find(Imnum_out>0))+1;
        ind2 = ind1+length(rr_cell)-1;
    end
    if a == 1;
        tind1 = length(trr_out(trr_out>0))+1;
    tind2 = length(trr_out(trr_out>0))+length(trr_cell);
    else
        tind1 = max(find(tImnum_out>0))+1;
        tind2 = tind1+length(trr_cell)-1;
    end
    rr_out(ind1:ind2) = rr_cell;
    N_tm_out(ind1:ind2) = N_tm_cell;
    N_t1_out(ind1:ind2) = N_t1_cell;
    N_t2_out(ind1:ind2) = N_t2_cell;
    N_a1_out(ind1:ind2) = N_a1_cell;
    F_tm_out(ind1:ind2) = F_tm_cell;
    F_t1_out(ind1:ind2) = F_t1_cell;
    F_t2_out(ind1:ind2) = F_t2_cell;
    F_a1_out(ind1:ind2) = F_a1_cell;
    F_chi_out(ind1:ind2) = F_chi_cell;
    N_chi_out(ind1:ind2) = N_chi_cell;
   Cellnum_out(ind1:ind2) = Cellnum_cell;
   Imnum_out(ind1:ind2)=Imnum_cell;
   has_toxo_out(ind1:ind2) = has_toxo_cell;
   
    trr_out(tind1:tind2) = trr_cell;
    tN_tm_out(tind1:tind2) = tN_tm_cell;
    tN_t1_out(tind1:tind2) = tN_t1_cell;
    tN_t2_out(tind1:tind2) = tN_t2_cell;
    tN_a1_out(tind1:tind2) = tN_a1_cell;
    tF_tm_out(tind1:tind2) = tF_tm_cell;
    tF_t1_out(tind1:tind2) = tF_t1_cell;
    tF_t2_out(tind1:tind2) = tF_t2_cell;
    tF_a1_out(tind1:tind2) = tF_a1_cell;
    tF_chi_out(tind1:tind2) = tF_chi_cell;
    tN_chi_out(tind1:tind2) = tN_chi_cell;
   tCellnum_out(tind1:tind2) = tCellnum_cell;
   tImnum_out(tind1:tind2)= tImnum_cell;
   in_cell_out(tind1:tind2) = in_cell_cell;

end



rr_out = rr_out(rr_out>0);
N_tm_out = N_tm_out(N_tm_out>0);
N_t1_out = N_t1_out(N_t1_out>0);
N_t2_out = N_t2_out(N_t2_out>0);
N_a1_out = N_a1_out(N_a1_out>0);
F_tm_out = F_tm_out(F_tm_out>0);
F_t1_out = F_t1_out(F_t1_out>0);
F_t2_out = F_t2_out(F_t2_out>0);
F_a1_out = F_a1_out(F_a1_out>0);
N_chi_out = N_chi_out(N_chi_out>0);
F_chi_out = F_chi_out(F_chi_out>0);
Cellnum_out = Cellnum_out(Cellnum_out>0);
Imnum_out=Imnum_out(Imnum_out>0);
has_toxo_out = has_toxo_out(has_toxo_out>0);

trr_out = trr_out(trr_out>0)';
tN_tm_out = tN_tm_out(tN_tm_out>0);
tN_t1_out = tN_t1_out(tN_t1_out>0);
tN_t2_out = tN_t2_out(tN_t2_out>0);
tN_a1_out = tN_a1_out(tN_a1_out>0);
tF_tm_out = tF_tm_out(tF_tm_out>0);
tF_t1_out = tF_t1_out(tF_t1_out>0);
tF_t2_out = tF_t2_out(tF_t2_out>0);
tF_a1_out = tF_a1_out(tF_a1_out>0);
tN_chi_out = tN_chi_out(tN_chi_out>0);
tF_chi_out = tF_chi_out(tF_chi_out>0);
tCellnum_out = tCellnum_out(tCellnum_out>0);
tImnum_out= tImnum_out(tImnum_out>0);
in_cell_out = in_cell_out(in_cell_out>0);


save('C:\Users\csiamof\Documents\03182019\Imaging_24hrs.mat','rr_out','N_tm_out','N_t1_out','N_t2_out','N_a1_out',...
    'F_tm_out','F_t1_out','F_t2_out','F_a1_out','N_chi_out','F_chi_out','Imnum_out','D_out');
clear all
load('C:\Users\csiamof\Documents\03182019\Imaging_24hrs.mat')
