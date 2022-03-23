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
Cellpix_out = initial;
Cytopix_out = initial;
Cellnum_out = initial;
RRnuc_out = initial;
NInuc_out = initial;
NIctyo_out = initial;
FIcyto_out = initial;
Ntmnuc_out = initial;
Imnum_out = initial;
D_out = initial;



filefront = 'C:\Users\ggallegolopez\Documents\Data_From_Undergrad_Computer\03182019\tiffs\Cells-';
ffilefront = 'C:\Users\ggallegolopez\Documents\Data_From_Undergrad_Computer\03182019\tiffs\Cells-';
filefrontm = 'C:\Users\ggallegolopez\Documents\Data_From_Undergrad_Computer\03182019\masks\Cells-';

%change these to match your data

im_num = [1 5 11 13 17 19 22 24 26 28 33 36 39 42 44 46 48 50 52 55 58 61 64 66 68 72 75 78 81 84 86 88 90 92 96 99 102 106 108 110 112 114 116 118 122 124 130 135 137 139 141 143 147 149 153 155 158 160 162 164 166 168 172 174 178 180 183 185 187 189 191 193 196 199 203 205 208 210 212 214 216 218 222 224 227 230 233 235 237 239 242 245 248 251 254 257 259 261 263 265 267 270 273 276 280 282 284 286 288 291 294 297 300 303 306 308 310 312 314 316 319 322 325 328 331 334 336 338];
fim_num = [2 6 12 14 18 20 23 25 27 29 31 34 37 40 43 45 47 49 51 53 56 59 62 65 67 71 73 76 79 82 85 87 89 91 93 97 100 103 109 111 113 115 117 119 123 125 129 131 134 136 138 140 142 144 148 150 154 156 159 161 163 165 167 169 173 175 179 181 184 186 188 190 192 194 197 200 204 206 209 211 213 215 217 219 223 225 228 231 234 236 238 240 243 246 249 252 255 258 260 262 264 266 268 271 274 277 281 283 285 287 289 292 295 298 301 304 307 309 311 313 315 317 320 323 326 329 332 335 337 339 341];

for a = 1:length(im_num)
    mask_image = imread(strcat(filefrontm,num2str(im_num(a), '%03.f'),'_photons _cells.tiff'));
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
    D_cell = maskmax;
    RRnuc_cell = maskmax;
    NInuc_cell = maskmax;
    NIcyto_cell = maskmax;
    FIcyto_cell = maskmax;
    Ntmnuc_cell = maskmax;
    
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
%         RRnuc_pix = xleninitial;
%         NInuc_pix = xleninitial;
%         Ntmnuc_pix = xleninitial;
        
        for j = 1:length(x);
%            if F_tm_image(x(j),y(j)) > 1000;
%            else
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
%            end
        end
%         [l m] = find(nuc_cell_image == i);
%         if length(l)>0;
%             RRnuc_pix = zeros(1,length(l));
%             NInuc_pix = zeros(1,length(l));
%             Ntmnuc_pix = zeros(1,length(l));
%             for k = 1:length(l);
%                 RRnuc_pix(k) = rr_image(l(k),m(k));
%                 NInuc_pix(k) = N_photons(l(k),m(k));
%                 Ntmnuc_pix(k) = N_tm_image(l(k),m(k));
%             end
%         
%         end
%         
%         [n o] = find(mask_image == i);
%         if length(n)>0;
%             NIcyto_pix = zeros(1,length(n));
%             FIcyto_pix = zeros(1,length(n));
%             
%             for q = 1:length(n);
%                 NIcyto_pix(q) = N_photons(n(q),o(q));
%                 FIcyto_pix(q) = F_photons(n(q),o(q));
%             end
%         end
%                 
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
        
%         [cell_x cell_y] = find(cell_image==i);
%         Cellpix_cell(i) = length(cell_x);
%         Cytopix_cell(i) = length(x);
%         Cellnum_cell(i) = i;
        Imnum_cell(i)=a;
        D_cell(i) = pi*(mean([(max(x)-min(x)), (max(y)-min(y))]))^2;
       F_chi_cell(i) = mean(F_chi_pix(F_chi_pix<10));
        N_chi_cell(i) = mean(N_chi_pix(N_chi_pix<10));
%         RRnuc_cell(i) = mean(RRnuc_pix(RRnuc_pix>0));
%         NInuc_cell(i) = mean(NInuc_pix(NInuc_pix>0));
%         NIcyto_cell(i) = mean(NIcyto_pix(NIcyto_pix>0));
%         FIcyto_cell(i) = mean(FIcyto_pix(FIcyto_pix>0));
%         Ntmnuc_cell(i) = mean(Ntmnuc_pix(Ntmnuc_pix>0));
        end
        end
    end
    if a == 1;
        ind1 = length(rr_out(rr_out>0))+1;
    ind2 = length(rr_out(rr_out>0))+length(rr_cell);
    else
        ind1 = max(find(Imnum_out>0))+1;
        ind2 = ind1+length(rr_cell)-1;
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
   Cellpix_out(ind1:ind2) = Cellpix_cell;
   Cytopix_out(ind1:ind2) = Cytopix_cell;
   Cellnum_out(ind1:ind2) = Cellnum_cell;
   Imnum_out(ind1:ind2)=Imnum_cell;
       D_out(ind1:ind2) = D_cell;
%        RRnuc_out(ind1:ind2) = RRnuc_cell;
%         NInuc_out(ind1:ind2) = NInuc_cell;
%         NIcyto_out(ind1:ind2)=NIcyto_cell;
%         FIcyto_out(ind1:ind2) = FIcyto_cell;
%         Ntmnuc_out(ind1:ind2) = Ntmnuc_cell;
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
Cellpix_out = Cellpix_out(Cellpix_out>0);
Cytopix_out = Cytopix_out(Cytopix_out>0);
Cellnum_out = Cellnum_out(Cellnum_out>0);
Imnum_out=Imnum_out(Imnum_out>0);
D_out = D_out(D_out>0);
% RRnuc_out = RRnuc_out(RRnuc_out>0);
% NInuc_out= NInuc_out(NInuc_out>0);
% NIcyto_out = NIcyto_out(NIcyto_out>0)';
% FIcyto_out = FIcyto_out(FIcyto_out>0);
% Ntmnuc_out= Ntmnuc_out(Ntmnuc_out>0);


save('C:\Users\ggallegolopez\Documents\Data_From_Undergrad_Computer\Imaging_24hrs.mat','rr_out','N_tm_out','N_t1_out','N_t2_out','N_a1_out',...
    'F_tm_out','F_t1_out','F_t2_out','F_a1_out','N_chi_out','F_chi_out','Imnum_out','D_out');
clear all
load('C:\Users\ggallegolopez\Documents\Data_From_Undergrad_Computer\Imaging_24hrs.mat')
