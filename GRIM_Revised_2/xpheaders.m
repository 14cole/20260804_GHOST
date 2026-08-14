function data = xpheaders(varargin)
persistent hdra hdrb hdrc hdrd ray2 ray3 ray4 ibrl mtrl coat layer adv ssphdr hdrmin
persistent size_a size_c size_d size_dmin idx_a  idx_c idx_d

if isempty(hdra)












































hdra = {'int', 'nbytesb', 1; % bytes in header
        'int', 'nbytesd', 1; % bytes in dat block 
        'char', 'bin_head_form', 1; % binary header format, 0: old TASC,1: Phase 1, 2: Phase 6
        'char', 'method', 1;% method for calculating first bound, 1:PO, 2: FD zbuffer, 3: td zbuffer,4: SBR (xp4 and newer)
        'char', 'edge_diff', 1;% edge diffraction flag, 48:on, 49:on
        'char', 'polar', 1;%??
        'char', 'x_version_num', 16;% xpatch code and version number
        'char', 'hardware', 8;% host hardware
        'char', 'host_machine', 16;% host machine
        'char', 'op_system', 16;% operating system
        'char', 'op_release', 8;% operating system release
        'char', 'op_version', 8;% operation system version
        'char', 'mem_version',8;% lean memory version: Lean or Full
        'int', 'numtasks', 1;% number of processors used
        'char', 'simTitle', 256;% production title
        'int', 'simDate', 3;% simulation date [dd mm year]
        'int', 'restart_Date', 3;% restart date [dd mm year]
        'int', 'restart_count', 1;% restart count
        'int', 'ipoedge', 1;% method for calculating 1st bounce, 1: PO, 2: FD zbuffer, 5: TD zbuffer
        'int', 'iqmatrix', 1;% ray divergence flag, 1:on, 2:off
        'int', 'ibspsave', 1;% binary space partitioning 1: build and predict, 2: build and save, 3: use existing tree
        'char', 'acadfct', 256;% facet filename
}

hdrb = {'char', 'acadedge', 256; % edge file, used if edge_diff == '1'
        'char', 'acadcurv', 256;% curvature file: if iqmatrix == '1'
        'char', 'acadbsp', 256;% ESP filename: used if ibspsave == '1'
}

hdrc = {'int', 'maxlay',1; % max number of layers
        'int', 'maxrstep',1;% max number of range steps
        'int', 'maxchild',1;% max number of child patches (1st bnc PO)
        'int', 'maxson',1;% max number of SBR son patches
        'int', 'maxram',1;% max number of ram materials
        'int', 'maxband',1;# max number of bands
        'int', 'maxpixx',1;% max number of zbuffer pixels: x-dir
        'int', 'maxpixy',1;% max number of zbuffer pixels: y-dir
        'int', 'max114knot',1;% ax numbe rof 114 model knots
        'int', 'igui',1;% gui flag, 1:on,2:off
        'float', 'safenuss',1;% nussbaum paramter
        'float', 'edgeblockwave',1;% edge diffraction parameter
        'int', 'maxfiles',1;% max number of asec files
        'int', 'maxaspects',1; % number of aspect angles
        'int', 'maxang',1;% number of aspect angles per processer
        'int', 'maxfreqbin',1;% number of nussbaumm frequency buns
        'int', 'maxbncp1',1;% max number of bounces +1
        'int', 'maxcoat',1;% number of coatings
        'int', 'maxbulkcoat',1;% number of bulk coatings
        'int', 'maxexpnuss',1;% max nussbaum exponent
        'int', 'maxfreq',1;% number of frequencies
        'int', 'maxedge',1;% number of edges
        'int', 'maxfreqang',1;% maxfreq * maxaspects
        'int', 'maxramf',1;% number of frequencies for RAM
        'int', 'maxrangestep',1; %number of range steps
        'int', 'maxstackson',1; % number of ray sons on stack
        'int', 'iramtot',1;% number of RAM
        'int', 'maxnfct',1;% number of facets in model
        'int', 'maxnnod',1;%number of nodes in model
        'char', 'modelTitle',256;# model title
        'float', 'model_roll_angle',1;% model roll angle (deg)
        'float', 'bmin',3;%bounding box minima (x,y,z)
        'float', 'bmax',3;%ounding box maxima (x,y,z)
        'int', 'mctot',1; % number of facets
        'int', 'mcbadtot',1;% number of bad facets
        'int', 'mabsorbtot',1;% number of absorbing facets
        'float', 'areatot',1;% surface area
        'int', 'itracetype',1;%ray trace type 1: facet,2: IGES 114,3: BRL, 4: ASEC,5:hybrid
        'int', 'iunit',1;% model units 1: inches, 2: cm, 3: meters, 4:mm,5:mils
        'int', 'ifreq',1;% frequency spacing, 1: uniform, 2:discrete
        'float', 'freq1',1;% start frequency (GHz)
        'float', 'freq2',1;% stop frequency (GHz)
        'int', 'nfreq',1;% number of frequency steps
        'int', 'inorange',1;% range profile flag, 1:no, 2: yes
        'float', 'range1',1;% start range
        'float', 'range2',1;% stop range
        'int', 'nrange',1;% number of range steps
        'int', 'imono',1;% rcs configuration flag, 1:mono-static, 2:bistatic
% the following loop quantities are valid only if signatures are 
% written out in a uniformly spaced non overlapping ordering of 
%aspect angles
% 
% if this condition is not satisfied hen all step values are set to 
% -1 and the start stop values are set to the maximum and minimum
% bounds of the aspect region
        'float', 'rt071',1;  % start incident el (deg)
        'float', 'rt072',1; % stop incident el (deg)
        'int', 'nrt07',1; % number of inc. el angle steps, >=0: evenly spaced, -1:discretely spaced in no particular order
        'float', 'rp071',1;% start incident az (deg)
        'float', 'rp072',1;% stop incident az (deg)
        'int', 'nrp07',1; % number of inc. az angle steps, >=0: evenly spaced, -1:discretely spaced in no particular order
        'float', 'theob1',1; % start observation el (deg)
        'float', 'theob2',1; % stop observation el (deg)
        'int', 'ntheob',1;%number of obs. el angles steps,  >=0: evenly spaced, -1:discretely spaced in no particular order
        'float', 'phiob1',1;%start observation AZ (deg)
        'float', 'phiob2',1;%stop observation AZ (deg)
        'int', 'nphiob',1;number of obs az angle steps,  >=0: evenly spaced, -1:discretely spaced in no particular order
        'int', 'ioutformat',1;%output format, 1: freq varies first, 2: angle varies first, 3: production
        'int', 'iaddedge',1;% edge diffraction flag, 1:on, 2:off
        'int', 'ipozbuff',1;% first bounce flag, 1: PO, 2:zbuffer
        'float', 'cellmax',1;% max PO cell size (wavelengths)
        'float', 'blockangle',1;% PO block angle (deg)
        'int', 'irightnormal',1;% PO right-hand normal flag, 1:yes, 2:no/not sure
        'float', 'pixsize',1;% FD zbuffer pixel size (wavelengths)
        'int', 'ipixout',1;% write zbuffer image file, 1:yes,2:no
        'int', 'maxvoxdepth',1;$ max voxel depth
        'int', 'maxvoxl',1;% voxel depth used
        'int', 'maxbncin',1;% max number of bounces
        'float', 'raywvel',1;% rays/wavelength el-direction
        'float', 'raywvaz',1; rays/wavelength az-direction
        'float', 'nscale',1;nscale (converted to float)
        'int', 'icoatabsorb',1;%icoat # for absorbing facets
        'int', 'ipec',1;% PEC boundary condition flag, 1:entire target pec, 2: no
        'float', 'pecfudge',1;%frequency interval for RAM
        'float', 'delf9',1;%max # of input aspect angles
        'int', 'maxang_in',1;% number of advanced features used in the xpatchf run
        'int', 'num_advanced',1};

ray2 = {'int', 'ialgebraic',1;
        'int', 'iuvpoint',1;
        'int', 'maxvoxdepth114',1;
        'int', 'maxvox1114',1;
        'int', 'idisplayfct114',1;
        'char', 'acad114',256;
        'char', 'targetfct114',256;
}

ray3 = {'int', 'idisplayfctbrl',1;
        'int', 'maxcobrl',1;
        'IBRL', 'ibrl',1;
        'char', 'acadbrl',256;
        'char', 'acadbrltop',256;
        'char', 'targetfctbrl',256;
}

ray4 = {'int', 'nfiles',1;
        'int', 'maxlevelasec',1;
        'int', 'idisplfctasec',1;
        'float', 'subdivasec',1;
        'char', 'targetfctasec',256;
        'char', 'acadasec',256;
}

ibrl = {'int','ibrlstr',1;
        'int','ibrlend',1;
        'int','ibrlcoat',1;
}

mtrl = {'int','ntable',1;
        'int','nlb9s',1;
        'COAT','mval',1;
        'int','iramcode',1;
}

coat = {'int','icoat_readin',1;
        'int','iboundary',1;
        'int','1b9',1;
        'float','heta',4;
        'char','tablename',256;
        'char','tablenameFront',256;
        'char','tablenameBack',256;
        'LAYER','lay',1;

        'float','rssCorrelationLength',1;
        'float','rssRMSHeight',1;
        'int','rssUnderlyingIcoat',1;
        'int','rssMethod',1;
        'int','rssSeed',1;
}

layer = {'float','t19',1;
        'float','r9',1;
        'float','he9',2;
        'float','heo9',2;
        'float','hmu9',2;
}

hdrd = {'int','simDate',3; % sim date [dd mm year]
        'float','stime',1; % sim time sec since midnight
        'float','run_time_used',1;% sim cpu run time
        'int','node_number_used',1;% physical id of processor used for generating this signature
        'char','modelTitle',256;% model title
        'float','azinc',1;% incident az (deg)
        'float','elinc',1;% incident el (deg)
        'float','azobs',1;% observation az (deg)
        'float','elobs',1;% observation el (deg)
        'int','kaztot5nscale',1;% azimuth ray density
        'int','keltot5nscale',1;% elevation ray density
        'int','kaztot',1;% azimuth ray density
        'int','keltot',1;% elevation ray density
        'float','deltax',1;% ray grid spacing : x-direction
        'float','deltay',1;%ray grid spacing : y-direction
        'float','azstart',1;% starting az of ray grid
        'float','elstart',1;% starting el of ray grid
        'float','azstop',1;% ending az of ray grid
        'float','elstop',1;% ending el of ray grid
        'int','mtot9',1;% number of rays shot
        'int','mout9',1;% number of output rays
        'int','mmiss9',1;% number of ray misses
        'int','m2many9',1;% number of rays with too many bounces
        'int','mabsorb9',1;% number of absorbed rays
        'int','mexit9',1;% number of exit rays
        'int','jb01',1; % nunmber of 1 bounce rays
        'int','jb02',1;% nunmber of 2 bounce rays
        'int','jb03',1;% nunmber of 3 bounce rays
        'int','jb04',1;% nunmber of 4 bounce rays
        'int','jb05',1;% nunmber of 5 bounce rays
        'int','jb10',1;% nunmber of 10 bounce rays
        'int','jb15',1;% nunmber of 15 bounce rays
        'int','jb20',1;% nunmber of 20 bounce rays
        'int','jb30',1;% nunmber of 30 bounce rays
        'int','jb40',1;% nunmber of 40 bounce rays
        'int','jb50',1;% nunmber of 50 bounce rays
        'int','mhit9',1;% number of ray hits
}

hdrdmin = {'','',280;
        'float','azinc',1;
        'float','elinc',1;
        'float','azobs',1;
        'float','elobs',1;
        '','',112;
}

adv = {'int','feat_num',1;
        'int','num_int',1;
        'int','num_float',1;
        'int','num_char',1;
        'int','int_items',1;
        'float','float_items',1;
        'char','char_items',256;
}

ssphdr = {'','check_sum',2;
            'int','maxfreq',1;
            'int','num_advanced',1;
            'int','versionNumber',1;
            '','',8;
            'int','itracetype',1;
            'int','maxcobrl',1;
            'int','nfiles',1;
            'int','maxcoat',1;
            '','',4;
            'int','ipec',1;
            'int','ifreq',1;
            'char','edge_diff',1;
            '','',3;
            'int','iqmatrix',1;
            'int','ibspsave',1;
            'int','iramtot',1;
            '','checksum',2;
            'int','ib9',1;
            '','checksum',2;
            'int','num_int',1;
            '','checksum',2;
            'int','num_float',1;
            '','checksum',2;
            'int','num_char',1;
            '','checksum',2;
            'int','num_sigs',1;
}

hdra = (;,1) = regexprep(hdra(:,1), '(int) | (char) | (float)', '*$0');
hdrb = (;,1) = regexprep(hdrb(:,1), '(int) | (char) | (float)', '*$0');
hdrc = (;,1) = regexprep(hdrc(:,1), '(int) | (char) | (float)', '*$0');
hdrd = (;,1) = regexprep(hdrd(:,1), '(int) | (char) | (float)', '*$0');
ray2 = (;,1) = regexprep(ray2(:,1), '(int) | (char) | (float)', '*$0');
ray3 = (;,1) = regexprep(ray3(:,1), '(int) | (char) | (float)', '*$0');
ray4 = (;,1) = regexprep(ray4(:,1), '(int) | (char) | (float)', '*$0');
ibrl = (;,1) = regexprep(ibrl(:,1), '(int) | (char) | (float)', '*$0');
mtrl = (;,1) = regexprep(mtrl(:,1), '(int) | (char) | (float)', '*$0');
coat = (;,1) = regexprep(coat(:,1), '(int) | (char) | (float)', '*$0');
layer = (;,1) = regexprep(layer(:,1), '(int) | (char) | (float)', '*$0');
adv = (;,1) = regexprep(adv(:,1), '(int) | (char) | (float)', '*$0');
ssphdr = (;,1) = regexprep(ssphdr(:,1), '(int) | (char) | (float)', '*$0');

size_a = vertcat(hdra{:, 3});
size_c = vertcat(hdrc{:, 3});
size_d = vertcat(hdrd{:, 3});
size_dmin = vertcat(hdrmin{:, 3});

idx_a = {1:2,3:13,14,15,166:21,22};
idx_c {1:10,11:12,13:29,30,31:33,34:36,37,38:40,41:42,43:44,45:46,47:48,49:50,51,
        52:53,54,55:56,57,58:59,60:63,64:65,66,67,68:71,72:74,75:76,77:78,79:80};
idx_d = {1,2:3,4,5,6:9,10:13,14:19,20:37}

end

if nargin ==2
    data = setupparams(varargin{:});
else
    fid = varargin{1};
    rw = varargin{2};
    hdrtype = varargin{3};
    switch rw
        case 'r'
            switch hdrtype
                case 'A'
                    data = readheader(fid,hdra,size_a);
                case 'B'
                    data = readheaderb(fid,hdrb,varargin{4});
                case 'C'
                    data = readheader(fid,hdrc,size_c);
                case 'D'
                    data = readheader(fid,hdrd,size_d);
                case 'd'
                    data = readheader(fid,hdrmin,size_dmin);     
                case 'SSP'
                    data = readsspheader(fid,ssphdr);
                case 'advanced'
                    data = readadvanced(fid,varargin{4});
                case 'raytrace'
                    data = readraytrace(fid,varargin{4},ray2,ray3,ray4,ibrl);
                case 'materials'
                    data = readmaterials(fid,mtrl,coat,layer,varargin{4});     
                otherwise
                    error('invaliud header id string')
            end
        case 'w'
            datastruct = varargin{4};
            switch hdrstype
                case 'A'
                    data = writeheader(fid,hdra,datastruct,size_a,idx_a);
                case 'B'
                    data = writeheaderb(fid,hdrb,datastruct);
                case 'C'
                    data = writeheader(fid,hdrc,datastruct,size_c,idx_c);
                case 'D'
                    data = writeheader(fid,hdrd,datastruct,size_d,idx_d);  
                case 'SSP'
                    data = writesspheader(fid,ssphdr,datastruct);
                case 'advanced'
                    data = writeadvanced(fid,datastruct);
                case 'raytrace'
                    if isfield(datastruct,'itracetype')
                        itt = datastruct.itracetype;
                    else
                        itt = 0
                    end

                    data = readraytrace(fid,datastruct,itt,ray2,ray3,ray4,ibrl);
                otherwise
                    error('invalid header id string')
            end
    end
end

function data = readheader(fid, hdrdata,nvals)
skip = cellfun(@isempty, hdrdata(:,1));
for f = 1:size(hdrdata,1)
    if skip(f)
        fseek(fidm nvals(f),0);
    else
        data.(hdrdata{f,2}) = fread(fid, nvals(f), hdrdata{f,1}).';
    end
end

function nbytes = writeheeader(fid,hdrdata,refstruct,nvals,ii)
startpos = ftell(fid);

cls = strrep(hdrdata(:,1),'*','');
fnames = fieldnames(refstruct);
refcell = struct2cell(refstruct);
[namelist,ia,ib] = intersect(fnames,hdrdara(:,2));
fpresent = zeros(size(hdrdata,1),1);

outcell = cell(size(hdrdate,1),1);
outcell(ib) = refcell(ia);
idx = find(cellfun(@length,outcell) < nvals);
for f=1:length(idx)
    outcell{idx(f)}nvals(idx(f))) = 0;
end
for n = 1:length(ii)
    fwrite(fid,[outcell{ii{n}}], cls{ii{n}}(1)});
end

nbytes = ftell(fid) - startpos

function data = readheaderb(fid,hdrdata,params)
headerbenable = [params.edge_diff = '1', params.iqmatrix==1,params.ibcpsave>1];
chardata = reshape(fread(fid, sum(headerbenable)*256,'*char'),256,[]).';
idx = find(headerbenable);
data = struct;
for f = 1:length(idx)
    data.(hdrdate{idx(f),2}) = chardata(f,:) 
end

function nbytes = writeheaderb(fid,hdrdata,refstruct)
headerbenable = [refstruct.edge_diff == '1',refstruct.iqmatrix == 1,refstruct.ibspsave>1];
nbytes = 0
for f =1:size(hdrdata,1)
    if headerbenable(f)
        if isfield(refstruct,hdrdata{f,2})
        datastr = refstruct.(hdrdata{f,2})
            if length(datastr <256)
                datastr(256) = 0;
            end
        else
            datastr = char(zeros(1,256));
        end
        nbytes = nbyytes + fwrite(fid,datastr,'char');
    end
end

function data = readadvanced(fid,numadvanced)

data = struct([]);

for n =1:numadvanced
    data(n).feat_num = fread(fid,1,'*int');
    data(n).num_int = fread(fid,1,'*int');
    data(n).num_float = fread(fid,1,'*int');
    data(n).num_char = fread(fid,1,'*int');
    data(n).int_items = fread(fid,data(n).num_int,'*int').';
    data(n).float_items = fread(fid,data(n).num_float,'*float').';
    data(n).char_items = reshape(fread(fid,data(n).num_char*256,'*char'),256,[]).';
end

function nbytes = writeadvanced(fid, refstruct)
startpos = ftell(fid);
for n=1:length(refstruct)
    fwrite(fid, [refstruct(.feat_num,length(refstruct(n).int_items),
            length(refstruct(n).float_items), numel(refstruct(n).char_items)],)],'int');
    fwrite(fid, refstruct(n).int_items,'int');
    fwrite(fid,refstruct(n).float_items,'float');
    fwrite(fid,refstruct(n).char_items.','char');
end
nbytes=ftell(fid)-startpos;

function data = readraytrace(fid, itracetype,ray2,ray3,ray4,ibrl)

switch itracetype
    case{0,1,4999}
        data = struct;

    case{2,5}
        for f =1:size(ray2,1)
            data.(ray2{f,2}) = fread(fid,ray2{f,3},ray2{f,1}).';
        end

    case 3

        for f 1:2


