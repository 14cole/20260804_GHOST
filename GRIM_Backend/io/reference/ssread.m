function {data, varargout} = ssread(filename)

if nargout = 2
    headerdstr = 'D';
else
    headerstr = 'd';
end

[fid, message] = fopen(filename, 'rb', 'ieee-be');

fseek(fid,0,'eof');
filesize = ftell(fid);
fseek(fid,0,'bof');

num_sigs = 0
while ftell(fid) ~= filesize
    nbytesb = fread(fid,1,'int');
    nbytesd = fread(fid,1,'int');
    if nbytesb && nbytesd
        fseek(fid, nbnytesb+nbytesd -8,0);
        num_sigs = num_sigs +1;
    else
        num_sigs =1;
        nbytesd = 440
        warning('nbytesb is not set. uable to estimate number of signals')
        break
    end
end

fseek(fid,0,'bof')

num_freqs = (nbytesd-408)/32;
blocksize = num_freqs * 8;
[data.vv, data.vh, data,hv, data.hh] = deal(zeros(num_freqs,num_sigs,'single'));

for n=1:num_sigs
    if nargout == 2 || (nargout >2 && n==1)
    headera(n) = xpheaders(fid,'r','A');
    headerb(n) = xpheaders(fid,'r','B',headera(n));
    headerc(n) = xpheaders(fid,'r','C');

    if headerc(n).num_advanced
        advanced{n} = xpheaders(fid,'r', 'advanced', headersc(n).num_advanced);
    else
        advanced{n} = struct([]);
    end

    raytrace(n) = xpheaders(fid,'r','raytrace', headerc(n).itracetype);

    materials = xpheaders(fid,'r','materials',{headera(n), advanced{n}});

    if headerc(n).ifreq ==2
        freqdata = fread(fid,headerc(n).maxfreq,'*float').';
    else
        freqdata = linspace(headerc(n).freq1, headerc(n).freq2,headerc(n).maxfreq);
    end
    else
        fseek(fid,fread(fid,1,'int')-4,0);
    end


    if nargout > 1
        headerd(n) = xpheaders(fid,'r',headerstr);
    else
        fseek(fid,408,0);
    end

    datachunk = fread(fid,blocksize,'*float');
    complexdata = complex(datachunk(1:2:end), datachunk(2:2:end));
    data.vv(:,n) = complexdatra(1:4:end);
    data.vh(:,n) = complexdatra(2:4:end);
    data.hv(:,n) = complexdatra(3:4:end);
    data.hh(:,n) = complexdatra(4:4:end);
end

fclose(fid);

if nargout ==2
    headerd = rmfield(headerd,{'modelTitle','simDate'});
    fnames = vertcat(fieldnames(headera, fieldnames(headerb),
                    fieldnames(headerc), fieldnames(headerd));
    sdata = vertcat(struct2cell(headera, struct2cell(headerb),
                    struct2cell(headerc), struct2cell(headerd));  
    
    varaagout{1} = cell2struct(sdata,fnames,1);
    [varargout{1}.freqdata] = deal(freqdata);
    for s = 1:num_sigs
        varargout{1}(s).raytrace = raytrace(s);
        if headerc(s).num_advanced
            varargout{1}(s).advanced = advances({s};
        end
    end
end

if nargout >= 4
    varargout{1} = freqdata;
    varargout{2} = [headerd.azinc];
    varargout{3} = [headerd.elinc];
    if nargout ==5
        fnames = vertcat(fieldnames(headera), fieldnames(headerb), fieldnames(headerc));
        sdata = vertcat(struct2cell(headera), struct2cell(headerb), struct2cell(headerc));

        varargout{4} = cell2struct(sdata,fnames,1);
        varargout{4}.freqdata = freqdata;
        varargout{4}.raytrace = raytrace(1);
        if headerc(1).num_advanced
            varargout{4}.advanced = advanced(1, :);
        end

        fnames = fieldnames(headerd);
        varargout{4}(num_sigs).fnames{1}) = headerd.(fnames{1});
        for f = 1:length(fnames)
            [varargout{4}.(fnames{f})] = headerd.(fnames{f});
        end
    end
end

