function fis=CreateFisUsingLookupTable(x, y,nmf,mftype)
    nx=size(x,2); % Feature counts
    
    %% Check Input Arguments
    if nargin<2 || isempty(nmf)
        nmf=3*ones(1,nx+1);
    end
    
    if nargin<3 || isempty(mftype)
        mftype=cell(1,nx+1);
        for i=1:nx+1
            mftype{i}='gaussmf';
        end
    end
    %% Create MFs
    A = cell(nx,1);
    for i=1:nx
        A{i}=CreateMembershipFunctions(x(:,i),nmf(i),mftype{i}); %A{i} is MFs for input i
    end

    B=CreateMembershipFunctions(y,nmf(end),mftype{end}); %B is MFs for output

    %% Create Rules Matrix
    if nx>1
        Ssize = nmf(1:end-1);
    else
        Ssize = [nmf(1) 1];
    end
    S = cell(Ssize);
    S = S(:);

    nRules=numel(S);
    for r=1:nRules
        S{r}=zeros(1,nmf(end));
    end

    %% Calculate Score of rules
    XIND = zeros(nRules,nx);
    for r=1:nRules
        xind=cell(1,nx);
        [xind{1:nx}] = ind2sub(Ssize,r);
        xind = cell2mat(xind);
        XIND(r,:)=xind;
        for yind=1:nmf(end)
            bmf = B{yind,1};
            bparam = B{yind,2};

            s = feval(bmf,y,bparam);
            for i=1:nx
                amf = A{i}{xind(i),1};
                aparm=A{i}{xind(i),2};
                s=s.*feval(amf,x(:,i),aparm);
            end
            % s=max(s);
            s=sum(s);
            S{r}(yind)=s;
        end
    end
    %% Delete Extra rules

    R = zeros(nRules,1);
    keep = true(size(S));
    for r=1:nRules
        [Smax, R(r)]=max(S{r});
        if Smax == 0
            keep(r) = false;
        end
    end
    
    Rules = [XIND R];
    Rules(:,end+1)=1;
    Rules(:,end+1)=1;
    Rules = Rules(keep,:);
    
    %% Create fis
    fis = mamfis('Name', 'Lookup Table FIS');
    for i=1:nx;
        fis=addInput(fis, [min(x(:,i)) max(x(:,i))],'Name', ['x' num2str(i)]);
        for j=1:nmf(i)
            fis = addMF(fis, ['x' num2str(i)], A{i}{j,1}, A{i}{j,2}, 'Name',...
                ['A(' num2str(i) ',' num2str(j) ')']);
        end
    end

    fis = addOutput(fis, [min(y) max(y)], 'Name', 'y');
    for bi=1:nmf(end)
        fis = addMF(fis, 'y', B{bi,1}, B{bi,2}, 'Name', ['B' num2str(bi)]);
    end

    fis=addrule(fis,Rules);
end