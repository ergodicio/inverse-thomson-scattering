classdef ThomsonInputs
    properties
        amp1
        amp2
        lam
        Te
        Z
        ne
        m
        fe
        %lnfe
        blur
        specCurvature
        fitprops
    end
    methods
        function obj = ThomsonInputs(varargin)
            %creates the ThomsonInputs object. standard inputs have 4
            %preperties. "Active" is 1 if the variable it being fit or 0
            %otherwise. "Value" is the initial value. "Location" is the
            %index where the variable can be found in the minimized array 
            %x. "Bound" are the set of default lower and upper bounds for
            %the minimizer.
            
            obj.amp1.Location=0;
            obj.amp1.Bounds=[0; 10];
            
            obj.amp2.Location=0;
            obj.amp2.Bounds=[0; 10];
            
            obj.lam.Location=0;
            obj.lam.Bounds=[523; 528];
            
            obj.Te.Location=0;
            obj.Te.Bounds=[0.01; 3];
            
            obj.Z.Location=0;
            obj.Z.Bounds=[1; 25];
            
            obj.ne.Location=0;
            obj.ne.Bounds=[.001; 2];
            
            obj.m.Location=0;
            obj.m.Bounds=[2; 5];
            
            obj.fe.Value=[];
            obj.fe.Location=0;
            
            %fit props is given a dummy Active properties to help the genX
            %and setTo and sameActiveState methods
            %blur and specCurve are also tentatively set to inactive inrder
            %for the code to run withou them (6/1/21)
            obj.blur.Active=0;
            obj.specCurvature.Active=0;
            obj.fitprops.Active=0;
            
            switch nargin
                case 0
                    
                    obj.amp1.Active=0;
                    obj.amp1.Value=[];

                    obj.amp2.Active=0;
                    obj.amp2.Value=[];

                    obj.lam.Active=0;
                    obj.lam.Value=[];

                    obj.Te.Active=0;
                    obj.Te.Value=[];

                    obj.Z.Active=0;
                    obj.Z.Value=[];

                    obj.ne.Active=0;
                    obj.ne.Value=[];

                    obj.m.Active=0;
                    obj.m.Value=[];

                    obj.fe.Active=0;
                    obj.fe.Length=64;
                    obj.fe.Type='DLM';
                    obj.fe.Bounds=repmat([-100;-.5],1,obj.fe.Length); 
            
                case 8
                    obj.amp1.Active=0;
                    obj.amp1.Value=varargin{5};

                    obj.amp2.Active=0;
                    obj.amp2.Value=varargin{6};

                    obj.lam.Active=0;
                    obj.lam.Value=varargin{1};

                    obj.Te.Active=0;
                    obj.Te.Value=varargin{3};

                    obj.Z.Active=0;
                    obj.Z.Value=varargin{7};

                    obj.ne.Active=0;
                    obj.ne.Value=varargin{2};

                    obj.m.Active=0;
                    obj.m.Value=varargin{4};

                    obj.fe.Active=0;
                    obj.fe.Length=64;
                    obj.fe.Type=varargin{8};
                    obj.fe.Bounds=repmat([-100;-.5],1,obj.fe.Length); 
                    
                    
                case 14 %used by OmegaGUI
                    obj.amp1.Active=varargin{9}.Value;
                    obj.amp1.Value=str2double(varargin{10}.String);

                    obj.amp2.Active=varargin{11}.Value;
                    obj.amp2.Value=str2double(varargin{12}.String);

                    obj.lam.Active=varargin{1}.Value;
                    obj.lam.Value=str2double(varargin{2}.String);

                    obj.Te.Active=varargin{5}.Value;
                    obj.Te.Value=str2double(varargin{6}.String);

                    obj.Z.Active=varargin{13}.Value;
                    obj.Z.Value=str2double(varargin{14}.String);

                    obj.ne.Active=varargin{3}.Value;
                    obj.ne.Value=str2double(varargin{4}.String);

                    obj.m.Active=varargin{7}.Value;
                    obj.m.Value=str2double(varargin{8}.String);
                    
                    obj.fe.Active=0;
                    obj.fe.Length=64; 
                    obj.blur.Value=[];
                    
                case 17 %Used by AngTSDataFitterGUI
                    %The expected order is toglam,boxlam,togne,boxne,togTe,...
                    %boxTe,togm,boxm,togamp1,boxamp1,togamp2,boxamp2,togZ,boxZ,...
                    %togfe,boxfe,boxfetype
                    obj.amp1.Active=varargin{9}.Value;
                    obj.amp1.Value=str2double(varargin{10}.String);

                    obj.amp2.Active=varargin{11}.Value;
                    obj.amp2.Value=str2double(varargin{12}.String);

                    obj.lam.Active=varargin{1}.Value;
                    obj.lam.Value=str2double(varargin{2}.String);

                    obj.Te.Active=varargin{5}.Value;
                    obj.Te.Value=str2double(varargin{6}.String);

                    obj.Z.Active=varargin{13}.Value;
                    obj.Z.Value=str2double(varargin{14}.String);

                    obj.ne.Active=varargin{3}.Value;
                    obj.ne.Value=str2double(varargin{4}.String);

                    obj.m.Active=varargin{7}.Value;
                    obj.m.Value=str2double(varargin{8}.String);

                    obj.fe.Active=varargin{15}.Value;
                    obj.fe.Length=str2double(varargin{16}.String);
                    obj.fe.Type=varargin{17}.String;
                    obj.fe.Bounds=repmat([-100;-.5],1,obj.fe.Length); 
            end
            
        end
        
        %these 2 methods are functions to add the blur and curvature properties which
        %are special use porterties and therefore non-standard
        function obj = addBlur(obj,blurActive,blurValue)
            obj.blur.Active=blurActive;
            obj.blur.Value=blurValue;
            obj.blur.Location=0;
            obj.blur.Bounds=[0; 10];
        end
        
        function obj = addCurvature(obj,curvActive,curvValue)
            obj.specCurvature.Active=curvActive;
            obj.specCurvature.Value=curvValue;
            obj.specCurvature.Location=0;
            obj.specCurvature.Bounds=[.1; 10];
        end
        
        function obj = addGradients(obj,TeActive,TeValue,neActive,neValue)
            %add gradient percentage values to ne and Te
            if TeActive
                obj.Te.gradient=TeValue;
            end
            if neActive
                obj.ne.gradient=neValue;
            end
        end
        
        function [obj,x,lb,ub] = genX(obj)
            %This method returns the minimizer array x and assigns all the
            %location values to the location property
            props=properties(obj);
            x=[];
            lb=[];
            ub=[];
            for i=1:length(props)
                if obj.(props{i}).Active
                    x=[x obj.(props{i}).Value];
                    lb=[lb obj.(props{i}).Bounds(1,:)];
                    ub=[ub obj.(props{i}).Bounds(2,:)];
                    obj.(props{i}).Location=length(x);
                end
            end
        end
        
        function obj = initFe(obj,xie)
            %populate the Value field of fe from the other fields
            if strcmp(obj.fe.Type,'DLM')
                obj.fe.Value=log(NumDistFunc({obj.fe.Type,obj.m.Value},xie,...
                    obj.fe.Type)');
            elseif strcmp(obj.fe.Type,'Fourkal')
                obj.fe.Value=log(NumDistFunc({obj.fe.Type,obj.m.Value,...
                    obj.Z.Value},xie,obj.fe.Type)');
            elseif strcmp(obj.fe.Type,'SpitzerDLM')
                obj.fe.Value=log(NumDistFunc({obj.fe.Type,obj.m.Value,...
                    obj.fe.theta,obj.fe.delT},xie,obj.fe.Type)');
            elseif strcmp(obj.fe.Type,'MYDLM')%this will eventualy need another parameter for density gradient
                obj.fe.Value=log(NumDistFunc({obj.fe.Type,obj.m.Value,...
                    obj.fe.theta,obj.fe.delT},xie,obj.fe.Type)');
            else
                error('Unrecognized distribtuion function type')
            end
            obj.fe.Value(obj.fe.Value<=-100)=-99;
        end
        
        function [Te,ne,lam,fecur] = genTS(obj,x,varargin)
            %This method returns the inputs for approxThomson3, i.e. Te ne
            %lam and fecur
            if obj.Te.Active
                Te=x(obj.Te.Location);
            else
                Te=obj.Te.Value;
            end
            if obj.ne.Active
                ne=x(obj.ne.Location);
            else
                ne=obj.ne.Value;
            end
            if obj.lam.Active
                lam=x(obj.lam.Location);
            else
                lam=obj.lam.Value;
            end
            if obj.fe.Active
                fecur=x(obj.fe.Location-obj.fe.Length+1:obj.fe.Location);
            elseif ~isempty(varargin) && obj.m.Active
                obj.m.Value=x(obj.m.Location);
                obj=obj.initFe(varargin{1});
                fecur=obj.fe.Value;
            elseif ~isempty(varargin)
                %modified 4/8/22 added a check and interpolation if the
                %length of xie and fe do not match
                %further modified 4/25/22 to restore opperation of spitzer
                %testing
                if length(varargin{1})==length(obj.fe.Value)
                    fecur=obj.fe.Value;
                elseif isfield(obj.fe,'vnorm')
                    fecur=interp1(obj.fe.vnorm,obj.fe.Value,varargin{1});
                    fnorm=trapz(varargin{1},fecur);%not sure this is the right place to renormalize
                    fecur=log(bsxfun(@rdivide,fecur,fnorm))';
                    obj.fe.Value=fecur;
                else
                    error('Distribtuion function and normalized phase velocity must be equal lengths')
                end
            else
                fecur=obj.fe.Value;
            end
        end
        
        function [Te,ne] = genGradients(obj,Te,ne,varargin)
            %converts the Te and ne values into vectors distributed within
            %the relevent ranges
            if ~isempty(varargin)
                arlen=varargin{1};
            else
                arlen=10;
            end
            
            if isfield(obj.Te,'gradient') && isfield(obj.ne,'gradient')
                Te=linspace((1-obj.Te.gradient/200)*Te,(1+obj.Te.gradient/200)*Te,arlen);
                ne=linspace((1-obj.ne.gradient/200)*ne,(1+obj.ne.gradient/200)*ne,arlen);
            elseif isfield(obj.Te,'gradient')
                Te=linspace((1-obj.Te.gradient/200)*Te,(1+obj.Te.gradient/200)*Te,arlen);
                ne=repmat(ne,1,arlen);
            elseif isfield(obj.ne,'gradient')
                ne=linspace((1-obj.ne.gradient/200)*ne,(1+obj.ne.gradient/200)*ne,arlen);
                Te=repmat(Te,1,arlen);
            end

        end
        function [amp1,amp2,blur] = genRest(obj,x)
            %This method returns the rest of the parameters required for
            %the remainder of the fitter, i.e amp1 amp2 blur. Future
            %properties can be added here
            if obj.amp1.Active
                amp1=x(obj.amp1.Location);
            else
                amp1=obj.amp1.Value;
            end
            if obj.amp2.Active
                amp2=x(obj.amp2.Location);
            elseif obj.amp1.Active
                amp2=x(obj.amp1.Location);
            else
                amp2=obj.amp2.Value;
            end
            if obj.blur.Active
                blur=x(obj.blur.Location);
            else
                blur=obj.blur.Value;
            end
        end
        
        function obj = setTo(obj,x)
            %This method returns the object with all active fit parameters
            %reset to the values given
            props=properties(obj);
            
            for i=1:length(props)
                if obj.(props{i}).Active
                    obj.(props{i}).Value=x(obj.(props{i}).Location);
                end
            end
            if obj.fe.Active
                obj.fe.Value=x(obj.fe.Location-obj.fe.Length+1:obj.fe.Location);
            end
        end
        
        function obj = genErrors(obj)
            %This method adds Error fields to each of the active fit
            %parameters
            props=properties(obj);
            if isfield(obj.fitprops,'hess')
                errormat=real(diag(sqrt(inv(obj.fitprops.hess))));
                for i=1:length(props)
                    if obj.(props{i}).Active
                        obj.(props{i}).Error=errormat(obj.(props{i}).Location);
                    end
                end
                if obj.fe.Active
                    %obj.fe.Error=[obj.fe.Value-errormat(obj.fe.Location-obj.fe.Length+1:obj.fe.Location)'; ...
                    %    obj.fe.Value+errormat(obj.fe.Location-obj.fe.Length+1:obj.fe.Location)'];
                    obj.fe.Error=errormat(obj.fe.Location-obj.fe.Length+1:obj.fe.Location)';
                end
            else
                disp('Errors can not be calculated. No Hessian present');
            end
        end
        
        function tf = sameActiveState(obj1,obj2)
            %compares the states of each properties in the two instances of
            %the ThomsonInputs object as well as a the length of the
            %distribtuion function. Return a true-false value saying if
            %they have the same states
            props=properties(obj1);
            c=0;
            for i=1:length(props)
                if ~isempty(obj1.(props{i})) && ~isempty(obj2.(props{i}))...
                        && obj1.(props{i}).Active==obj2.(props{i}).Active
                    c=c+1;
                end
            end
            if c==length(props)
                if obj1.fe.Active
                    if obj1.fe.Length==obj2.fe.Length
                        tf=true;
                    else
                        tf=false;
                    end
                else
                    tf=true;
                end
            else
                tf=false;
            end
        end
        function print(obj)
            props=properties(obj);
            for i=1:length(props)
                if obj.(props{i}).Active
                    disp([props{i} ':  ' num2str(obj.(props{i}).Value)])
                end
            end
        end
    end
end