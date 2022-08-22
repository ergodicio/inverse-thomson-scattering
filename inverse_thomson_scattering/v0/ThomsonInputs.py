import numpy as np
from numpy.linalg import inv
import scipy.interpolate as sp
import inspect
from inverse_thomson_scattering.v0.numDistFunc import NumDistFunc

class ThomsonInputs:

    amp1 = dict([])
    amp2 = dict([])
    lam = dict([])
    Te = dict([])
    Z = dict([])
    ne = dict([])
    m = dict([])
    fe = dict([])
    blur = dict([])
    specCurvature = dict([])
    fitprops = dict([])

    def __init__(self,*args):
    #creates the ThomsonInputs object. standard inputs have 4 properties. "Active" is 1 if the variable it being fit or 0 otherwise. "Value" is the initial value. "Location" is the index where the variable can be found in the minimized array x. "Bound" are the set of default lower and upper bounds for the minimizer.
        self.amp1['Location']=0
        self.amp1['Bounds']=np.array([[0],[10]])
            
        self.amp2['Location']=0
        self.amp2['Bounds']=np.array([[0],[10]])
            
        self.lam['Location']=0
        self.lam['Bounds']=np.array([[523],[528]])
            
        self.Te['Location']=0
        self.Te['Bounds']=np.array([[0.01],[3]])
            
        self.Z['Location']=0
        self.Z['Bounds']=np.array([[1],[25]])
            
        self.ne['Location']=0
        self.ne['Bounds']=np.array([[.001],[2]])
            
        self.m['Location']=0
        self.m['Bounds']=np.array([[2],[5]])
            
        self.fe['Value']=[]
        self.fe['Location']=0
            
        #fit props is given a dummy Active properties to help the genX and setTo and sameActiveState methods blur and specCurve are also tentatively set to inactive inorder for the code to run without them (6/1/21)
        self.blur['Active']=0
        self.specCurvature['Active']=0
        self.fitprops['Active']=0
            
        if len(args) == 0:
            self.amp1['Active']=0
            self.amp1['Value']=[]

            self.amp2['Active']=0
            self.amp2['Value']=[]

            self.lam['Active']=0
            self.lam['Value']=[]
            
            self.Te['Active']=0
            self.Te['Value']=[]
            
            self.Z['Active']=0
            self.Z['Value']=[]
            
            self.ne['Active']=0
            self.ne['Value']=[]
            
            self.m['Active']=0
            self.m['Value']=[]

            self.fe['Active']=0
            self.fe['Length']=64
            self.fe['Type']='DLM'
            self.fe['Bounds']=np.ones([2,self.fe['Length']])*np.array([[-100],[-.5]])
            
        elif len(args) == 8:
            self.amp1['Active']=0
            self.amp1['Value']=args[4]

            self.amp2['Active']=0
            self.amp2['Value']=args[5]

            self.lam['Active']=0
            self.lam['Value']=args[0]
            
            self.Te['Active']=0
            self.Te['Value']=args[2]
            
            self.Z['Active']=0
            self.Z['Value']=args[6]
            
            self.ne['Active']=0
            self.ne['Value']=args[1]
            
            self.m['Active']=0
            self.m['Value']=args[3]

            self.fe['Active']=0
            self.fe['Length']=64
            self.fe['Type']=args[7]
            self.fe['Bounds']=np.ones([2,self.fe['Length']])*np.array([[-100],[-.5]])
            
            self.blur['Value']=[]
            
        elif len(args) == 14: #was used by OMEGAGUI this assumes numerical inputs have already beconverted from string to float
            self.amp1['Active']=args[8]
            self.amp1['Value']=args[9]

            self.amp2['Active']=args[10]
            self.amp2['Value']=args[11]

            self.lam['Active']=args[0]
            self.lam['Value']=args[1]
            
            self.Te['Active']=args[4]
            self.Te['Value']=args[5]
            
            self.Z['Active']=args[12]
            self.Z['Value']=args[13]
            
            self.ne['Active']=args[2]
            self.ne['Value']=args[3]
            
            self.m['Active']=args[7]
            self.m['Value']=args[7]

            self.fe['Active']=0
            self.fe['Length']=64
            self.blur['Value']=[]
            
        elif len(args) == 17: #was used by ANGTSDATAFITTERGUI this assumes numerical inputs have already beconverted from string to float
            self.amp1['Active']=args[8]
            self.amp1['Value']=args[9]

            self.amp2['Active']=args[10]
            self.amp2['Value']=args[11]

            self.lam['Active']=args[0]
            self.lam['Value']=args[1]
            
            self.Te['Active']=args[4]
            self.Te['Value']=args[5]
            
            self.Z['Active']=args[12]
            self.Z['Value']=args[13]
            
            self.ne['Active']=args[2]
            self.ne['Value']=args[3]
            
            self.m['Active']=args[6]
            self.m['Value']=args[7]

            self.fe['Active']=args[14]
            self.fe['Length']=args[15]
            self.fe['Type']=args[16]
            self.fe['Bounds']=np.ones([2,self.fe['Length']])*np.array([[-100],[-.5]])

        
    #these 2 methods are functions to add the blur and curvature properties which are special use proterties and therefore non-standard
    def addBlur(self,blurActive,blurValue):
        self.blur['Active']=blurActive
        self.blur['Value']=blurValue
        self.blur['Location']=0
        self.blur['Bounds']=np.array([[0],[10]])
        
    def addCurvature(self,curvActive,curvValue):
        self.specCurvature['Active']=curvActive
        self.specCurvature['Value']=curvValue
        self.specCurvature['Location']=0
        self.specCurvature['Bounds']=np.array([[.1],[10]])
        
    # add gradient percentage values to ne and Te
    def addGradients(self,TeActive,TeValue,neActive,neValue):
        if TeActive:
            self.Te['gradient']=TeValue
            
        if neActive:
            self.ne['gradient']=neValue
    
    # This method returns the minimizer array x and assigns all the location values to the location property. updated for python aliasing 8-11-22
    def genX(self):
        #props=vars(self);
        props = [attr[0] for attr in inspect.getmembers(self) if not attr[0].startswith('__') and not inspect.ismethod(attr[1])]
        #print(props)
        x=[]
        lb=[]
        ub=[]
        for i, prop in enumerate(props):
            if getattr(self,prop)['Active']:
                x.append(getattr(self,prop)['Value'])
                lb=np.append(lb,getattr(self,prop)['Bounds'][0,:],axis=0)
                ub=np.append(ub,getattr(self,prop)['Bounds'][1,:],axis=0)
                #ub.append(self.(prop)['Bounds'][1,:])
                getattr(self,prop)['Location']=len(x)-1
#        for i, prop in enumerate(props):
#            if self.(prop)['Active']
#                x.append(self.(prop)['Value'])
#                lb=np.append(lb,self.(prop)['Bounds'][0,:],axis=1)
#                ub=np.append(ub,self.(prop)['Bounds'][1,:],axis=1)
#                #ub.append(self.(prop)['Bounds'][1,:])
#                self.(prop)['Location']=i
        return x, lb, ub
    
    # populate the Value field of fe from the other fields
    def initFe(self,xie):
        if self.fe['Type']=='DLM':
            self.fe['Value'] = np.log( NumDistFunc( [self.fe['Type'], self.m['Value']], xie, self.fe['Type']))
            
        elif self.fe['Type']=='Fourkal':
            self.fe['Value'] = np.log( NumDistFunc( [self.fe['Type'], self.m['Value'], self.Z['Value']], xie, self.fe['Type']))
            
        elif self.fe['Type']=='SpitzerDLM':
            self.fe['Value'] = np.log( NumDistFunc( [self.fe['Type'], self.m['Value'], self.fe['theta'], self.fe['delT']], xie, self.fe['Type']))
            
        elif self.fe['Type']=='MYDLM': #This will eventually need another parameter for density gradient
            self.fe['Value'] = np.log( NumDistFunc( [self.fe['Type'], self.m['Value'], self.fe['theta'], self.fe['delT']], xie, self.fe['Type']))
        else :
            raise NameError('Unrecognized distribtuion function type')
            
        self.fe['Value'][self.fe['Value']<=-100]=-99
        
    #This method returns the inputs for approxThomson3 and 4, i.e. Te ne lam and fecur
    def genTS(self,x,*args):
            
        if self.Te['Active']:
            Te=x[self.Te['Location']]
        else:
            Te=self.Te['Value']

        if self.ne['Active']:
            ne=x[self.ne['Location']]
        else:
            ne=self.ne['Value']

        if self.lam['Active']:
            lam=x[self.lam['Location']]
        else:
            lam=self.lam['Value']

        if self.fe['Active']:
            fecur=x[self.fe['Location']-self.fe['Length']+1:self.fe['Location']]
        elif len(args)!=0 and self.m['Active']:
            self.m['Value']=x[self.m['Location']]
            self.initFe(args[0])
            fecur=self.fe['Value']
        elif len(args)!=0:
            #modified 4/8/22 added a check and interpolation if the length of xie and fe do not match
            #further modified 4/25/22 to restore opperation of spitzer testing
            #if len(args[0])==np.shape(self.fe['Value'])[1]:
            if len(args[0])==len(self.fe['Value']):
                fecur=self.fe['Value']
            elif 'vnorm' in self.fe:
                #The logarithms are off in this section which only makes sense if there are differences in the spitzer testing code 8-11-22
                fecur=sp.interp1d(self.fe['vnorm'],self.fe['Value'],'linear', bounds_error=False, fill_value=-np.inf)
                fecur = fecur(args[0])
                fnorm=np.trapz(fecur,args[0]) #not sure this is the right place to renormalize
                fecur=np.log(fecur/fnorm)
                self.fe.Value=fecur
            else:
                raise NameError('Distribtuion function and normalized phase velocity must be equal lengths')

        else:
            fecur=self.fe['Value']

        return Te, ne, lam, fecur
        
    #converts the Te and ne values into vectors distributed within the relevent ranges
    def genGradients(self,Te,ne,*args):
        if len(args) != 0:
            arlen=args[0]
        else:
            arlen=10
            
        if 'gradient' in self.Te.keys() and 'gradient' in self.ne.keys():
            Te=np.linspace((1-self.Te['gradient']/200)*Te,(1+self.Te['gradient']/200)*Te,arlen)
            ne=np.linspace((1-self.ne['gradient']/200)*ne,(1+self.ne['gradient']/200)*ne,arlen)
        elif 'gradient' in self.Te.keys():
            Te=np.linspace((1-self.Te['gradient']/200)*Te,(1+self.Te['gradient']/200)*Te,arlen)
            ne=ne*np.ones(len(ne),arlen)
        elif 'gradient' in self.ne.keys():
            ne=np.linspace((1-self.ne['gradient']/200)*ne,(1+self.ne['gradient']/200)*ne,arlen)
            Te=Te*np.ones(len(Te),arlen)
            
        return Te, ne
    
    #This method returns the rest of the parameters required for the remainder of the fitter, i.e amp1 amp2 blur. Future properties can be added here
    def genRest(self,x):
        if self.amp1['Active']:
            amp1=x[self.amp1['Location']]
        else:
            amp1=self.amp1['Value']
        
        if self.amp2['Active']:
            amp2=x[self.amp2['Location']]
        elif self.amp1['Active']:
            amp2=x[self.amp1['Location']]
        else:
            amp2=self.amp2['Value']
            
        if self.blur['Active']:
            blur=x[self.blur['Location']]
        else:
            blur=self.blur['Value']

        return amp1, amp2, blur
    
    #This method returns the object with all active fit parameters reset to the values given
    def setTo(self,x):
        
        #props=vars(self)
        props = [attr[0] for attr in inspect.getmembers(self) if not attr[0].startswith('__') and not inspect.ismethod(attr[1])]
        #for i, prop in enumerate(props):
        #    if self.(prop)['Active']:
        #        self.(prop)['Value'] = x[self.(prop)['Location']]
        for i, prop in enumerate(props):
            if getattr(self,prop)['Active']:
                getattr(self,prop)['Value'] = x[getattr(self,prop)['Location']]
                                         
        if self.fe['Active']:
            self.fe['Value']=x[self.fe['Location']-self.fe['Length']+1:self.fe['Location']]
                                         

    #This method adds Error fields to each of the active fit parameters
    def genErrors(self):
        #props=vars(self)
        props = [attr[0] for attr in inspect.getmembers(self) if not attr[0].startswith('__') and not inspect.ismethod(attr[1])]
        if 'hess' in self.fitprops.keys():
            errormat=np.real(np.diag(np.sqrt(inv(self.fitprops['hess']))))
            
            for i, prop in enumerate(props):
                if getattr(self,prop)['Active']:
                    getattr(self,prop)['Error']=errormat[getattr(self,prop)['Location']]
            
            if self.fe['Active']:
                self.fe['Error']=errormat[self.fe['Location']-self.fe['Length']+1:self.fe['Location']]
                
        else:
            print('Errors can not be calculated. No Hessian present.')
        
    #compares the states of each properties in the two instances of the ThomsonInputs object as well as a the length of the distribtuion function. Return a true-false value saying if they have the same states
    def sameActiveState(obj1,obj2):
        #props= vars(obj1)
        props = [attr[0] for attr in inspect.getmembers(obj1) if not attr[0].startswith('__') and not inspect.ismethod(attr[1])]
        c=0
        for i, prop in enumerate(props):
                if getattr(obj1,prop)['Active'] == getattr(obj2,prop)['Active']:
                    c=c+1
        
        if c==len(props):
            if obj1.fe['Active']:
                if obj1.fe['Length']==obj2.fe['Length']:
                    tf=True
                else:
                    tf=False

            else:
                tf=True
        else:
            tf=False
        
        return tf
    
    #changed name from matlab version to prevent conflict with built in print
    def disp(self):
        #props= vars(self)
        props = [attr[0] for attr in inspect.getmembers(self) if not attr[0].startswith('__') and not inspect.ismethod(attr[1])]
        
        for i, prop in enumerate(props):
            if getattr(self,prop)['Active']:
                print(prop, ': ', str(getattr(self,prop)['Value']))
