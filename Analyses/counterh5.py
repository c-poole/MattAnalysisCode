import os
import glob

from numpy import *

#sys.path.append('Z:\\Public\\Hybrid\\Useful Code\\Python')
#from counter_analysis import counterData
import matplotlib
#matplotlib.use('png')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erf
from sklearn import mixture
import h5py
import pandas as pd
import seaborn as sbs



gmix = mixture.GMM(n_components=2)

class counter():
    data=[]

    def __init__(self,folder='./',experiment=0):
        self.dir=os.getcwd().split(os.sep)[-1]
        self.filename = glob.glob(folder+'*.hdf5')[0]
        self.h5file = h5py.File(self.filename)
        self.measurements = self.h5file['settings/experiment/measurementsPerIteration'].value
        self.iterations = len(self.h5file['experiments/{}/iterations/'.format(experiment)].items())
        self.iVars=[]
        self.varPath=self.h5file['experiments/{}/iterations/0/variables'.format(experiment)]
        ROdrops = self.varPath['throwaway_bins'].value
        RObins = self.varPath['measurement_bins'].value
        max_meas = 0
        for i in self.h5file['experiments/{}/iterations/'.format(experiment)].iteritems():
            max_meas = max(max_meas, len(i[1]['measurements/'].items()))
        print ROdrops
        print RObins
        self.measurements = max_meas
        self.rawData=empty((self.iterations,self.measurements,2,RObins))
        self.shotData=empty((self.iterations,self.measurements,2))
        print self.rawData.shape
        if self.iterations > 1:
            for i in self.h5file['settings/experiment/independentVariables/'].iteritems():
                try:
                    tmp=eval(i[1]['function'].value)
                    if (type(tmp) == list) | (type(tmp) == ndarray) | (type(tmp) == tuple): self.iVars.append((i[0],i[1]['function'].value))
                except NameError as e:
                    print e
        for i in self.h5file['experiments/{}/iterations/'.format(experiment)].iteritems():
            j = 0
            for m in i[1]['measurements/'].iteritems():
                try:
                    temp=array(m[1]['data/counter/data'].value[0])
                    self.rawData[int(i[0]),j,0]=temp[ROdrops:RObins + ROdrops]
                    self.rawData[int(i[0]),j,1]=temp[-RObins:]
                    self.shotData[int(i[0]),j,0]=temp[ROdrops:RObins + ROdrops].sum()
                    self.shotData[int(i[0]),j,1]=temp[-RObins:].sum()
                    j += 1
                except KeyError:
                    pass
                except IndexError:
                    print i[0]
                    print j
                    print m[0]
                    raise IndexError
        if self.iterations > 1:
            df=0
            j=0
            for i in eval(self.iVars[0][1]):
				try:
					for shot in range(2):
						d={self.iVars[0][0]: i, "Shot": shot, "Counts": self.shotData[j,:,shot]}
						if df==0:
							df=1
							self.DataFrame=pd.DataFrame(data=d)
						else:
							tmpDF=pd.DataFrame(data=d)
							self.DataFrame=self.DataFrame.append(tmpDF,ignore_index=True)
					j+=1
				except IndexError:
					print "iVar index: {} is out of range".format(i)
					break

        self.cuts=[nan,nan,nan]
        self.rload=[1,1,1]
        self.retention=[1,1]

    def vplot(self):
        plt.clf()
        sbs.violinplot(x=self.iVars[0][0],y='Counts', hue = 'Shot', data=self.DataFrame, split=True, inner='stick')


    def get_cuts(self,hbins=40,save_cuts=True,itr=0):

        #=====================Fit Functions=================
        def intersection(A0,A1,m0,m1,s0,s1):
            return (m1*s0**2-m0*s1**2-sqrt(s0**2*s1**2*(m0**2-2*m0*m1+m1**2+2*log(A0/A1)*(s1**2-s0**2))))/(s0**2-s1**2)

        def area(A0,A1,m0,m1,s0,s1):
            return sqrt(pi/2)*(A0*s0+A0*s0*erf(m0/sqrt(2)/s0)+A1*s1+A1*s1*erf(m1/sqrt(2)/s1))

        # Normed Overlap for arbitrary cut point
        def overlap(xc,A0,A1,m0,m1,s0,s1):
            err0=A0*sqrt(pi/2)*s0*(1-erf((xc-m0)/sqrt(2)/s0))
            err1=A1*sqrt(pi/2)*s1*(erf((xc-m1)/sqrt(2)/s1)+erf(m1/sqrt(2)/s1))
            return (err0+err1)/area(A0,A1,m0,m1,s0,s1)

        # Relative Fraction in 1
        def frac(A0,A1,m0,m1,s0,s1):
            return 1/(1+A0*s0*(1+erf(m0/sqrt(2)/s0))/A1/s1/(1+erf(m1/sqrt(2)/s1)))

        def dblgauss(x,A0,A1,m0,m1,s0,s1):
            return A0*exp(-(x-m0)**2 / (2*s0**2)) +  A1*exp(-(x-m1)**2 / (2*s1**2))


        #====================================================


        plt.close('all')
        titles=['Shot 1,Cut={:.2f}','Shot 2,Cut={:.2f}', 'PS Shot 2,Cut={:.2f}']
        f, axarr = plt.subplots(1,3,figsize=(12,6))
        for i in range(2):
            tmp=self.shotData[itr,:,i]
            gmix.fit(array([tmp]).transpose())
            est=[gmix.weights_.max()/10,gmix.weights_.min()/10,gmix.means_.min(),gmix.means_.max(),sqrt(gmix.means_.min()),sqrt(gmix.means_.max())]
            h = histogram(tmp,normed=True,bins=hbins)
            axarr[i].hist(tmp,bins=hbins,histtype='step',normed=True)
            try:
                #Shot 0
                #======
                #Optimise Fit to Double-Gaussian Distribution Using Best-Fit Starting Values
                popt,pcov = curve_fit(dblgauss,h[1][1:],h[0],est)
                #popt=[A0,A1,m0,m1,s0,s1] : Absolute value
                popt=abs(popt)
                self.cuts[i]=intersection(*popt)
                axarr[i].plot(h[1][1:]-.5,dblgauss(h[1][1:],*popt))
                axarr[i].plot(array([1,1])*self.cuts[i],axarr[i].get_ylim(),'k')
                self.rload[i]=frac(*popt)
                axarr[i].plot(h[1][1:]-.5,dblgauss(h[1][1:],*popt))
                axarr[i].set_title(titles[i].format(self.cuts[i]))
            except RuntimeError:
                self.cuts[i]=nan
                self.rload[i]=nan
        self.retention[0]=self.rload[1]/self.rload[0]


        self.cut()
        tmp=self.shotData[0,where((self.shotData[0,:,0]>self.cuts[0])*1.0==1.0),1][0]
        gmix.fit(array([tmp]).transpose())
        est=[gmix.weights_.max()/10,gmix.weights_.min()/10,gmix.means_.min(),gmix.means_.max(),sqrt(gmix.means_.min()),sqrt(gmix.means_.max())]
        h = histogram(tmp,normed=True,bins=hbins)
        axarr[2].hist(tmp,bins=hbins,histtype='step',normed=True)
        try:
                #Shot 0
                #======
                #Optimise Fit to Double-Gaussian Distribution Using Best-Fit Starting Values
            popt,pcov = curve_fit(dblgauss,h[1][1:],h[0],est)
            #popt=[A0,A1,m0,m1,s0,s1] : Absolute value
            popt=abs(popt)
            self.cuts[2]=intersection(*popt)
            axarr[2].plot(h[1][1:]-.5,dblgauss(h[1][1:],*popt))
            axarr[2].plot(array([1,1])*self.cuts[i],axarr[i].get_ylim(),'k')
            self.rload[2]=frac(*popt)
            axarr[2].plot(h[1][1:]-.5,dblgauss(h[1][1:],*popt))
        except RuntimeError:
            self.cuts[2]=nan
            self.rload[2]=nan

        axarr[2].set_title(titles[2].format(self.cuts[2]))
        self.retention[1]=((tmp>self.cuts[2])*1.0).sum()/len(tmp)
        print self.cuts,self.rload, self.retention
        plt.suptitle(self.dir[:19].replace('_',' ')+' Calibration , Load Frac={:.1%}, Retention={:.1%}'.format(self.rload[0],self.retention[1]),size=16)
        plt.show()
        plt.savefig('../'+self.dir+'_CalCutPlots.pdf')
        if save_cuts==True: savetxt('../'+self.dir[:19]+'_Cuts.txt' ,concatenate((self.cuts,self.rload,self.retention)))

    def cut(self):
        rshape=self.rawData.shape
        out=zeros((rshape[0],rshape[1],2))
        #print out.shape

        if isnan(self.cuts[0]): self.load_cuts()
        for i in range(2):
            out[:,:,i]=self.shotData[:,:,i]>self.cuts[i]
        savetxt(self.dir+'_Binarized.txt',out.reshape(rshape[0],rshape[1]*2),header='Rload_cut={}'.format(self.retention[1]),fmt='%i')
        self.binData=out
        return out

    def load_cuts(self):
        files=glob.glob('../*_Cuts.txt')
        files.sort()
        try:
            tmp = loadtxt(files[-1])
            print tmp
            self.cuts=tmp[0:3]
            self.rload=tmp[3:6]
            self.retention=tmp[6:8]
            print self.cuts,self.rload, self.retention
        except IndexError:
            print 'Bad Cut File!'

    def hist3D(self,shot=0):
        mx=int(self.shotData[:,:,shot].max())
        plt.close('all')
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(self.iterations):
            tmp=histogram(self.shotData[i,:,shot],bins=mx/2)
            ax.bar(tmp[1][:-1],tmp[0],zs=eval(self.iVars[0][1])[i],zdir='y',alpha=.5)

        ax.set_xlabel('cts')
        ax.set_zlabel('N')
        #ax.set_ylabel(self.varVar[0])
        ax.view_init(elev=30,azim=38)
        #ax.set_title(self.varName[0]+' (' +self.varVar[0]+ ') Bin='+str(self.dt)+'ms')
        plt.show()
        #plt.savefig('Hist3D_'+self.dir+'_Bin_'+str(self.dt)+'ms.png')

    def binparscan(self, cut0=30, cut1=30, fit=True):
        if len(self.iVars)>1:

            p=(self.shotData[:,:,1] > cut1).sum(1)*1./(self.shotData[:,:,0] > cut0).sum(1)
            err=sqrt((1-p)*p/(self.shotData[:,:,0] > cut0).sum(1))
            it_scan = 0
            if self.iVars[0][0]=='shelve_state': it_scan = 1
            it_arr = eval(self.iVars[it_scan][1])
            it_len = len(it_arr)

            plt.clf()

            plt.errorbar(it_arr,p[:it_len],yerr=err[:it_len],label='F=3')
            plt.errorbar(it_arr,p[it_len:],yerr=err[it_len:],label='F=4')
            plt.xlabel(self.iVars[it_scan][0])
            plt.ylabel('Retention')


        if len(self.iVars) == 1:
            p=(self.shotData[:,:,1] > cut1).sum(1)*1./(self.shotData[:,:,0] > cut0).sum(1)
            err=sqrt((1-p)*p/(self.shotData[:,:,0] > cut0).sum(1))
            it_arr = eval(self.iVars[0][1])
            it_len = len(it_arr)



            plt.clf()
            plt.errorbar(it_arr,p,yerr=err, label = 'Data',fmt='.')

            if fit:
                sin_func = lambda t,f,A,b: abs(A)*sin(pi*f*(t))**2 + b
                popt,pcov = curve_fit(sin_func, it_arr, p, sigma=err, p0=[10,p.max()-p.min(),p.min()])
                plt.plot(linspace(it_arr[0],it_arr[-1],1000), sin_func(linspace(it_arr[0],it_arr[-1],1000),*popt),label = 'fit: frequency = {:.5f} kHz'.format(popt[0]))
                print 'A={:.5},b={:.5},piTime={}'.format(popt[1],popt[2],1/(2*popt[0]))
            plt.xlabel(self.iVars[0][0])

        plt.ylabel('Retention')
        plt.legend()
        return p,err

    def fitparscan(self,hbins=30,tr=0):
    # Currently just plots the overlap between Atom and Background
    #
    #
            #=====================Fit Functions=================
        def intersection(A0,A1,m0,m1,s0,s1):
            return (m1*s0**2-m0*s1**2-sqrt(s0**2*s1**2*(m0**2-2*m0*m1+m1**2+2*log(A0/A1)*(s1**2-s0**2))))/(s0**2-s1**2)

        def area(A0,A1,m0,m1,s0,s1):
            return sqrt(pi/2)*(A0*s0+A0*s0*erf(m0/sqrt(2)/s0)+A1*s1+A1*s1*erf(m1/sqrt(2)/s1))

        # Normed Overlap for arbitrary cut point
        def overlap(xc,A0,A1,m0,m1,s0,s1):
            err0=A0*sqrt(pi/2)*s0*(1-erf((xc-m0)/sqrt(2)/s0))
            err1=A1*sqrt(pi/2)*s1*(erf((xc-m1)/sqrt(2)/s1)+erf(m1/sqrt(2)/s1))
            return (err0+err1)/area(A0,A1,m0,m1,s0,s1)

        # Relative Fraction in 1
        def frac(A0,A1,m0,m1,s0,s1):
            return 1/(1+A0*s0*(1+erf(m0/sqrt(2)/s0))/A1/s1/(1+erf(m1/sqrt(2)/s1)))

        def dblgauss(x,A0,A1,m0,m1,s0,s1):
            return A0*exp(-(x-m0)**2 / (2*s0**2)) +  A1*exp(-(x-m1)**2 / (2*s1**2))


        #====================================================


        plt.close('all')
        out=zeros((2,self.shotData.shape[0]))
        perr=[]
        fracout=zeros((2,self.shotData.shape[0]))
        for i in range(out.shape[1]):
            for shot in range(2):
                tmp=self.shotData[i,:,shot]
                gmix.fit(array([tmp]).transpose())
                est=[gmix.weights_.max()/10,gmix.weights_.min()/10,gmix.means_.min(),gmix.means_.max(),sqrt(gmix.means_.min()),sqrt(gmix.means_.max())]
                h = histogram(tmp,normed=True,bins=hbins)
                try:
                    #Shot 0
                    #======
                    #Optimise Fit to Double-Gaussian Distribution Using Best-Fit Starting Values
                    popt,pcov = curve_fit(dblgauss,h[1][1:],h[0],est)
                    #popt=[A0,A1,m0,m1,s0,s1] : Absolute value
                    popt=abs(popt)

                    xc=intersection(*popt)

                    perr.append(overlap(xc,*sqrt(diag(pcov))))

                    if isnan(xc):
                        print 'Bad Cut on Shot: {} Iteration: {}'.format(shot,i)
                        out[shot,i]=nan
                        fracout[shot,i]=nan
                        #perr.append(1)
                    else:
                        out[shot,i]=overlap(xc,*popt)
                        if frac(*popt) < 1:
                            fracout[shot,i]=frac(*popt)
                        else:
                            fracout[shot,i]=1/frac(*popt)
                except (RuntimeError,RuntimeWarning):
                    print 'Bad fit on Shot: {} Iteration: {}'.format(shot,i)
                    out[shot,i]=nan
                    fracout[shot,i]=nan
        fracout=fracout[1]/fracout[0]
        print "Fracout=",fracout
        print "Perr[0]=", perr[::2]
        print "out=", out
        fracout[where(fracout>=1)]=nan
        if len(self.iVars)>1:
            out=out.reshape(len(self.varSpace[1]),len(self.varSpace[0]))
            if tr == 0:out=out.transpose()
            f,axarr = plt.subplots(2,sharex=True,figsize=(12,6))
            labels=['{} = {:.3f}'.format(self.varName[(tr+1)%2],self.varSpace[(tr+1)%2][i]) for i in range(len(self.varSpace[(tr+1)%2]))]
            axarr[0].errorbar(self.varSpace[tr],out,'.')
            axarr[0].set_title('Raw Data')
            axarr[1].errorbar(self.varSpace[tr],out/self.retention[1],'.')
            axarr[1].set_title('Scaled Data')
            axarr[1].set_xlabel(self.varName[tr])
        else:
            varSpace=eval(self.iVars[0][1])
            f,axarr = plt.subplots(3,sharex=True,figsize=(12,6))
            labels=[self.iVars[0][0]]
            axarr[0].plot(varSpace,out[0],'.')#, yerr = perr[::2])
            axarr[0].set_title('Background-Atom Overlap %: Shot 0')
            axarr[1].plot(varSpace,out[1],'.')#, yerr = perr[1::2])
            axarr[1].set_title('Background-Atom Overlap %: Shot 1')
            axarr[2].plot(varSpace,fracout,'.')#, yerr = perr[1::2])
            axarr[2].set_title('Post Experiment Retention')
            axarr[2].set_xlabel(self.iVars[0][0])


        plt.suptitle('Double Gaussian Fitted Parameter Scan')
        #plt.legend(labels,fontsize='small')
        plt.show()
        plt.savefig(self.dir+'_FitParScan1D.pdf')


        return out
