from utils import *

import pickle
import os
import time
import numpy as np
import polychrom


from polychrom import polymerutils
from polychrom import forces
from polychrom import forcekits
from polychrom.simulation import Simulation
from polychrom.starting_conformations import grow_cubic
from polychrom.hdf5_format import HDF5Reporter, list_URIs, load_URI, load_hdf5_file

import simtk.openmm 
import shutil


import warnings
import h5py 
import glob
import matplotlib.pyplot as plt

import h5py

import numpy as np
import pandas as pd
import h5py 
import matplotlib.pyplot as plt
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-l', '--lifetime', default=900)
parser.add_argument('-r', '--loading-range', default=500)
parser.add_argument('-s', '--separation', default=800)
parser.add_argument('-d', '--drift', default=0.9)
parser.add_argument('-c', '--capture-probability', default=0.9)
parser.add_argument('-f', '--free-probability', default=0.003)
parser.add_argument('--left-ctcf', default=1)
parser.add_argument('--right-ctcf', default=9999)
parser.add_argument('--trajectory-length', default=10000)
parser.add_argument('--enforce-ctcf', default=1)


args_trml = parser.parse_args()


dir_name ='20reps_lr' + str(args_trml.loading_range) + '_sep' + str(args_trml.separation) + '_lif' + str(args_trml.lifetime) + \
    '_dr' + str(args_trml.drift) + '_cp' + str(args_trml.capture_probability) + \
    '_rls' + str(args_trml.free_probability) + '_lctcf' + str(args_trml.left_ctcf) + \
    '_rctcf' + str(args_trml.right_ctcf)+ '_time' + str(args_trml.trajectory_length) + str(args_trml.enforce_ctcf)



root = '/rds/general/user/kn119/home/loop_extrusion/jet_angles_cohesin_blocks/'
os.makedirs(root, exist_ok = True)
root = os.path.join(root, dir_name)
os.makedirs(root, exist_ok = True)


myfile_name = 'no.h5'

destination = 'no/'

loading_range = int(args_trml.loading_range)//2

N1 = 5000 # size of one system 
M = 20
N = N1 * M 

LS = (N1//2 - loading_range, N1//2 + loading_range)

LIFETIME = int(args_trml.lifetime)
SEPARATION = int(args_trml.separation)
LEFNum = N // SEPARATION 
trajectoryLength = int(args_trml.trajectory_length)

my_plot_title = 'No CTCF'
TADs = [int(args_trml.left_ctcf)-50*i for i in range(int(args_trml.enforce_ctcf))]
TADs = [1, 4999]

ctcfLeftRelease = {}
ctcfRightRelease = {}
ctcfLeftCapture = {}
ctcfRightCapture = {}

for i in range(M):
    for tad in TADs:
        pos = i * N1 + tad 
        ctcfLeftCapture[pos] = float(args_trml.capture_probability)  # 80% capture probability 
        ctcfLeftRelease[pos] = float(args_trml.free_probability)  # hold it for ~300 blocks on average
        ctcfRightCapture[pos] = float(args_trml.capture_probability)
        ctcfRightRelease[pos] =  float(args_trml.free_probability)



args = {}
args["ctcfRelease"] = {-1:ctcfLeftRelease, 1:ctcfRightRelease}
args["ctcfCapture"] = {-1:ctcfLeftCapture, 1:ctcfRightCapture}        
args["N"] = N 
args["LIFETIME"] = LIFETIME
args["LIFETIME_STALLED"] = LIFETIME // 20 # 
args["Probabilistic_Loading"] = False
args["Probabilistic_Profile"] = None
args["Site_Loading"] = True
args['N1'] = N1
args['LS'] = LS
args['M'] = M
args['drift']=float(args_trml.drift)

occupied = np.zeros(N)
occupied[0] = 1
occupied[-1] = 1 

cohesins = []

occupied = np.zeros(N)
occupied[0] = 1
occupied[-1] = 1 
cohesins = []

for i in range(LEFNum):
    loadOne(cohesins,occupied, args)
    
with h5py.File(os.path.join(root,myfile_name), mode='w') as myfile:
    
    dset = myfile.create_dataset("positions", 
                                 shape=(trajectoryLength, LEFNum, 2), 
                                 dtype=np.int32, 
                                 compression="gzip")
    steps = 50    # saving in 50 chunks because the whole trajectory may be large 
    bins = np.linspace(0, trajectoryLength, steps, dtype=int) # chunks boundaries 
    for st,end in zip(bins[:-1], bins[1:]):
        cur = []
        for i in range(st, end):
            coh_loaded = len(cohesins)
            if coh_loaded <= LEFNum:
                for fc in range(LEFNum - coh_loaded):
                    loadOne(cohesins, occupied, args)
            translocate(cohesins, occupied, args)  # actual step of LEF dynamics 
            positions = [(cohesin.left.pos, cohesin.right.pos) for cohesin in cohesins]
            if len(cohesins) < LEFNum:
                positions.extend([(args["N1"]//2,args["N1"]//2+1) for i in range(LEFNum - len(cohesins))])
            cur.append(positions)  # appending current positions to an array 
        cur = np.array(cur)  # when we finished a block of positions, save it to HDF5 
        dset[st:end] = cur
    myfile.attrs["N"] = N
    myfile.attrs["LEFNum"] = LEFNum
    # added this on 15-07-2021
    myfile.attrs["LIFETIME"] =args[ 'LIFETIME']
    myfile.attrs["LIFETIME_STALLED"] =args[ 'LIFETIME_STALLED']
    myfile.attrs["N1"] =args[ 'N1']
    myfile.attrs["M"] =args[ 'M']
    myfile.attrs["LS_one"] =args[ 'LS'][0]
    myfile.attrs["LS_two"] =args[ 'LS'][1]
    
# ###########################################################################################################
# ###########################################################################################################
# ###########################################################################################################
# ###########################################################################################################
# ###########################################################################################################
# ###########################################################################################################
# ###########################################################################################################
# ###########################################################################################################



os.makedirs(os.path.join(root, destination), exist_ok = True)
destination_folder = os.path.join(root, destination)


myfile = h5py.File(os.path.join(root, myfile_name), mode='r')


N = myfile.attrs["N"]
LEFNum = myfile.attrs["LEFNum"]
LEFpositions = myfile["positions"]

Nframes = LEFpositions.shape[0]
# Nframes = 50000
    
steps = 500   # MD steps per step of cohesin
stiff = 1
dens = 0.1
box = (N / dens) ** 0.33  # density = 0.1.




data = grow_cubic(N, int(box) - 2)  # creates a compact conformation 
block = 0  # starting block 
data.shape



# new parameters because some things changed 
saveEveryBlocks = 10   # save every 10 blocks (saving every block is now too much almost)
restartSimulationEveryBlocks = 100

# parameters for smc bonds
smcBondWiggleDist = 0.2
smcBondDist = 0.5
# assertions for easy managing code below 
assert (Nframes % restartSimulationEveryBlocks) == 0 
assert (restartSimulationEveryBlocks % saveEveryBlocks) == 0

savesPerSim = restartSimulationEveryBlocks // saveEveryBlocks
simInitsTotal  = (Nframes) // restartSimulationEveryBlocks
print('Total simulations {}, and {} saves per simulation.'.format(simInitsTotal, savesPerSim))


class bondUpdater(object):

    def __init__(self, LEFpositions):
        """
        :param smcTransObject: smc translocator object to work with
        """
        self.LEFpositions = LEFpositions
        self.curtime  = 0
        self.allBonds = []

    def setParams(self, activeParamDict, inactiveParamDict):
        """
        A method to set parameters for bonds.
        It is a separate method because you may want to have a Simulation object already existing

        :param activeParamDict: a dict (argument:value) of addBond arguments for active bonds
        :param inactiveParamDict:  a dict (argument:value) of addBond arguments for inactive bonds

        """
        self.activeParamDict = activeParamDict
        self.inactiveParamDict = inactiveParamDict


    def setup(self, bondForce,  blocks=100, smcStepsPerBlock=1):
        """
        A method that milks smcTranslocator object
        and creates a set of unique bonds, etc.

        :param bondForce: a bondforce object (new after simulation restart!)
        :param blocks: number of blocks to precalculate
        :param smcStepsPerBlock: number of smcTranslocator steps per block
        :return:
        """


        if len(self.allBonds) != 0:
            raise ValueError("Not all bonds were used; {0} sets left".format(len(self.allBonds)))

        self.bondForce = bondForce

        #precalculating all bonds
        allBonds = []
        
        loaded_positions  = self.LEFpositions[self.curtime : self.curtime+blocks]
        allBonds = [[(int(loaded_positions[i, j, 0]), int(loaded_positions[i, j, 1])) 
                        for j in range(loaded_positions.shape[1])] for i in range(blocks)]

        self.allBonds = allBonds
        self.uniqueBonds = list(set(sum(allBonds, [])))

        #adding forces and getting bond indices
        self.bondInds = []
        self.curBonds = allBonds.pop(0)

        for bond in self.uniqueBonds:
            paramset = self.activeParamDict if (bond in self.curBonds) else self.inactiveParamDict
            ind = bondForce.addBond(bond[0], bond[1], **paramset) # changed from addBond
            self.bondInds.append(ind)
        self.bondToInd = {i:j for i,j in zip(self.uniqueBonds, self.bondInds)}
        
        self.curtime += blocks 
        
        return self.curBonds,[]


    def step(self, context, verbose=False):
        """
        Update the bonds to the next step.
        It sets bonds for you automatically!
        :param context:  context
        :return: (current bonds, previous step bonds); just for reference
        """
        if len(self.allBonds) == 0:
            raise ValueError("No bonds left to run; you should restart simulation and run setup  again")

        pastBonds = self.curBonds
        self.curBonds = self.allBonds.pop(0)  # getting current bonds
        bondsRemove = [i for i in pastBonds if i not in self.curBonds]
        bondsAdd = [i for i in self.curBonds if i not in pastBonds]
        bondsStay = [i for i in pastBonds if i in self.curBonds]
        if verbose:
            print("{0} bonds stay, {1} new bonds, {2} bonds removed".format(len(bondsStay),
                                                                            len(bondsAdd), len(bondsRemove)))
        bondsToChange = bondsAdd + bondsRemove
        bondsIsAdd = [True] * len(bondsAdd) + [False] * len(bondsRemove)
        for bond, isAdd in zip(bondsToChange, bondsIsAdd):
            ind = self.bondToInd[bond]
            paramset = self.activeParamDict if isAdd else self.inactiveParamDict
            self.bondForce.setBondParameters(ind, bond[0], bond[1], **paramset)  # actually updating bonds
        self.bondForce.updateParametersInContext(context)  # now run this to update things in the context
        return self.curBonds, pastBonds
        
        
        
        
milker = bondUpdater(LEFpositions)

reporter = HDF5Reporter(folder=destination_folder,
                        max_data_length=100, overwrite=True, blocks_only=False)
                        
                        
                        
#reporter = HDF5Reporter(folder="trajectory", max_data_length=100, overwrite=True, blocks_only=False)

#down_files = ['LEFPositions.h5']
for iteration in range(simInitsTotal):
    
    # simulation parameters are defined below 
    a = Simulation(
            platform="cuda",
            integrator="variableLangevin", 
            error_tol=0.01, 
            GPU = "0", 
            collision_rate=0.03, 
            N = len(data),
            reporters=[reporter],
            PBCbox=[box, box, box],
            precision="mixed")  # timestep not necessary for variableLangevin

    ############################## New code ##############################
    a.set_data(data)  # loads a polymer, puts a center of mass at zero
    
    a.add_force(
        forcekits.polymer_chains(
            a,
            chains=[(0, None, False)],

                # By default the library assumes you have one polymer chain
                # If you want to make it a ring, or more than one chain, use self.setChains
                # self.setChains([(0,50,1),(50,None,0)]) will set a 50-monomer ring and a chain from monomer 50 to the end

            bond_force_func=forces.harmonic_bonds,
            bond_force_kwargs={
                'bondLength':1.0,
                'bondWiggleDistance':0.1, # Bond distance will fluctuate +- 0.05 on average
             },

            angle_force_func=forces.angle_force,
            angle_force_kwargs={
                'k':1.5
                # K is more or less arbitrary, k=4 corresponds to presistence length of 4,
                # k=1.5 is recommended to make polymer realistically flexible; k=8 is very stiff
            },

            nonbonded_force_func=forces.polynomial_repulsive,
            nonbonded_force_kwargs={
                'trunc':1.5, # this will let chains cross sometimes
                'radiusMult':1.05, # this is from old code
                #'trunc':10.0, # this will resolve chain crossings and will not let chain cross anymore
            },

            except_bonds=True,
             
        )
    )
    
    # ------------ initializing milker; adding bonds ---------
    # copied from addBond
    kbond = a.kbondScalingFactor / (smcBondWiggleDist ** 2)
    bondDist = smcBondDist * a.length_scale

    activeParams = {"length":bondDist,"k":kbond}
    inactiveParams = {"length":bondDist, "k":0}
    milker.setParams(activeParams, inactiveParams)
     
    # # this step actually puts all bonds in and sets first bonds to be what they should be
    milker.setup(bondForce=a.force_dict['harmonic_bonds'],
                blocks=restartSimulationEveryBlocks)

    # If your simulation does not start, consider using energy minimization below
    if iteration==0:
        a.local_energy_minimization() 
    else:
        a._apply_forces()
    
    for i in range(restartSimulationEveryBlocks):        
        if i % saveEveryBlocks == (saveEveryBlocks - 1):  
            a.do_block(steps=steps)
        else:
            a.integrator.step(steps)  # do steps without getting the positions from the GPU (faster)
        if i < restartSimulationEveryBlocks - 1: 
            curBonds, pastBonds = milker.step(a.context)  # this updates bonds. You can do something with bonds here
    data = a.get_data()  # save data and step, and delete the simulation
    del a
    
    reporter.blocks_only = True  # Write output hdf5-files only for blocks

    # time.sleep(10)
    # for filenam in os.listdir('trajectory'):
    #     if filenam not in set(down_files):
    #         files.download('trajectory/'+filenam)
    #         down_files.append(filenam)

      
 
    time.sleep(0.2)  # wait 200ms for sanity (to let garbage collector do its magic)
    
reporter.dump_data()


# ###########################################################################################################
# ###########################################################################################################
# ###########################################################################################################
# ###################        Plot       #####################################################################
# ###########################################################################################################
# ###########################################################################################################
# ###########################################################################################################
# ###########################################################################################################
from polychrom.hdf5_format import list_URIs, load_URI

from polychrom.contactmaps import monomerResolutionContactMap, monomerResolutionContactMapSubchains, binnedContactMap
from multiprocessing import Pool 
import glob
from polychrom import polymerutils
import numpy as np 
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt 
import os
import pickle


dirs = [destination]

hmaps = []

for i in dirs:
    URIs = list_URIs(os.path.join(root, i))
    starts = list(range(0, N, N1))
    hmap = monomerResolutionContactMapSubchains(filenames=URIs,  mapStarts=starts, mapN=N1, cutoff=6,
                        n=1, loadFunction=lambda x:load_URI(x)["pos"])
    hmaps.append(hmap)


def logbinsnew(a, b, ratio=0, N=0):
    a = int(a)
    b = int(b)
    a10, b10 = np.log10([a, b])
    if ratio != 0:
        if N != 0:
            raise ValueError("Please specify N or ratio")
        N = int(np.log(b / a) / np.log(ratio))
    elif N == 0:
        raise ValueError("Please specify N or ratio")
    data10 = np.logspace(a10, b10, N)
    data10 = np.array(np.rint(data10), dtype=int)
    data10 = np.sort(np.unique(data10))
    assert data10[0] == a
    assert data10[-1] == b
    return data10


_bins = logbinsnew(1, N1, 1.03)
_bins = [(0, 1)] + [(_bins[i], _bins[i+1]) for i in range(len(_bins) - 1)]
bins = np.array(_bins, dtype=np.int64, order="C")
M = bins.shape[0]


datas = []
for i in range(len(hmaps)):
    data = np.array(hmaps[i], dtype=np.double, order="C")
    N = data.shape[0]
    for k in range(M):
        start, end = bins[k, 0], bins[k, 1]
        ss = 0
        count = 0
        for offset in range(start, end):
            for j in range(0, N - offset):
                x = data[offset + j, j]
                if np.isnan(x):
                    continue
                ss += x
                count += 1

        meanss = ss / count
        if meanss != 0:
            for offset in range(start,end):
                for j in range(0,N-offset):
                    data[offset + j, j] /= meanss
                    if offset > 0: data[j, offset+j] /= meanss
    np.save(os.path.join(root, dirs[i], 'data.npy'), data)
    datas.append(data)

with open(os.path.join(root, 'normalized_data.pickle'), 'wb') as handle:
    pickle.dump(datas, handle)
    
with open(os.path.join(root, 'hmaps.pickle'), 'wb') as handle:
    pickle.dump(hmaps, handle)


im = plt.imshow(np.log(datas[0]+1e-3), cmap = 'coolwarm', vmin = -1, vmax = 1)
plt.colorbar(im ,fraction=0.046, pad = 0.04)
plt.title(my_plot_title)

# im = ax[0,1].imshow(np.log(datas[1]), cmap = 'coolwarm', vmin = -1, vmax = 1)
# plt.colorbar(im, ax = ax[0,1], fraction=0.046, pad = 0.04)
# ax[0,1].set_title('Left')

# im = ax[1,0].imshow(np.log(datas[2]), cmap = 'coolwarm', vmin = -1, vmax = 1)
# plt.colorbar(im, ax = ax[1,0], fraction=0.046, pad = 0.04)
# ax[1,0].set_title('Right')

# im = ax[1,1].imshow(np.log(datas[3]), cmap = 'coolwarm', vmin = -1, vmax = 1)
# plt.colorbar(im, ax = ax[1,1], fraction=0.046, pad = 0.04)
# ax[1,1].set_title('None')

plt.savefig( os.path.join(root ,my_plot_title+'.png'), bbox_inches='tight')
plt.savefig( os.path.join(root ,my_plot_title+'.eps'), bbox_inches='tight')
plt.show()


