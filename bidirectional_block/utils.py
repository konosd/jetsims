import numpy as np
import random


class leg(object):
    def __init__(self, pos, attrs={"stalled":False, "CTCF":False}):
        """
        A leg has two important attribues: pos (positions) and attrs (a custom list of attributes)
        """
        self.pos = pos
        self.attrs = dict(attrs)

class cohesin(object):
    """
    A cohesin class provides fast access to attributes and positions 
    
    
    cohesin.left is a left leg of cohesin, cohesin.right is a right leg
    cohesin[-1] is also a left leg and cohesin[1] is a right leg         
    
    Also, cohesin.any("myattr") is True if myattr==True in at least one leg
    cohesin.all("myattr") is if myattr=True in both legs
    """
    def __init__(self, leg1, leg2):
        self.left = leg1
        self.right = leg2
   
    def any(self, attr):
        return self.left.attrs[attr] or self.right.attrs[attr]
    
    def all(self, attr):
        return self.left.attrs[attr] and self.right.attrs[attr]    
    
    def __getitem__(self, item):
        if item == -1:
            return self.left
        elif item == 1:
            return self.right 
        else:
            raise ValueError()
        

def unloadProb(cohesin, args):
    """
    Defines unload probability based on a state of cohesin 
    """
    if cohesin.any("stalled"):
        # if one side is stalled, we have different unloading probability 
        # Note that here we define stalled cohesins as those stalled not at CTCFs 
        return 1 / args["LIFETIME_STALLED"]
    # otherwise we are just simply unloading 
    return 1 / args["LIFETIME"]    
    


def loadOne(cohesins, occupied, args): 
    """
    A function to load one cohesin 
    """
    if args['Probabilistic_Loading']:
        while True:
            a = random.choices(np.arange(0,len(occupied)), weights = args['Probabilistic_Profile'])[0]
            if (occupied[a] == 0) and (occupied[a+1] == 0):
                occupied[a] = 1
                occupied[a+1] = 1 
                cohesins.append(cohesin(leg(a), leg(a+1)))
                break
    elif args['Site_Loading']:
        while True:
            a = np.random.randint(args['M'])*args['N1']+np.random.randint(args['LS'][0], args['LS'][1])
            if (occupied[a] == 0) and (occupied[a+1] == 0):
                occupied[a] = 1
                occupied[a+1] = 1 
                cohesins.append(cohesin(leg(a), leg(a+1)))
                break
            else:
                break
    else:
        while True:
            a = np.random.randint(args["M"])*args['N1']+args["N1"]//2
            if (occupied[a] == 0) and (occupied[a+1] == 0):
                occupied[a] = 1
                occupied[a+1] = 1 
                cohesins.append(cohesin(leg(a), leg(a+1)))
                break
            else:
                break




def capture(cohesin, occupied, args):
    """
    We are describing CTCF capture here. 
    This function is specific to this particular project, and 
    users are encouraged to write functions like this 
    
    Note the for-loop over left/right sites below, and using cohesin[side] 
    to get left/right leg. 
    
    Also note how I made ctcfCapture a dict with -1 coding for left side, and 1 for right side 
    and ctcfCapture are dicts as well: keys are locations, and values are probabilities of capture
    """    
    for side in [1, -1]:
        # get probability of capture or otherwise it is 0 
        if np.random.random() < args["ctcfCapture"][side].get(cohesin[side].pos, 0):  
            cohesin[side].attrs["CTCF"] = True  # captured a cohesin at CTCF  
            cohesin[-1* side].attrs["CTCF"] = True  # captured a cohesin at CTCF  DOUBLE SIDED
    return cohesin 


def release(cohesin, occupied, args):
    
    """
    AN opposite to capture - releasing cohesins from CTCF 
    """
    
    if not cohesin.any("CTCF"):
        return cohesin  # no CTCF: no release necessary 
        
    # attempting to release either side 
    for side in [-1, 1]: 
        if (np.random.random() < args["ctcfRelease"][side].get(cohesin[side].pos, 0)) and (cohesin[side].attrs["CTCF"]):
            cohesin[side].attrs["CTCF"] = False 
    return cohesin 


def translocate(cohesins, occupied, args):
    """
    This function describes everything that happens with cohesins - 
    loading/unloading them and stalling against each other 
    
    It relies on the functions defined above: unload probability, capture/release. 
    """
    # first we try to unload cohesins and free the matching occupied sites 
    # print(f'{len(cohesins)} in')
    for i in range(len(cohesins)):
        if i>=len(cohesins):
            # print(f'{len(cohesins)} in, {i}')
            break
        
        prob = unloadProb(cohesins[i], args)
        if np.random.random() < prob:
            occupied[cohesins[i].left.pos] = 0 
            occupied[cohesins[i].right.pos] = 0 
            del cohesins[i]
            loadOne(cohesins, occupied, args)
            
    
    # then we try to capture and release them by CTCF sites 
    for i in range(len(cohesins)):
        cohesins[i] = capture(cohesins[i], occupied, args)
        cohesins[i] = release(cohesins[i], occupied, args)
    
    # finally we translocate, and mark stalled cohesins because 
    # the unloadProb needs this 
    for i in range(len(cohesins)):
        cohesin = cohesins[i] 
        for leg in [-1,1]: 
            if not cohesin[leg].attrs["CTCF"]:
                # This is the extra line to make stochastic drift and not just moving outwards.
                new_position = np.where(np.random.uniform()>args['drift'], -leg, leg) 
                # cohesins that are not at CTCFs and cannot move are labeled as stalled 
                if occupied[cohesin[leg].pos  + new_position] != 0:
                    cohesin[leg].attrs["stalled"] = True
                else:
                    cohesin[leg].attrs["stalled"] = False 
                    occupied[cohesin[leg].pos] = 0
                    occupied[cohesin[leg].pos + new_position] = 1
                    cohesin[leg].pos += new_position        
        cohesins[i] = cohesin
        
def color(cohesins, args):
    "A helper function that converts a list of cohesins to an array colored by cohesin state"    
    def state(attrs):
        if attrs["stalled"]:
            return 2
        if attrs["CTCF"]:
            return 3
        return 1
    ar = np.zeros(args["N"])
    for i in cohesins:
        ar[i.left.pos] = state(i.left.attrs)
        ar[i.right.pos] = state(i.right.attrs)  
    return ar

def generate_probability_profile(profile, N1, M):
    R = int(len(profile)/N1)
    profile = profile.reshape(-1,R).mean(axis = 1)
    profile = profile/np.sum(profile)
    profile = np.tile(profile, M)/M
    profile = profile + 0.1*np.min(profile[profile>0])
    return profile


def loadOne_probabilistic(cohesins, occupied, profile, args): 
    """
    A function to load one cohesin 
    """
    while True:
        a = random.choices(np.arange(0,len(occupied)), weights = profile)[0]
        if (occupied[a] == 0) and (occupied[a+1] == 0):
            occupied[a] = 1
            occupied[a+1] = 1 
            cohesins.append(cohesin(leg(a), leg(a+1)))
            break
