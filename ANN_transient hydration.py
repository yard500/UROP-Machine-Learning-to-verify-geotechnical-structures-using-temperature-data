# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 10:59:12 2021

@author: Wanru Yang
"""
#
# 2021-10-05 | DMG Taborda | Added an error catch for Qratio, added verification graphs after ANN, added output writing to screen, added plots of data
#_______________________________________________________________________________
#
#---IMPORT LIBRARIES------------------------------------------------------------
#_______________________________________________________________________________
#
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
import pandas as pd
from alive_progress import alive_bar
#
#%%  DATA GENERATION
#_____________________________________________________________________________
#
#---FUNCTION DEFINITIONS--------------------------------------------------------
#_______________________________________________________________________________
#
#
# whichBCsSet() checks which constant-temperature boundary conditions have been
# set, and stores this info in BCset as 1 (fixed-temp. BC set) or 0 (not set/
# no heat flux boundary), with the 0th index corresponding to the leftmost
# (central) boundary and the 1st to the rightmost/outermost boundary:
def whichBCsSet():
    global BCset
    BCset = np.array([BC != False for BC in BCs]).astype(int)
# 
#
# initVec() creates zeros vectors, with length defaulting for global, useful for
# initialising vectors for assignment.
#n_node = None # Place-holder allowing function to be defined.
def initVec(length = None):
    if length == None:
        length = n_node
    return np.zeros([length])
#
#
# initMat() creates matrices vectors, with both dimensions defaulting to global,
# useful for initialising matricess for assignment.
def initMat(rows = None, cols = None):
    if rows == None:
        rows = n_node
    if cols == None:
        cols = n_node
    return np.zeros([rows, cols])
# As n_node is used as a default parameter in initVec and initMat, it must now
# be given a dummy value to allow these functions to be defined:
n_node = None

#
#
# processMesh() processes the user-inputted mesh and executes the definition of
# a set of useful arrays and variables characterising the mesh, allowing for
# manipulation thereof.
def processMesh():
    global Rs
    global noSubRegs
    global noConcSRs
    global subReg_range
    global subReg_k
    global subReg_rho_c
    global noEsInSRs
    global rs
    global n_node
    global n_concrete
    global firstNodeInSR
    
    
    # Rs is an array listing boundaries of all mesh sub-regions, increasing-r.
    Rs = np.concatenate(([0], np.cumsum(subReg_range), [r_domain]))
    
    # noSubRegs is the number of sub-regions.
    noSubRegs = len(Rs) - 1
    
    # noConcSRs is the number of concrete sub-regions.
    noConcSRs = sum([int(R <= rad) for R in Rs[1 :]])
    
    # subReg_range is the lengths of all subregions, increasing-r.
    subReg_range = np.diff(Rs)
   
    
    # subReg_k contains the k value of each sub-region (depends on the material)
    subReg_k = np.array([k_s[int(R > rad)] for R in Rs[1 :]])
    
    # subReg_rho_c contains the rho * c value for each sub-region
    subReg_rho_c = np.array([rho_c_s[int(R > rad)] for R in Rs[1 :]])
    
    # noEsInSRs gives the number of elements in each sub-region. 
    # Elementwise matrix multiplication
    # subReg_1 is the length of subregion element 
    noEsInSRs = np.multiply(subReg_range,
                            (1 / subReg_l).astype(int)).astype(int)
    # print(np.shape(subReg_range))
    # print(np.shape(subReg_l))
    # rs is the array containing the radial abscissa of each node; here we will
    # assemble it by looping through each sub-region.
    rs = np.array([0])
    for i in range(0,noSubRegs):
        rs_new = np.linspace(Rs[i], Rs[i + 1], num = noEsInSRs[i],
                             endpoint = False)
        rs = np.concatenate((rs, rs_new + subReg_l[i]))
        
    n_node = len(rs)
    # n_concrete gives the number of nodes which belong to one or more concrete
    # elements.
    n_concrete = sum(noEsInSRs[0:noConcSRs]) + 1
    # firstNodeInSR gives the global index of the first node within each sub-reg
    # along with domain-final node:
    firstNodeInSR = np.concatenate(([0], np.cumsum(noEsInSRs)[0:noSubRegs - 1],
                                    [n_node - 1]))
    # print(firstNodeInSR)

    
#
# processPlotPts() initialises the arrays used to record temps. at the user-
# defined positions over time and at the user-defined times across the space
# domain, and generates the relevant indeces to extract these temps. from
# the T, the global temperature vector at a given time.
def processPlotPts():
    global Tvr
    global Tvt
    global metres_rs
    global closeness
    global ts
    
    if np.size(metres)==1:
        closeness=np.isclose(rs, metres)
        metres_rs= np.where(closeness)[0]
    else:
    # Initialise array to store T vs. r at given times -space:
        Tvr = initMat(rows = len(days))
        # Initialise array to store T vs. t at given positions -time:
        Tvt = initMat(rows = len(metres), cols = len(ts))
        
        # Find indeces of rs where desired positions are:
        metres_rs = initVec(len(metres))
        
        for i in range(0, len(metres)):
            metres_rs[i] = np.where(np.isclose(rs, metres[i]))[0][0]
        metres_rs = metres_rs.astype(int)
#
#
#
# buildK() assembles K, C, and PI by looping through each subregion:
def buildKandCandPI():
    global K
    global C
    global PI
    # We'll build each matrix by considering each sub-region, but all at once.
    #
    # Initial consideration of K:
    #
    # Formula for K(element) is:
    #                          [1  -1]
    # K_e = pi*k*(r_i+r_j)/L_e*[-1  1]
    # Consider sK (for "scalar"), equal to pi*k/L_e for each sub-region:
    sK = np.pi * subReg_k / subReg_l
    
    # We can assemble the sparse matrix using just the diagonals. We will batch
    # compute the general elements within a single sub-region and individually
    # compute elements corresponding at nodes at sub-region boundaries:
    mainDiagK = initVec() # mainDiagK to contain the values to be inserted into
                          # the main/leading diagonal of K
    offDiagK = initVec(n_node - 1) # offDiagK to contain the values to be
                                   # inserted into the -1 and +1 diagonals of K
    #
    # Initial consideration of C:
    #
    # Formula for C(element) is:
    #                    [3r_i + r_j   r_i + r_j]
    # C_e = pi*rho*c*L/6*[r_i + r_j   r_i + 3r_j]
    # sC is pi*rho*c*L/6
    sC = np.pi * subReg_rho_c * subReg_l / 6
    mainDiagC = initVec()
    offDiagC = initVec(n_node - 1)
    #
    # Initial consideration of PI:
    # Uniquely, PI, which is the matrix which multiplies with the P / heating
    # power vector to give the integrated power due to hydration over an
    # element, will have a smaller size as it will only be populated within the
    # concrete sub-regions.
    # Formula for PI(element) is:
    #               [2r_i + r_j       0     ]
    # PI_e = pi*L/3*[    0        r_i + 2r_j]
    # Again normalising the matrix by dividing its elements by L (elem. length)
    # and taking this L outisde gives sC ("scalar") for C:
    # sPI is pi*L/3
    sPI = np.pi * subReg_l / 3
    mainDiagPI = initVec(n_concrete) # Note that PI has only its leading
    #                                  diagonal, because only having diagonal value.
    # Initialise main diag
    for SR in range(0, noSubRegs): # assembling over subregions 
        L = subReg_l[SR] # Lengths of elements in sub-region
        
        # SRNode0 and SRNodeN hold the global index of the first and last node
        # of each sub-region:
        SRNode0, SRNodeN = firstNodeInSR[SR : SR + 2]
        
        ## Let's do K then C:
        # Update mainDiagK based on this sub-region:
        # First element of main-diagonal for this sub-region:
        mainDiagK[SRNode0] = mainDiagK[SRNode0] + sK[SR] * (2 * rs[SRNode0] + L)
        # Elems. of main-diagonal for this sub-region corrpg. to interior node:
        mainDiagK[SRNode0 + 1 :
              SRNodeN] = 4 * sK[SR] * np.arange(rs[SRNode0 + 1], rs[SRNodeN], L)
        # Last element of main-diagonal for this sub-region:
        mainDiagK[SRNodeN] = sK[SR] * (2 * rs[SRNodeN] - L)
        
        # Off-diagonals for this sub-region:
        # Generate series 1, 3, 5, 7, ...
        odds = 2 * np.arange(1, noEsInSRs[SR] + 1) - 1
        offDiagK[SRNode0 : SRNodeN] = -sK[SR] * (2 * rs[SRNode0] +  L * odds)
        
        
        # Let's do C:
        # Update mainDiagC based on this sub-region:
        # First element of main-diagonal for this sub-region:
        mainDiagC[SRNode0] = mainDiagC[SRNode0] + sC[SR] * (3 * rs[SRNode0]
                                                            + rs[SRNode0 + 1])
        # Elems. of main-diagonal for this sub-region corrpg. to interior node:
        mainDiagC[SRNode0 + 1 : SRNodeN] = 8 * sC[SR] * \
                                           (rs[SRNode0] + L *
                                            range(1, noEsInSRs[SR]))
        # Last element of main-diagonal for this sub-region:
        mainDiagC[SRNodeN] = sC[SR] * (rs[SRNodeN - 1] + 3 * rs[SRNodeN])
        # Off-diagonals for this sub-region:
        offDiagC[SRNode0 : SRNodeN] = sC[SR] * \
                                      (2 * rs[SRNode0] + L *
                                       (2 *np.arange(1, noEsInSRs[SR] + 1) - 1))
        # Let's do PI:
    for SR in range(0, noConcSRs):
        L = subReg_l[SR] # Length of elements in this sub-region
        # SRNode0 and SRNodeN hold the global index of the first and last node
        # of each sub-region:
        SRNode0, SRNodeN = firstNodeInSR[SR : SR + 2]
        # Update mainDiagPI based on this sub-region:
        # First element of main-diagonal for this sub-region:
        mainDiagPI[SRNode0] = mainDiagPI[SRNode0] + sPI[SR] * (2 * rs[SRNode0]
                                                            + rs[SRNode0 + 1])
        # Elems. of main-diagonal for this sub-region corrpg. to interior node:
        mainDiagPI[SRNode0 + 1 : SRNodeN] = 6 * sPI[SR] * \
                                           (rs[SRNode0] + L *
                                            range(1, noEsInSRs[SR]))
        # Last element of main-diagonal for this sub-region:
        mainDiagPI[SRNodeN] = sPI[SR] * (rs[SRNodeN - 1] + 2 * rs[SRNodeN])
    K = diags([offDiagK, mainDiagK, offDiagK], [-1, 0, 1]).toarray()
    C = diags([offDiagC, mainDiagC, offDiagC], [-1, 0, 1]).toarray()
    PI = diags(mainDiagPI).toarray()
#
#
# buildLHSandRHS builds the following matrices:
def buildMatLHSandMatRHS():
    global matLHS
    global matRHS
    # With reference to Equation (6.38) in "Fundamentals of the Finite...":
    matLHS = C + theta * dt * K
    matRHS = C - (1 - theta) * dt * K
#
#
# getDtFNTheta() generates f(t=n+1), the heat input vector for the time step
# following the current one, and based on this, generates dt*{f}^(n+theta),
# where dt is the time step, for the current time step.
def getDtFNTheta():
    # 𝑄𝑎𝑐𝑐 is the accumulated heat of hydration per unit mass of cement under a
    # unique temperature-time history
    global Qacc 
    global fn
    global dtFNTheta
    #global debug
    global fn1
    tCut = t if t >= tn else tn # Cut-off time condition check for finding n*
    # eq. 21 p. 38 in Sajadi's report (2020):
    ttb_ns = (tau / tCut)**beta
    ns = (np.exp(ttb_ns) - 1) * ((1 + 1 / beta) / ttb_ns - 1) # n*
    if t == 0: # Prevents /0, leaves Pref & Qref full of 0s for time = 0
        Pref = initVec(n_concrete) # Temporarily stores Pref at all nodes
        Qref = initVec(n_concrete) # Temporarily stores Qref at all nodes
        Qacc = initVec(n_concrete) # Stores Qacc at all nodes, updates each time
                                   # step.
        fn = initVec(n_concrete) # Stores f(t=n) at all nodes, updates each time
                                 # step.
    else:
        ttb = (tau / t)**beta
        # eq. 11 p. 36 - Power [J/(s m^3)]:
        Pref = khi * rho_con * Qmax * beta * ttb * np.exp(- ttb) / t
        # eq. 10 p. 35 - Reference Q [J/kg]:
        Qref = Qmax * np.exp(- ttb)
        # eq. 19 p. 37:
    Tdiff = 1 / T[0 : n_concrete] - 1 / Tref
    Qratio = (Qmax - Qacc) / (Qmax - Qref)
    #
    # Add an error catch
    for i in range(0,len(Qratio)):
        if Qratio[i] < 0.0:
            Qratio[i] = 0.0
    #
    # P-power of hydration in eq 19 p.37
    P = Pref * np.exp(- Ea / R * Tdiff) * Qratio**ns
    #print('Qacc =', Qacc)
    #print('Tdiff =', Tdiff)
    #print('Pref =', Pref)
    #print('P = ', P)
    #print('Qr =', Qratio)
    #print(ns)
    #print('T =', T)
    Qacc = Qacc + 1 / (khi * rho_con) * P * dt
    
    # Incorprate into FEA - Assemble f(t=n+1):
    fn1 = np.matmul(PI, P)
    
    # Nasr report eq5
    dtFNTheta = dt * (theta * fn1 + (1 - theta) * fn)
    dtFNTheta = np.concatenate((dtFNTheta, initVec(n_node - n_concrete)))
    fn = fn1
    #DEBUG
    #if debug == 1:
    #    quit()
    #debug = 1
#
#
# Tn1() solves for T(t=n+1) given T(t=n) and dt*{f}^(n+theta).
def Tn1():
    global T
    # Prescribe constant temp. boundaries:
    T[[0, -1]] = [T[0] + BCset[0] * (BCs[0] - T[0]),
                  T[-1] + BCset[1] * (BCs[1] - T[-1])]
    
    #
    # Algebraic manipulation:
    # Note: throughout this program, I use BCset[0] and BCset[1] to regard or
    # disregard rows and columns of matrices corresponding to boundary nodes
    # where a fixed-temp. boundary has been set.
    # Recall that BCset has two indeces corresponding to the inner- and outer-
    # boundaries, and that the value of each index will be 1 if a constant-temp.
    # condition is set and 0 otherwise.
    # For example, if you see an index written as
    # T[BCset[0] : n_node - BCset[1]]
    # then if no fixed-temp. BC is set at the centre, BCset[0] will be 0 and
    # therefore the returned sub-array will start at the 0th index within T,
    # whereas if this BC has been set with a fixed temp., BCset[0] will be 1
    # and therefore the 0th index will be missed out, and similar for the other
    # boundary.
    #
    # With reference to Equation (6.38) in "Fundamentals of the Finite...",
    # and also to slides 12-15 of the powerpoint:
    #
    # This multiplies the matrix matRHS, which is equal to
    # C - (1 - theta) * dt * K
    # by the T vector, which has all temps. at time t = n :
    RHSTn = np.matmul(matRHS, T)
    #
    # This calculates the entire RHS of Equation (6.38) by adding
    # ([C]) - (1 - theta) * dt * [K]) * {T}(t=n) to
    # dt * (theta * {f}(t=n+1) + (1 - theta) * {f}(t-n))
    # which results in a vector: 
    fullRHS = RHSTn + dtFNTheta
    #
    # This removes from the full RHS vector calculated above, the rows
    # corresponding to the boundary nodes where a constant temperature has been
    # set, striking the rows - see slide 14:
    fullRHSstruck = fullRHS[BCset[0] : n_node - BCset[1]]
    
    #
    # This removes the same rows from the vector matLHS which is equal to
    # C - (1 - theta) * dt * K
    # as can be seen in Equation (6.38), however, it should be noted that matLHS
    # is not the entire LHS of the equation as is the case with fullRHS, rather
    # matLHS multiplies with {T}(t=n+1) to give the entire LHS. See slide 14:
    LHSstruck = matLHS[BCset[0] : n_node - BCset[1], :]
    #
    # newRHS is intially a copy of the vector of the full RHS of the equation
    # with constant-temp, the same as fullRHSstruck, and then we subtract from
    # it, element-wise, the product of the columns of LHSstruck corresponding
    # to fixed-temp. BCs with the fixed temps. at those boundaries, as is done
    # in the lower half of slide 14:
    newRHS = np.copy(fullRHSstruck)
    newRHS[[0, - 1]] = np.array( \
                [newRHS[0] - BCset[0]*sum(LHSstruck[:, 0] * BCs[0]),
                 newRHS[- 1] - BCset[1]*sum(LHSstruck[:, - 1] * BCs[1])])
    #
    # Using newRHS rather than fullRHSstruck in the same equation is equivalent
    # if we also remove the columns of LHSstruck that correspond to the fixed-
    # temp. BCs (see slide 14):
    LHSstruckSqr = LHSstruck[:, BCset[0] : n_node - BCset[1]]
    #
    # We now habe our equation in the form:
    # [LHSstruckSqr] * {T}(t=n+1) = [newRHS]
    #    ^matrix        ^vector        ^matrix
    # and therefore we can solve this equation using matrix division to give
    # {T}(t=n+1), which we save over T - although we do not touch the elements
    # in T corresponding to fixed-temp. BCs, as these do not change:
    T[BCset[0] : n_node - BCset[1]] = \
        np.matmul(np.linalg.inv(LHSstruckSqr), newRHS)
#
#
# startRecordingTemps() is invoked immediately before recVals() to initialise
# the arrays to store temperatures at chosen positions and times, and to reset
# counters used to save this data to the correct rows and column of these
# arrays.
def startRecordingTemps():
    global TatDay
    global whichDay
    global TatPos
    global oneHr
    global ts
    
    
    if np.size(metres)==1:
        # TatPos stores temps. across time domain at user-specified positions:
        TatPos = initMat(rows = np.size(metres),cols =len(ts))
        oneHr = 0
        

    else:
        TatPos = initMat(rows = len(metres),cols = int(t_domain / (60**2) + 1))
        oneHr = 0    # Counts how many 1 hour have passed, allowing us to
                     # record temperatures at user-specified positions every 1
                     # hours and save this in the correct volumn of the TatPos
                     # array. To be used in recVals.
                     
        # # TatDay stores temps. across r domain at user-specified numbers of days:
        # TatDay = initMat(rows = len(days))
        # whichDay = 0 # Counts how many of the user-specified times we have already
        #              # recorded temperatures at representative positions for,
        #              # allowing data to be stored in the correct row of the TatDay
        #              # array. To be used in recVals.
        # # TatPos stores temps. across time domain at user-specified positions:

                     
    #
#
# recTemps() records temperature values at representative positions across the
# spatial domain at user-specified times, and records temperature values at rep-
# resentative times across the temporal domain at specified positions.
def recTemps():
    global TatDay
    global whichDay
    global TatPos
    global oneHr
    global wait1
    global wait2
    # # Record T at selected times all across space domain:   
    # if sum([t / (24 * 60**2) == day for day in days]) == 1:
    #     TatDay[whichDay, :] = T
    #     whichDay += 1
    #     wait1 = 0
    # Record T at selected r values in 1 hour intervals:
    if t % (1 * 60**2) == 0:
        TatPos[:, oneHr] = T[metres_rs]
        oneHr += 1
        wait2 = 0
#
#
# loopSolve() invokes the various commands to progress through the time domain
# to simulate the development of temperatures over time.
def loopSolve():
    global dtFNTheta # remove this when hydration is considered
    global T
    global t
    global ts
    # T is a 2D array of temp. values at all nodes, updated each step:
    T = initVec() + Tinit
    for n in range(0, len(ts) - 1, 1):
        # t stores the current timestep:
        t = ts[n]
        recTemps()
        getDtFNTheta()
        Tn1()
    # Collect temps. at the final timestep:
    t = ts[n + 1]
    recTemps()

# loop is used to loop in the different inputs arrays and calling above functions to generate 
# temperature and stored in a dataset for further training
data_generated=pd.DataFrame(columns = ['pile radius','k_soil','k_concrete','Qmax','tau','beta','time','temperature']) 

def loop():
    global n
    global rad
    global r_domain
    global subReg_range 
    global k_s
    global metres
    global Qmax
    global tau
    global beta
    global data_generated
    #
    # Set matplotlib to interactive mode
    plt.ion()
    #
    print('**** Generating random data for ANN training')
    print('     Requested cases: '+str(n))
    #
    with alive_bar(n, bar = 'classic', spinner = 'stars') as bar:
        for i in range(0,n,1):
            #print('+-+-+-+-+-+-+ Generating '+str(i+1)+' of '+str(n))
            r_domain=r_domain0[i]
            rad=rad0[i]
            k_s=k_s0[:,i]
            subReg_range=[subReg_range0[0][i],subReg_range0[1][i]]
            metres= metres0[i]
            Qmax=Qmax0[i]
            tau=tau0[i]
            beta=beta0[i]
            
            
            processMesh()
            processPlotPts()
            buildKandCandPI()
            buildMatLHSandMatRHS()
            startRecordingTemps()
            loopSolve()

            data_generated=data_generated.append(pd.DataFrame({'pile radius':[rad],'k_soil':[k_s[0]],'k_concrete':[k_s[1]],\
                                            'Qmax':[Qmax],'tau':[tau],'beta':[beta],\
                                                'time':[ts],'temperature':[TatPos]}))
            data_generated=data_generated.reset_index(drop=True)
            
            # plt.scatter(ts,TatPos)
            #
            bar()

        
#_______________________________________________________________________________
#
#---PROBLEM INPUT---------------------------------------------------------------
#_______________________________________________________________________________
#
# GENERAL PROBLEM PARAMETERS:
n           = 1000                                   # size of inputs
k_con       = np.random.randint(100,400,n)/100       # [W/(m K)]
rho_con     = 2400                                   # [kg/m^3]
c           = 895                                    # [J/(kg K)]
rho_c_con   = rho_con * c                            # [J/(m^3 K)]

k_soil      = np.random.randint(100,400,n)/100       # [W/(m K)]
rho_c_soil  = 1820000                                # [J/(m^3 K)]
k_s0        = np.array([k_con, k_soil])              # [W/(m K)]
rho_c_s     = np.array([rho_c_con, rho_c_soil])      # [J/(m^3 K)]
area        = 1                         # [m^2]
rad0        = np.random.randint(3,12,n)/10           # [m] pile radius
d           = 2 * rad0                               # [m] pile diameter
r_domain0   = rad0 * 20                              # [m]
Tinit       = 280                                    # [K] intial temp. everywhere

# TIME PARAMETERS
dt          =1*60**2                                 # [s] time step for FEA
t_domain    = 50*24*60**2                            # [s] simulation duration: 50 days
# Different time intervals for temperature recording ：
t1=np.arange(0,5*24+1,1)*60**2                         # [s] first 5 days, recording temperature at 1h time step
t2=np.arange(5*24+12,25*24+1,12)*60**2                 # [s] next 20 days, recording temperature at 12h time step
t3=np.arange(25*24+24,50*24+1,24)*60**2                # [s] last 25 days, recording temperature at 24h time step
ts=np.concatenate((t1,t2,t3),axis=0)
theta       = 0.5                       # [-]

# HYDRATION PARAMETERS:
tn           = 2.0*24*60**2              # [s]
Qmax0        = np.random.randint(100,600,n)*1e3       # [J/kg cement]
tau0         = np.random.randint(1e4,1e5,n)                    # [s]
beta0        = np.random.randint(25,150,n)/100                      # [-]
khi         = 0.17                      # [kg cement / kg concrete]
Ea          = 45000                     # [J/mol]
R           = 8.314                     # [J/(mol K)]
Tref        = 296.15                    # [K]
#

# Input fixed temperature BC's: (value of fixed temp. or False to indicate a no-
# heat-flux boundary)
# boundary:    r = o  r = rmax
BCs = np.array([False,   Tinit])

# Configure mesh by filling in following array, wherein the x domain is split
# into non-overlapping subregions defined by their lengths (final subregion
# extends to extent of problem's x domain), each split into the desired no. of
# elements per meter for each region. (please use all integer values thx)
subReg_range0 = np.array([rad0, d * 2]) # length of all but last subregions [m]
subReg_l = np.array([1/100, 1/20, 1/2]) # element length in each subregion [m]

# [NOT USED for GENERATING DATASET]
# Lastly please choose which time values at which to plot T vs. x:
days = [1, 5, 20, 50]

# [USED for GENERATING DATASET BECAUSE TEMPERATURE AT FIXED POSITION IS NEEDED]
# And which positions at which to plot T vs. t:
metres0 = rad0-7/1e2


#__________________________________________________________________________
#
#---CALLING OF PROGRAMS TO SIMULATE HEAT TRANSFER-------------------------------
#_______________________________________________________________________________
#
whichBCsSet()
loop()



# For convenience, ANN code is in the same file with data generation code
# The generated data can also be stored as csv for further machine learning using the following line:
# data_generated.to_csv(r'C:\Users\CICIW\Desktop\data_generated.csv', sep=',')




#%% MACHINE LEARNING
#_______________________________________________________________________________
#
#---ARTIFICAL NEUTRAL NETWORK （ANN) MACHINE LEARNING ------------------------------
#_______________________________________________________________________________
#

# This ANN directly adopts Multi-layer Perceptron regressor 
# which is inbuilt in sklearn package
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
#
print('**** Initiating ANN training')
#
# data_generated = pd.read_csv('data_generated_1000.csv')
#
# Establishing an ANN model consists of three steps: 
# (i) preparing the required data for training the network; 
# (ii) evaluating neural networks with different structures and choosing the optimal one;
# (iii) testing the neural network performance using data which have not been used previously for training the network.
#
# Inputs are: pile radius,k_soil,k_concrete, Qmax, tau, beta,time after casting
# Output which will be predicted is temperature at position of pile radius-7cm
#
# Length of time/temperature for each variables wrt corresponding time
n_time=len(data_generated.loc[:,'time'][0])
# Length of data_generated 
n_data=len(data_generated)
# Number of Inputs 
n_input=6
#
# Intialising data_generated which is rearranged into an array 
x_input=np.zeros((n_data,n_time+n_input))
# Initialising temperature array - label
y_label=np.zeros((n_data,n_time))
#
print('     Organising data')
col=list(data_generated.columns)
for i in range(n_data):
    inputs=np.ones((1,0))
    for j in range(len(col)-2):
        inputs=np.concatenate((inputs,np.array(data_generated.loc[i,col[j]]).reshape(1,1)),axis=1)
    inputs=np.concatenate((inputs,data_generated.loc[i,col[6]].reshape(1,n_time)),axis=1) 
    x_input[i]=inputs
    y_label[i]=data_generated.loc[i,col[7]]
    
y_label[np.isnan(y_label)]=np.nanmean(y_label)
#
print('     Normalising data')
# Normalising data
x_MinMax = preprocessing.MinMaxScaler()
x_input  = x_MinMax.fit_transform(x_input)

# scaler = preprocessing.StandardScaler().fit(x_input)
# x_input = scaler.transform(x_input)
# scaler = preprocessing.StandardScaler().fit(y_label)
# y_label = scaler.transform(y_label)

#_______________________________________________________________________________
#
#---ANN INPUT---------------------------------------------------------------
#_______________________________________________________________________________

## Hyperparameters for the MLP Regressor 
hidden_layer=(40,10)     # 2 hidden layer, 30 neurons in the 1st hidden layer and 20 neurons in the 2nd
epochs=int(1e4) # Number of passes of the entire dataset to the network during training.
activation_function='relu' # Activation fucntion, choose the most common 'rectified linear unit'
#
#
#
print('     Splitting data')
# Split data_generated into training/validation datasets
# Training data/Validation data =0.8:0.2
x_train, x_validation, y_train, y_validation = train_test_split(x_input, y_label,test_size=0.2)
#
print('     Training ANN')
#
clf = MLPRegressor(hidden_layer_sizes=hidden_layer,max_iter=epochs,alpha=1e-4,\
                   solver='adam', activation=activation_function,learning_rate='adaptive')
#
#
clf.fit(x_train, y_train)
#
print('     Calculating error')
#
mse=[]
y_validation_pred=[]
for i in range(len(x_validation)-1):
    
    y_validation_pred.append(clf.predict([x_validation[i]]))
    mse.append(np.mean((y_validation[i] - y_validation_pred[i])**2))
#    
#
n=np.mean(y_validation,axis=0)
m=np.mean(y_validation_pred,axis=0).reshape(n_time)
#
#
plt.plot(n,'o',label='Validation Set')
plt.plot(m,label='Predicted Validation Set')
plt.ylabel('Temperature  [K]')
#
plt.show()
#
# Verify training data
y_train_pred = clf.predict(x_train)
plt.scatter(y_train,y_train_pred)
plt.show()
#
# Verify validation data
y_validation_pred = clf.predict(x_validation)
plt.scatter(y_validation,y_validation_pred)
plt.show()



