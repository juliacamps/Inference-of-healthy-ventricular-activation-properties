# Import external modules
import seaborn as sns
sns.set(style="darkgrid", palette="colorblind")#, font_scale = 2)

import numba
from sklearn.utils import check_array

import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

import random
import pymp
import time
import math
import os
import matplotlib.pyplot as plt
from pyDOE import lhs

from scipy import signal
from scipy.stats import norm


# ------------------------------------------- MAIN SMC-ABC FUNCTIONS ----------------------------------------------------
def run_inference(meshName_val, meshVolume_val, final_path, tmp_path, target_type, metric, threadsNum_val, npart,
                    keep_fraction, rootNodeResolution, conduction_speeds, target_snr_db, load_target,
                    target_fileName):

    global meshName
    meshName = meshName_val
    global meshVolume
    meshVolume = meshVolume_val
    global threadsNum
    threadsNum = threadsNum_val
    global nan_value
    nan_value = np.array([np.nan]).astype(np.int32)[0]
    # Eikonal configuration
    global leadNames
    leadNames = [ 'I' , 'II' , 'V1' , 'V2' , 'V3' , 'V4' , 'V5' , 'V6']#, 'lead_prog' ]
    global nb_leads
    global nb_bsp
    frequency = 1000 # Hz
    freq_cut= 150 # Cut-off frequency of the filter Hz
    w = freq_cut / (frequency / 2) # Normalize the frequency
    global b_filtfilt
    global a_filtfilt
    b_filtfilt, a_filtfilt = signal.butter(4, w, 'low')
    endoRange = np.array([0.1, 0.2]) # value range
    gtRange = np.array([0.028, 0.06]) # gl and gn will be set as an initial +10% -10% of gt
    global gf_factor

    gf_factor = 1.6
    global gn_factor
    gn_factor = 1.
    # SMC-ABC configuration
    desired_Discrepancy = 0.01
    max_MCMC_steps = 100
    global nlhsParam
    nlhsParam = 4
    retain_ratio = 0.5
    nRootNodes_range = [6, 10]
    nRootNodes_centre = 8 #
    nRootNodes_std = 1
    global p_cdf
    p_cdf = np.empty((nRootNodes_range[1]), dtype='float64')
    for i in range(nRootNodes_range[0]-1, nRootNodes_range[1]):
        N_on = i+1
        p_cdf[i] = abs(norm.cdf(N_on-0.5, loc=nRootNodes_centre, scale=nRootNodes_std)
                                        - norm.cdf(N_on+0.5, loc=nRootNodes_centre, scale=nRootNodes_std))
    # CALCULATE MISSING RESULTS
    global experiment_output
    experiment_output = target_type
    global is_ECGi
    is_ECGi = False
    # Paths and tags
    dataPath = 'data/' + meshName + '/'
    # Load mesh
    global nodesXYZ
    nodesXYZ = np.loadtxt(dataPath + meshName + '_coarse_xyz.csv', delimiter=',')
    global edges
    edges = (np.loadtxt(dataPath + meshName + '_coarse_edges.csv', delimiter=',') - 1).astype(int)
    lvface = np.unique((np.loadtxt(dataPath + meshName + '_coarse_lvface_checked.csv',
                                   delimiter=',') - 1).astype(int)) # lv endocardium triangles
    rvface = np.unique((np.loadtxt(dataPath + meshName + '_coarse_rvface_checked.csv',
                                   delimiter=',') - 1).astype(int)) # rv endocardium triangles
    global tetraFibers
    tetraFibers = np.loadtxt(dataPath + meshName + '_coarse_tetrahedronFibers.csv', delimiter=',') # tetrahedron fiber directions
    tetraFibers = np.reshape(tetraFibers, [tetraFibers.shape[0], 3, 3], order='F')
    global edgeVEC
    edgeVEC = nodesXYZ[edges[:, 0], :] - nodesXYZ[edges[:, 1], :] # edge vectors
    lvActivationIndexes = (np.loadtxt(dataPath + meshName + '_coarse_lv_activationIndexes_'+rootNodeResolution+'Res.csv', delimiter=',') - 1).astype(int) # possible root nodes for the chosen mesh
    rvActivationIndexes = (np.loadtxt(dataPath + meshName + '_coarse_rv_activationIndexes_'+rootNodeResolution+'Res.csv', delimiter=',') - 1).astype(int)
    global rootNodeActivationIndexes
    rootNodeActivationIndexes = np.concatenate((lvActivationIndexes, rvActivationIndexes), axis=0)
    nRootLocations=rootNodeActivationIndexes.shape[0]
    nparam = nlhsParam + nRootLocations
    if is_ECGi:
        rootNodesIndexes_true = rootNodeActivationIndexes
    else:
        rootNodesIndexes_true = (np.loadtxt(dataPath + meshName + '_coarse_rootNodes.csv') - 1).astype(int)
        rootNodesIndexes_true = np.unique(rootNodesIndexes_true)
    global target_rootNodes
    target_rootNodes = nodesXYZ[rootNodesIndexes_true, :]
    param_boundaries = np.concatenate((np.array([gtRange*gf_factor, gtRange, gtRange*gn_factor, endoRange]), np.array([[0, 1] for i in range(nRootLocations)])))
    if experiment_output == 'ecg':
        global tetrahedrons
        tetrahedrons = (np.loadtxt(dataPath + meshName + '_coarse_tri.csv', delimiter=',') - 1).astype(int)
        tetrahedronCenters = np.loadtxt(dataPath + meshName + '_coarse_tetrahedronCenters.csv', delimiter=',')
        ecg_electrodePositions = np.loadtxt(dataPath + meshName + '_electrodePositions.csv', delimiter=',')
        if experiment_output == 'ecg':
            nb_leads = 8  # Originally 9, I took out the Lead progression because due to the downsampling sometimes it looks really bad  # 8 + lead progression (or 12)
            electrodePositions = ecg_electrodePositions
        nb_bsp = electrodePositions.shape[0]

        aux = [[] for i in range(0, nodesXYZ.shape[0], 1)]
        for i in range(0, tetrahedrons.shape[0], 1):
            aux[tetrahedrons[i, 0]].append(i)
            aux[tetrahedrons[i, 1]].append(i)
            aux[tetrahedrons[i, 2]].append(i)
            aux[tetrahedrons[i, 3]].append(i)
        global elements
        elements = [np.array(n) for n in aux]
        aux = None # Clear Memory

        # Precompute PseudoECG stuff
        # Calculate the tetrahedrons volumes
        D = nodesXYZ[tetrahedrons[:, 3], :] # RECYCLED
        A = nodesXYZ[tetrahedrons[:, 0], :]-D # RECYCLED
        B = nodesXYZ[tetrahedrons[:, 1], :]-D # RECYCLED
        C = nodesXYZ[tetrahedrons[:, 2], :]-D # RECYCLED
        D = None # Clear Memory

        global tVolumes
        tVolumes = np.reshape(np.abs(np.matmul(np.moveaxis(A[:, :, np.newaxis], 1, -1
                                               ), (np.cross(B, C)[:, :, np.newaxis]))),
                              tetrahedrons.shape[0]) #Tetrahedrons volume, no need to divide by 6 since it's being normalised by the sum which includes this 6 scaling factor
        tVolumes = tVolumes/np.sum(tVolumes)

        # Calculate the tetrahedron (temporal) voltage gradients
        Mg = np.stack((A, B, C), axis=-1)
        A = None # Clear Memory
        B = None # Clear Memory
        C = None # Clear Memory

        # Calculate the gradients
        global G_pseudo
        G_pseudo = np.zeros(Mg.shape)
        for i in range(Mg.shape[0]):
            G_pseudo[i, :, :] = np.linalg.inv(Mg[i, :, :])
            # If you obtain a Singular Matrix error type, this may be because one of the elements in the mesh is
            # really tinny if you are using a truncated mesh generated with Paraview, the solution is to do a
            # crinkle clip, instead of a regular smooth clip, making sure that the elements are of similar size
            # to each other. The strategy to identify the problem is to search for what element in Mg is giving
            # a singular matrix and see what makes it "special".
        G_pseudo = np.moveaxis(G_pseudo, 1, 2)
        Mg = None # clear memory

        # Calculate gradient of the electrode over the tetrahedrom centre, normalised by the tetrahedron's volume
        r=np.moveaxis(np.reshape(np.repeat(tetrahedronCenters, electrodePositions.shape[0], axis=1),
               (tetrahedronCenters.shape[0],
                tetrahedronCenters.shape[1], electrodePositions.shape[0])), 1, -1)-electrodePositions

        global d_r
        d_r= np.moveaxis(np.multiply(
            np.moveaxis(r, [0, 1], [-1, -2]),
            np.multiply(np.moveaxis(np.sqrt(np.sum(r**2, axis=2))**(-3), 0, 1), tVolumes)), 0, -1)
    elif experiment_output == 'atm':
        #outputTag = 'ATMap'
        global epiface
        epiface = np.unique((np.loadtxt(dataPath + meshName + '_coarse_epiface.csv', delimiter=',') - 1).astype(int)) # epicardium nodes
        global epiface_tri
        epiface_tri = (np.loadtxt(dataPath + meshName + '_coarse_epiface.csv', delimiter=',') - 1).astype(int) # epicardium nodes
    else:
        raise
    # Set endocardial edges aside
    global isEndocardial
    isEndocardial=np.logical_or(np.all(np.isin(edges, lvface), axis=1),
                          np.all(np.isin(edges, rvface), axis=1))

    # Build adjacentcies
    global unfoldedEdges
    unfoldedEdges = np.concatenate((edges, np.flip(edges, axis=1))).astype(int)
    aux = [[] for i in range(0, nodesXYZ.shape[0], 1)]
    for i in range(0, len(unfoldedEdges), 1):
        aux[unfoldedEdges[i, 0]].append(i)
    global neighbours
    # make neighbours Numba friendly
    neighbours_aux = [np.array(n) for n in aux]
    aux = None # Clear Memory
    m = 0
    for n in neighbours_aux:
        m = max(m, len(n))
    neighbours = np.full((len(neighbours_aux), m), np.nan, np.int32) # needs to be float because np.nan is float, otherwise it casts np.nan to an actual number
    for i in range(len(neighbours_aux)):
        n = neighbours_aux[i]
        neighbours[i, :n.shape[0]] = n
    neighbours_aux = None

    # neighbours_original
    aux = [[] for i in range(0, nodesXYZ.shape[0], 1)]
    for i in range(0, len(unfoldedEdges), 1):
        aux[unfoldedEdges[i, 0]].append(i)
    global neighbours_original
    neighbours_original = [np.array(n) for n in aux]
    aux = None # Clear Memory
    
    # Load and compile Numba
    compilation_params = np.array([np.concatenate((np.array([0.1, 0.1, 0.1, 0.1]),
                                       np.ones(rootNodeActivationIndexes.shape).astype(int)))])
    eikonal_ecg(compilation_params, rootNodeActivationIndexes)
    compProb(np.array([[0, 1, 0, 1]]), np.array([[0, 1, 0, 1]]), retain_ratio, nRootNodes_centre, nRootNodes_std, p_cdf)
    
    # Target result
    if load_target:
        if experiment_output == 'atm':
            target_output = np.loadtxt(target_fileName)[epiface]
        else:
            target_output = np.loadtxt(target_fileName)
        print(target_output.shape)
    else:
        target_output = eikonal_ecg(np.array([conduction_speeds])/1000, rootNodesIndexes_true)
        if experiment_output == 'atm':
            target_output = target_output[0, epiface]
        else:
            target_output = target_output[0, :, :]
            target_output = target_output[:, np.logical_not(np.isnan(target_output[0, :]))]
            
    if target_snr_db > 0:
        if experiment_output == 'ecg':
            target_output_aux = np.zeros((target_output.shape[0], target_output.shape[1]+200))
            target_output_aux[:, 100:-100] = target_output
            target_output = target_output_aux
        
            # Add noise white Gaussian noise to the signals using target SNR
            ecg_watts = target_output ** 2
            # Calculate signal power and convert to dB
            sig_avg_watts = np.mean(ecg_watts, axis=1)
            sig_avg_db = 10 * np.log10(sig_avg_watts)
            # Calculate noise according to [2] then convert to watts
            noise_avg_db = sig_avg_db - target_snr_db
            noise_avg_watts = 10 ** (noise_avg_db / 10)
            # Generate an sample of white noise
            target_output_noised = np.zeros(target_output.shape)
            mean_noise = 0 # white Gaussian noise
            for i in range(target_output_noised.shape[0]):
                noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts[i]), ecg_watts.shape[1])
                # Noise up the original signal
                target_output_noised[i,:] = target_output[i,:] + noise_volts
            # Denoise the noised signal
            freq_cut= 100 # Cut-off frequency of the filter Hz
            w = freq_cut / (frequency / 2) # Normalize the frequency
            b_filt, a_filt = signal.butter(8, w, 'low')
            target_output_denoised = signal.filtfilt(b_filt, a_filt, target_output_noised) # Filter ECG signal
    
            target_output = target_output[:, 100:-100]
            target_output_denoised = target_output_denoised[:, 100:-100]
            target_output_noised = target_output_noised[:, 100:-100]
            
            print('noise correlation: ' + str(np.mean(np.asarray([np.corrcoef(target_output[i, :], target_output_denoised[i, :])[0,1]
                for i in range(target_output.shape[0])]))))
            target_output = target_output_denoised # always work with the denoised ECG as if obtained in the clinic
        
        else: # Activation map data
            # Add noise white Gaussian noise to the signals using target SNR
            ecg_watts = target_output ** 2
            # Calculate signal power and convert to dB
            sig_avg_watts = np.mean(ecg_watts)
            sig_avg_db = 10 * np.log10(sig_avg_watts)
            # Calculate noise according to [2] then convert to watts
            noise_avg_db = sig_avg_db - target_snr_db
            noise_avg_watts = 10 ** (noise_avg_db / 10)
            # Generate an sample of white noise
            mean_noise = 0 # white Gaussian noise
            noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), ecg_watts.shape[0])
            # Noise up the original signal
            target_output_noised = target_output + noise_volts
            
            print('noise correlation: ' + str(np.corrcoef(target_output, target_output_noised)[0,1]))
            target_output = target_output_noised


    global dtw_lead_weights # prepare global variable used for the dtw computation for bsp data

    if experiment_output == 'ecg':
        dtw_lead_weights = np.ones((nb_leads)).astype(float)
    else:
        dtw_lead_weights = None
    print(dtw_lead_weights)
    t_start = time.time()
    print('Starting ' + final_path)
    sampler = ABCSMC(nparam=nparam,
                     simulator=eikonal_ecg,
                     target_data=target_output,
                     desired_Discrepancy=desired_Discrepancy,
                     max_MCMC_steps=max_MCMC_steps,
                     param_boundaries=param_boundaries,
                     nlhsParam=nlhsParam,
                     maxRootNodeJiggleRate=0.1,
                     nRootLocations=nRootLocations,
                     retain_ratio=retain_ratio,
                     nRootNodes_range=nRootNodes_range,
                     resultFile=final_path,
                     experiment_output=experiment_output,
                     experiment_metric=metric,
                     hermite_order=None,
                     hermite_mean=None,
                     hermite_std=None,
                     nRootNodes_centre=nRootNodes_centre,
                     nRootNodes_std=nRootNodes_std,
                     npart=npart,
                     keep_fraction=keep_fraction,
                     conduction_speeds=conduction_speeds,
                     target_rootNodes=rootNodesIndexes_true
                    )
    sampler.sample(desired_Discrepancy)
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    else:
        print("The file:"+tmp_path+"\n does not exist")
    
    print('Finished: '+final_path)
    
# ------------------------------------------- DISCRETE SMC-ABC FUNCTIONS ----------------------------------------------------

def jiggleDiscreteNonFixed(n, part_binaries, retain_ratio, nRootNodes_centre, nRootNodes_std, nRootNodes_range):
    n_parts, n_vars = part_binaries.shape
    alpha = n_parts * (1 - retain_ratio) / (retain_ratio * n_vars - 1)
    on = (np.zeros((n, n_vars))).astype(bool)
    for i in range(n):
        if np.random.uniform() < 0.8:
            N_on = int(round(sum(part_binaries[int(np.random.randint(0, part_binaries.shape[0])), :])))
        else:
            N_on = int(round(np.random.normal(nRootNodes_centre, 1)))
            # Too expensive to compute when the threshold is the centre of the distribution
            # thus, simplified version isntead
#             N_on = int(min(nRootNodes_range[1], N_on))
#             while N_on < nRootNodes_range[0]:# or N_on > nRootNodes_range[1]:
            while N_on < nRootNodes_range[0] or N_on > nRootNodes_range[1]:
                N_on = int(round(np.random.normal(nRootNodes_centre, 1)))
#                 N_on = int(min(nRootNodes_range[1], N_on))
        
        # Use only the probability of the particles with same number of nodes active
        part_binaries_N = part_binaries[np.sum(part_binaries, axis=1)==N_on, :]
        for j in range(N_on):
            open_sites = np.nonzero(np.logical_not(on[i, :]))[0]
            w = np.random.dirichlet(alpha + np.sum( part_binaries_N[
                np.all(part_binaries_N[:, on[i, :]], axis=1), :][:, open_sites], axis=0), 1)[0]
            r = open_sites[(np.random.multinomial(1, w)).astype(bool)]
            on[i, r] = True
    return on


@numba.njit
def compProb(new_binaries, part_binaries, retain_ratio, nRootNodes_centre, nRootNodes_std, p_cdf):
    # This code can be compiled as non-python code by Numba, which makes it quite fast
    n_parts, n_vars = part_binaries.shape
    alpha = n_parts * (1 - retain_ratio) / (retain_ratio * n_vars - 1)
    
    p_trial_i = 0.
    N_on = int(np.sum(new_binaries))
    p_nRootNodes = (0.8 * np.sum(np.sum(part_binaries, axis=1)==N_on)/part_binaries.shape[0] + 0.2 * p_cdf[int(N_on-1)])
    if N_on < 9:
        part_binaries_N = part_binaries[np.sum(part_binaries, axis=1)==N_on, :]
        # Permutations that Numba can understand
        A = np.nonzero(new_binaries)[0]
        k = len(A)
        numba_permutations = [[i for i in range(0)]]
        for i in range(k):
            numba_permutations = [[a] + b for a in A for b in numba_permutations if (a in b)==False]
        for combo in numba_permutations:
            on = np.zeros((n_vars), dtype=np.bool_)
            p_trial_here = p_nRootNodes
            for j in range(len(combo)):
                pb = part_binaries_N[:, on]
                aux_i = np.empty((pb.shape[0]), dtype=np.bool_)
                for part_i in range(pb.shape[0]):
                    aux_i[part_i] = np.all(pb[part_i])
                aux_p = part_binaries_N[aux_i, :]
                aux = np.sum(aux_p[:, combo[j]], axis=0)
                aux1 = ((n_vars - j) * alpha + np.sum((aux_p[:, np.logical_not(on)])))
                aux2 = (alpha + aux) / aux1
                p_trial_here *= aux2
                on[combo[j]] = 1
            p_trial_i += p_trial_here
    else:
        p_trial_i = N_on/n_vars
        for i in range(1, N_on):
            p_trial_i *= i/(n_vars-i)
        p_trial_i *= p_nRootNodes
    return p_trial_i


def generateRootNodes(n, nRootLocs, nRootNodes_centre, nRootNodes_std, nRootNodes_range):
    rootNodes = np.zeros((n, nRootLocs))
    for i in range(n):
        N_on = 0
        while N_on < nRootNodes_range[0] or N_on > nRootNodes_range[1]:
            N_on = int(round(np.random.normal(loc=nRootNodes_centre, scale=nRootNodes_std)))
        rootNodes[i, np.random.permutation(nRootLocs)[:N_on-1]] = 1
        rootNodes[i, i%nRootLocs] = 1 # Ensure that all root nodes are represented at least once
    return rootNodes


def doNothing(X):
    return X


# ------------------------------------------- DISTANCE FUNCTIONS ----------------------------------------------------
def dtw_ecg(predicted, target_ecg, window=None, time_penalty=None, max_slope=1.5, w_max=10.):
    """Dynamic Time Warping distance specific for comparing electrocardiogram signals.
    It implements a trianglogram constraint (inspired from Itakura parallelogram (Itakura, 1975)).
    It also implements weight penalty with a linear increasing cost away from the true diagonal (i.e. i=j).
    Moreover, it implements a step-pattern with slope-P value = 0.5 from (Sakoe and Chiba, 1978).
    Finally, the algorithm penalises the difference between the lenght of the two signals and adds it to the DTW distance.
    Options
    -------
    max_slope : float Maximum slope of the trianglogram.
    w_max : float weigth coeficient to the distance from the diagonal.
    small_c :  float weight coeficient to the difference in lenght between the signals being compared.
    References
    ----------
    .. [1] F. Itakura, "Minimum prediction residual principle applied to
           speech recognition". IEEE Transactions on Acoustics,
           Speech, and Signal Processing, 23(1), 67–72 (1975).
    .. [2] H. Sakoe and S. Chiba, "Dynamic programming algorithm optimization
           for spoken word recognition". IEEE Transactions on Acoustics,
           Speech, and Signal Processing, 26(1), 43-49 (1978).
    """
    # Don't compute repetitions
    # Numpy Unique has an issue with NaN values: https://github.com/numpy/numpy/issues/2111
    small_c = 0.05 * 171 / meshVolume
    p = np.copy(predicted[:, 0, :])
#     print(p.shape)
#     print('dk')
    p[np.isnan(p)]=np.inf
    p, particle_index, unique_index = np.unique(p, return_index=True, return_inverse=True, axis=0)
#     print(unique_index)
#     print(unique_index.shape)
    p=None
    predicted = predicted[particle_index, :, :]
#     print(predicted.shape)
    
    # This code has parallel pympy sections as well as numba parallel sections
    nParts = predicted.shape[0]
    nLeads = predicted.shape[1]
    # res = np.zeros((nParts))
    # if True:
    #     for conf_i in range(0, nParts):
    res = pymp.shared.array(nParts, dtype='float64')
    with pymp.Parallel(min(nParts, threadsNum)) as p1:
        for conf_i in p1.range(0, nParts):
            mask = np.logical_not(np.isnan(predicted[conf_i, :, :]))
            pred_ecg = np.squeeze(predicted[conf_i:conf_i+1, :, mask[0, :]])  # using slicing index does not change the shape of the object,
                                                                              # however, mixing slicing with broadcasting does change it, which then
                                                                              # requires moving the axis with np.moveaxis
            # Lengths of each sequence to be compared
            n_timestamps_1 = pred_ecg.shape[1]
            n_timestamps_2 = target_ecg.shape[1]
            
            # Computes the region (in-window area using a trianglogram)
            # WARNING: this is a shorter version of the code for generating the region which does not account for special cases, the full version is the fuction "trianglorgram" from myfunctions.py
            max_slope_ = max_slope
            min_slope_ = 1 / max_slope_
            scale_max = (n_timestamps_2 - 1) / (n_timestamps_1 - 2)
            max_slope_ *= scale_max
            scale_min = (n_timestamps_2 - 2) / (n_timestamps_1 - 1)
            min_slope_ *= scale_min
            centered_scale = np.arange(n_timestamps_1) - n_timestamps_1 + 1
            lower_bound = min_slope_ * np.arange(n_timestamps_1)
            lower_bound = np.round(lower_bound, 2)
            lower_bound = np.floor(lower_bound) # Enforces that at least one pixel is available when we take out the restriction that the true diagonal should be always available to the wraping path
            upper_bound = max_slope_ * np.arange(n_timestamps_1) + 1
            upper_bound = np.round(upper_bound, 2)
            upper_bound = np.ceil(upper_bound) # Enforces that at least one pixel is available when we take out the restriction that the true diagonal should be always available to the wraping path
            region_original = np.asarray([lower_bound, upper_bound]).astype('int64')
            region_original = np.clip(region_original[:, :n_timestamps_1], 0, n_timestamps_2) # Project region on the feasible set
            
            part_dtw = 0.
            # Compute the DTW for each lead and particle separately so that leads can be wraped differently from each other
            for lead_i in range(nLeads):
                region = np.copy(region_original)
                x = pred_ecg[lead_i, :]
                y = target_ecg[lead_i, :]
                
                # Computes cost matrix from dtw input
                dist_ = lambda x, y : (x - y) ** 2

                # Computs the cost matrix considering the window (0 inside, np.inf outside)
                cost_mat = np.full((n_timestamps_1, n_timestamps_2), np.inf)
                m = np.amax(cost_mat.shape)
                for i in numba.prange(n_timestamps_1):
                    for j in numba.prange(region[0, i], region[1, i]):
                        cost_mat[i, j] = dist_(x[i], y[j]) * (w_max * abs(i-j)/max(1., (i+j))+1.) # This new weight considers that wraping in time is cheaper the later it's done #* (w_max/(1+math.exp(-g*(abs(i-j)-m/2)))+1.) # + abs(i-j)*small_c # Weighted version of the DTW algorithm

                # Computes the accumulated cost matrix
                acc_cost_mat = np.full((n_timestamps_1, n_timestamps_2), np.inf)
                acc_cost_mat[0, 0: region[1, 0]] = np.cumsum(
                    cost_mat[0, 0: region[1, 0]]
                )
                acc_cost_mat[0: region[1, 0], 0] = np.cumsum(
                    cost_mat[0: region[1, 0], 0]
                )
                region_ = np.copy(region)
                region_[0] = np.maximum(region_[0], 1)
                for i in range(1, n_timestamps_1):
                    for j in range(region_[0, i], region_[1, i]):
                        # Implementation of a Slope-constraint as a step-pattern:
                        # This constraint will enforce that the algorithm can only take up to 2 consecutive steps along the time wraping directions.
                        # I decided to make it symetric because in (Sakoe and Chiba, 1978) they state that symetric means that DTW(A, B) == DTW(B, A), although I am not convinced why it's not the case in the asymetric implementation.
                        # Besides, the asymetric case has a bias towards the diagonal which I thought could be desirable in our case, that said, having DTW(A, B) == DTW(B, A) may prove even more important, especially in further
                        # applications of this algorithm for ECG comparison.
                        # This implementation is further explained in (Sakoe and Chiba, 1978) and correspondes to the one with P = 0.5, (P = n/m, where P is a rule being inforced, n is the number of steps in the diagonal
                        # direction and m is the steps in the time wraping direction).
                        acc_cost_mat[i, j] = min(
                            acc_cost_mat[i - 1, j-3] + 2*cost_mat[i, j-2] + cost_mat[i, j-1] + cost_mat[i, j],
                            acc_cost_mat[i - 1, j-2] + 2*cost_mat[i, j-1] + cost_mat[i, j],
                            acc_cost_mat[i - 1, j - 1] + 2*cost_mat[i, j],
                            acc_cost_mat[i - 2, j-1] + 2*cost_mat[i-1, j] + cost_mat[i, j],
                            acc_cost_mat[i - 3, j-1] + 2*cost_mat[i-2, j] + cost_mat[i-1, j] + cost_mat[i, j]
                        )
                dtw_dist = acc_cost_mat[-1, -1]/(n_timestamps_1 + n_timestamps_2)
                part_dtw += math.sqrt(dtw_dist) #I would rather have leads not compensating for each other
            res[conf_i] = part_dtw / nLeads + small_c * (n_timestamps_1-n_timestamps_2)**2/min(n_timestamps_1,n_timestamps_2)
    return res[unique_index]


def computeDiscrepancyForATMaps(predicted, target, window=None, time_penalty=None):
    res = np.sqrt(np.mean((predicted[:, epiface] - target)**2, axis=1))
    return res


def computeATMerror(prediction_list, target):
    error = np.sqrt(np.mean((prediction_list[:, epiface] - target[np.newaxis, :])**2, axis=1))
    return error


# ------------------------------------------- CORE SMC-ABC FUNCTIONS ----------------------------------------------------
class ABCSMC(object):

    def __init__(self, nparam, simulator, target_data,
                 desired_Discrepancy,
                 max_MCMC_steps,
                 param_boundaries, nlhsParam,
                 maxRootNodeJiggleRate,
                 retain_ratio, nRootNodes_range,
                 resultFile, experiment_output,
                 experiment_metric, hermite_mean, hermite_std, hermite_order,
                 nRootNodes_centre, nRootNodes_std, nRootLocations, npart, keep_fraction,
                 conduction_speeds, target_rootNodes):

        self.experiment_output = experiment_output
        self.experiment_metric = experiment_metric
        self.data = target_data
        self.nRootNodes_range = nRootNodes_range
        self.nRootNodes_centre = nRootNodes_centre
        self.nRootNodes_std = nRootNodes_std
        self.nRootLocations = nRootLocations
        self.npart = npart
        self.keep_fraction = keep_fraction
        self.max_MCMC_steps = max_MCMC_steps
        self.simulator = simulator
        self.nparam = nparam
        self.nlhsParam = nlhsParam
        self.param_boundaries = param_boundaries
        self.conduction_speeds = conduction_speeds
        self.target_rootNodes = target_rootNodes
        self.window = 32
        self.lim_window = 5
        self.time_penalty = 1.

        self.nRootLocations = nRootLocations
        part_theta = self.iniParticles(self.npart, nparam, param_boundaries)
        self.part_theta = part_theta
        part_output = self.simulator(self.part_theta, rootNodeActivationIndexes)
        if self.experiment_output == 'ecg':
            if self.experiment_metric == 'dtw':
                self.f_discrepancy = dtw_ecg
                self.f_summary = doNothing
                self.f_compProb = compProb
                self.f_discrepancy_plot = dtw_trianglorgram
        elif self.experiment_output == 'atm':
            self.f_discrepancy = computeDiscrepancyForATMaps
            self.f_summary = doNothing
            self.f_compProb = compProb
        self.part_discrepancy = self.f_discrepancy(part_output, self.data, self.window, self.time_penalty)
        self.maxRootNodeJiggleRate = maxRootNodeJiggleRate
        self.retain_ratio = retain_ratio
        self.resultFile=resultFile

    
    def iniParticles(self, npart, nparam, ranges):
        # Do LHS only for Transversal and Endocardial speed and adjust the others
        lhs_theta = lhs(2, samples=npart, criterion='maximin')
        part_theta = np.zeros((npart, self.nlhsParam))
        
        part_theta[:, 1] = lhs_theta[:, 0] * (self.param_boundaries[1, 1] - self.param_boundaries[1, 0]
                                             ) + self.param_boundaries[1, 0] # sheet speed
        part_theta[:, 0] = part_theta[:, 1] * gf_factor # fibre speed
        part_theta[:, 2] = part_theta[:, 1] * gn_factor # sheet-normal speed
        part_theta[:, 3] = lhs_theta[:, 0] * (self.param_boundaries[3, 1] - self.param_boundaries[3, 0]
                                                 ) + self.param_boundaries[3, 0] # endocardial speed
        
        # by leaving 4 decimals we keep a precision of 1 mm/s, knowing that values will be around 100-20 cm/s
        part_theta = np.round(part_theta, decimals=4)
        part_theta = np.maximum(
                np.minimum(part_theta, self.param_boundaries[:self.nlhsParam, 1]),
                self.param_boundaries[:self.nlhsParam, 0])
        rootNodes = generateRootNodes(npart, self.nRootLocations, self.nRootNodes_centre, self.nRootNodes_std, self.nRootNodes_range)
        part_theta = np.concatenate((part_theta, rootNodes), axis=1)
        
        return part_theta
    
    
    def MVN_move(self, Ctheta, cuttoffD, nMoves, jiggleIndex, npartMove):
        accepted = np.empty((npartMove), dtype=np.bool_)
        

        for i in range(nMoves):
            current_theta = self.part_theta[jiggleIndex:, :self.nlhsParam]
            copiedRootNodes = self.part_theta[jiggleIndex:, self.nlhsParam:]
            currentRootNodes = self.part_theta[:jiggleIndex, self.nlhsParam:]

            copiedRootNodes_unique, unique_indexes = np.unique(copiedRootNodes, return_inverse=True, axis=0)
            copied_probs = pymp.shared.array((copiedRootNodes_unique.shape[0]), dtype=np.float64)
            with pymp.Parallel(min(threadsNum, copiedRootNodes_unique.shape[0])) as p1:
                for j in p1.range(copiedRootNodes_unique.shape[0]):
                    copied_probs[j] = self.f_compProb(copiedRootNodes_unique[j, :], currentRootNodes, self.retain_ratio,
                                     self.nRootNodes_centre, self.nRootNodes_std, p_cdf)

            copied_probs = copied_probs[unique_indexes]
            # Jiggle discrete parameters
            prop_rootNodes = jiggleDiscreteNonFixed(npartMove, currentRootNodes, self.retain_ratio,
                                                      self.nRootNodes_centre, self.nRootNodes_std,
                                                    self.nRootNodes_range)

            t_s = time.time()
            prop_rootNodes_unique, unique_indexes = np.unique(prop_rootNodes, return_inverse=True, axis=0)
            prop_probs = pymp.shared.array((prop_rootNodes_unique.shape[0]), dtype=np.float64)
            with pymp.Parallel(min(threadsNum, prop_rootNodes_unique.shape[0])) as p1:
                for j in p1.range(prop_rootNodes_unique.shape[0]):
                    prop_probs[j] = self.f_compProb(prop_rootNodes_unique[j, :], currentRootNodes, self.retain_ratio,
                                     self.nRootNodes_centre, self.nRootNodes_std, p_cdf)
            prop_probs = prop_probs[unique_indexes]
            non_select_roots = np.random.rand(npartMove) > copied_probs/prop_probs
            prop_rootNodes[non_select_roots, :] = copiedRootNodes[non_select_roots, :]
            prop_theta = np.concatenate((current_theta, prop_rootNodes), axis=1)
            prop_output = self.simulator(prop_theta, rootNodeActivationIndexes)
            prop_discrepancy = self.f_discrepancy(prop_output, self.data, self.window, self.time_penalty)
            accepted = np.logical_or((prop_discrepancy < cuttoffD), accepted)
            self.part_theta[jiggleIndex:][(prop_discrepancy < cuttoffD)] = prop_theta[
                (prop_discrepancy < cuttoffD)]
            self.part_discrepancy[jiggleIndex:][(prop_discrepancy < cuttoffD)] = prop_discrepancy[
                (prop_discrepancy < cuttoffD)]
            
            # Jiggle continuous parameters
            # May have changed
            jiggledRootNodes = self.part_theta[jiggleIndex:, self.nlhsParam:]
            prop_theta = (current_theta + np.random.multivariate_normal(
                np.zeros((self.nlhsParam,)), Ctheta, size=npartMove))
            prop_theta = np.round(prop_theta, decimals=4) # precision of 1 mm/s
            # Regulate max and min parameter boundaries
            prop_theta = np.maximum(
                np.minimum(prop_theta, self.param_boundaries[:self.nlhsParam, 1]),
                self.param_boundaries[:self.nlhsParam, 0])
            # Make sure that gt <= gl and gn <= gt
            prop_theta[:, 0] = np.maximum(prop_theta[:, 0], prop_theta[:, 1])
            prop_theta[:, 2] = np.minimum(prop_theta[:, 1], prop_theta[:, 2])
            prop_theta = np.concatenate((prop_theta, jiggledRootNodes), axis=1)
            prop_output = self.simulator(prop_theta, rootNodeActivationIndexes)
            prop_discrepancy = self.f_discrepancy(prop_output, self.data, self.window, self.time_penalty)
            accepted = np.logical_or((prop_discrepancy < cuttoffD), accepted)
            self.part_theta[jiggleIndex:][(prop_discrepancy < cuttoffD)] = prop_theta[
                (prop_discrepancy < cuttoffD)]
            self.part_discrepancy[jiggleIndex:][(prop_discrepancy < cuttoffD)] = prop_discrepancy[
                (prop_discrepancy < cuttoffD)]
        return accepted

    
    def sample(self, targetDiscrepancy):
        looping = True
        self.visualise(int(np.argmin(self.part_discrepancy)))
        visualise_count_ini = 3 # zero will print at every iteration
        visualise_count = visualise_count_ini
        worst_keep = int(np.round(self.npart*self.keep_fraction))
        ant_part_theta = None
        ant_discrepancy = None
        ant_cutoffdiscrepancy = None
        iteration_count = 0

        window_count_ini = 4
        window_count = window_count_ini
        worst_keep_ini = int(np.round(self.npart*self.keep_fraction))
        worst_keep = worst_keep_ini
        while looping:
            index = np.argsort(self.part_discrepancy)
            self.part_discrepancy = self.part_discrepancy[index]
            self.part_theta = self.part_theta[index, :]
            
            # Select the new cuttoff discrepancy
            cuttoffDiscrepancy = self.part_discrepancy[worst_keep]
            selection = np.random.randint(low=0, high=worst_keep, size=self.npart-(worst_keep))
            self.part_theta[worst_keep:] = self.part_theta[selection]
            self.part_discrepancy[worst_keep:] = self.part_discrepancy[selection]
            
            # Optimal factor
            Ctheta = 2.38**2/self.nlhsParam * np.cov(self.part_theta[:, :self.nlhsParam].T)
            
            est_accept = self.MVN_move(Ctheta, cuttoffDiscrepancy, 1, worst_keep, len(selection))
            est_accept_rate = np.mean(est_accept)
            MCMC_steps = min(
                math.ceil(math.log(0.05)/math.log(1-min(max(est_accept_rate, 1e-8), 1-1e-8))),
                self.max_MCMC_steps)
            # Run the remaining MCMC steps to compleete the amount just calculated
            t_s = time.time()
            accepted = self.MVN_move(Ctheta, cuttoffDiscrepancy, MCMC_steps-1, worst_keep, len(selection))
            nUnique = len(np.unique(self.part_theta, axis=0))
            bestInd = np.argmin(self.part_discrepancy)
            
            # Stoping criteria
#             if nUnique < int(np.round(self.npart*self.keep_fraction)) or cuttoffDiscrepancy < targetDiscrepancy:
            unique_lim_nb = int(np.round(self.npart*0.5))
            if nUnique < unique_lim_nb or cuttoffDiscrepancy < targetDiscrepancy:
                looping = 0
                np.savetxt(self.resultFile, self.part_theta, delimiter=',')
                # CAUTION! SINCE DISCREPANCY IS ONLY A LIST, THIS WON'T ADD ANY ',' AND WILL IGNORE THE DELIMITER
                np.savetxt(self.resultFile.replace('population', 'discrepancy'), self.part_discrepancy, delimiter=',')
                prop_data = self.simulator(np.array([self.part_theta[bestInd, :]]), rootNodeActivationIndexes)
                if self.experiment_output == 'ecg':
                    prop_data = prop_data[0, :, :]
                elif self.experiment_output == 'atm':
                    prop_data = prop_data[0, :]
                np.savetxt(self.resultFile.replace('population', 'prediction'), prop_data, delimiter=',')
                visualise_count = 0


            if visualise_count < 1:
                randId = random.randint(0, self.npart-1)
                self.visualise(randId)
                visualise_count = visualise_count_ini
                rootNodesIndexes = []
                for i in range(self.part_theta.shape[0]):
                    x = self.part_theta[i, nlhsParam:]
                    y = np.empty_like(x)
                    rootNodesParam = np.round_(x, 0, y)
                    y = None
                    rootNodesIndexes.append(rootNodeActivationIndexes[rootNodesParam==1])
                rootNodesIndexes = np.concatenate(rootNodesIndexes, axis=0)
                
                fig = plt.figure(constrained_layout=True, figsize = (15,10))
                fig.suptitle(self.resultFile +' nUnique % ' + str(nUnique/self.npart*100), fontsize=24)
                ax = fig.add_subplot(231, projection='3d')
                pred_roots = nodesXYZ[rootNodesIndexes, :]
                target_roots = nodesXYZ[self.target_rootNodes, :]
                ax.scatter(pred_roots[:, 0], pred_roots[:, 1], pred_roots[:, 1], c='b', marker='o', s=10)
                if not is_ECGi:
                    ax.scatter(target_roots[:, 0], target_roots[:, 1], target_roots[:, 1], c='r', marker='o', s=50)
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('z')
                
                ax = fig.add_subplot(232, projection='3d')
                ax.scatter(pred_roots[:, 0], pred_roots[:, 1], pred_roots[:, 1], c='b', marker='o', s=10)
                if not is_ECGi:
                    ax.scatter(target_roots[:, 0], target_roots[:, 1], target_roots[:, 1], c='r', marker='o', s=50)
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('z')
                ax.view_init(10, 0)
                
                ax = fig.add_subplot(233, projection='3d')
                ax.scatter(pred_roots[:, 0], pred_roots[:, 1], pred_roots[:, 1], c='b', marker='o', s=10)
                if not is_ECGi:
                    ax.scatter(target_roots[:, 0], target_roots[:, 1], target_roots[:, 1], c='r', marker='o', s=50)
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('z')
                ax.view_init(10, 45)
                
                ax = fig.add_subplot(234, projection='3d')
                ax.scatter(pred_roots[:, 0], pred_roots[:, 1], pred_roots[:, 1], c='b', marker='o', s=10)
                if not is_ECGi:
                    ax.scatter(target_roots[:, 0], target_roots[:, 1], target_roots[:, 1], c='r', marker='o', s=50)
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('z')
                ax.view_init(10, 90)
                
                ax = fig.add_subplot(235, projection='3d')
                ax.scatter(pred_roots[:, 0], pred_roots[:, 1], pred_roots[:, 1], c='b', marker='o', s=10)
                if not is_ECGi:
                    ax.scatter(target_roots[:, 0], target_roots[:, 1], target_roots[:, 1], c='r', marker='o', s=50)
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('z')
                ax.view_init(10, 135)
                
                ax = fig.add_subplot(236, projection='3d')
                ax.scatter(pred_roots[:, 0], pred_roots[:, 1], pred_roots[:, 1], c='b', marker='o', s=10)
                if not is_ECGi:
                    ax.scatter(target_roots[:, 0], target_roots[:, 1], target_roots[:, 1], c='r', marker='o', s=50)
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('z')
                ax.view_init(10, 180)
                
                plt.show()
                
                # Conduction speeds
                speed_name_list = ['Fibre speed', 'Sheet speed', 'Normal speed', 'Endocardial speed']
                fig, axs = plt.subplots(2, len(speed_name_list), constrained_layout=True, figsize = (20,10))
                fig.suptitle(meshName +' nUnique % ' + str(nUnique/self.npart*100), fontsize=24)
                # Prepare previous results
                if not ant_part_theta is None:
                    ant_good_particles = ant_discrepancy < cuttoffDiscrepancy
                    ant_good_theta = ant_part_theta[ant_good_particles, :]*1000
                    ant_bad_theta = ant_part_theta[np.logical_not(ant_good_particles), :]*1000
                # Iterate over speeds
                for speed_iter in range(len(speed_name_list)):
                    # Plot new results
                    axs[1, speed_iter].plot(self.part_theta[:worst_keep, speed_iter]*1000, self.part_discrepancy[:worst_keep], 'bo')
                    axs[1, speed_iter].plot(self.part_theta[worst_keep:, speed_iter]*1000, self.part_discrepancy[worst_keep:], 'go')
                    axs[1, speed_iter].plot(self.part_theta[randId, speed_iter]*1000, self.part_discrepancy[randId], 'ro')
                    axs[1, speed_iter].set_title('New ' + speed_name_list[speed_iter], fontsize=16)
                    if not is_ECGi and not (ant_part_theta is None):
                        true_speed_value = self.conduction_speeds[speed_iter]
                        old_median = np.median(ant_part_theta[:, speed_iter]*1000)
                        new_median = np.median(self.part_theta[:, speed_iter]*1000)
                        axs[1, speed_iter].axvline(x=self.conduction_speeds[speed_iter], c='grey')
                        if abs(true_speed_value - new_median) <= abs(true_speed_value - old_median):
                            axs[1, speed_iter].axvline(x=new_median, c='green')
                        else:
                            axs[1, speed_iter].axvline(x=new_median, c='red')
                            
                    # Plot previous results for comparison
                    if not ant_part_theta is None:
                        axs[0, speed_iter].plot(ant_good_theta[:, speed_iter], ant_discrepancy[ant_good_particles], 'go')
                        axs[0, speed_iter].plot(ant_bad_theta[:, speed_iter], ant_discrepancy[np.logical_not(ant_good_particles)], 'ro')
                        axs[0, speed_iter].set_title('Ant ' + speed_name_list[speed_iter], fontsize=16)
                        if not is_ECGi:
                            axs[0, speed_iter].axvline(x=self.conduction_speeds[speed_iter], c='grey')
                        axs[0, speed_iter].axvline(x=np.median(ant_part_theta[:, speed_iter]*1000), c='purple')
                        axs[0, speed_iter].axhline(y=cuttoffDiscrepancy, color='blue')
                    axs[1, speed_iter].set_xlabel('cm/s', fontsize=16)

                axs[0, 0].set_ylabel('discrepancy', fontsize=16)
                axs[1, 0].set_ylabel('discrepancy', fontsize=16)
                plt.show()
            else:
                # visualise_count = visualise_count -1 #TODO Change this to visualise the output
                pass
                
            ant_sort = np.argsort(self.part_discrepancy)
            ant_part_theta = self.part_theta[ant_sort, :]
            ant_discrepancy = self.part_discrepancy[ant_sort]
            
            iteration_count = iteration_count +1
            
        return self.part_theta
    
        
    def visualise(self, particle_index=None):

        if particle_index is not None:
            prop_data = self.simulator(self.part_theta, rootNodeActivationIndexes)
            if self.experiment_output == 'ecg':
                fig, axs = plt.subplots(2, 8, constrained_layout=True, figsize = (40,10))
                fig.suptitle(meshName + ' Window: ' + str(self.window) +'  Discrepancy: ' + str(self.part_discrepancy[particle_index]), fontsize=24)
                for i in range(8):
                    leadName = leadNames[i]
                    axs[0, i].plot(prop_data[0, i, :], 'b-', label='pred', linewidth=0.08)
                    axs[1, i].plot(prop_data[particle_index, i, :], 'b-', label='pred')
                    for j in range(1, prop_data.shape[0]):
                        axs[0, i].plot(prop_data[j, i, :], 'b-', linewidth=0.08)
                    axs[0, i].plot(self.data[i, :], 'r-', label='true')
                    axs[1, i].plot(self.data[i, :], 'r-', label='true')
                    axs[0, i].set_title('Lead '+leadName, fontsize=20)
                    aux = prop_data[particle_index, i, :]
                    aux = aux[~np.isnan(aux)]
                    dtw_cost, penalty_cost = self.f_discrepancy_plot(aux, self.data[i, :], self.window, self.time_penalty)
                    axs[1, i].set_title(str(round(dtw_cost, 2))+'   -   '+str(round(penalty_cost, 2)), fontsize=16)
                axs[0, i].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
                axs[1, i].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
                plt.show()
            else:
                print(meshName +' Activation Time Map Error: ' + str(computeATMerror(prop_data, self.data)[0]))
        return None


def insertSorted(aList, newV):
    ini_index = 0
    end_index = len(aList)
    index = int((end_index-ini_index)/2)
    for i in range(0, len(aList), 1):
        if newV[1] < aList[index][1]:
            if end_index-ini_index <= 1 or index + 1 == end_index:
                index = index + 1
                break
            else:
                ini_index = index + 1
                index = int(index + (end_index-ini_index)/2 + 1)
        elif newV[1] > aList[index][1]:
            if end_index-ini_index <= 1 or index == ini_index:
                index = index # Place before the current position
                break
            else:
                end_index = index
                index = int(index - (end_index-ini_index)/2)
        else:
            index = ini_index
            break
    aList.insert(index, newV)


@numba.njit()
def eikonal_one_ecg_part1(params, rootNodesIndexes):
    endoSpeed = params[nlhsParam-1]
    x = params[nlhsParam:]
    y=np.empty_like(x)
    rootNodesParam = np.round_(x, 0, y)
    y=None
    rootNodesIndexes = rootNodesIndexes[rootNodesParam==1]
    
    # Compute the cost of all endocardial edges
    navigationCosts = np.empty((edges.shape[0]))
    for index in range(edges.shape[0]):
        if isEndocardial[index]:
            # Cost for the propagation in the endocardium
            navigationCosts[index] = math.sqrt(np.dot(edgeVEC[index, :], edgeVEC[index, :])) / endoSpeed
    
    # Current Speed Configuration
    g = np.zeros((3, 3), np.float64) # g matrix
    np.fill_diagonal(g, [params[0]**2, params[1]**2, params[2]**2], wrap=False)# Needs to square each value

    for index in range(edges.shape[0]):
        if not isEndocardial[index]:
            # Cost equation for the Eikonal model + Fibrosis at the end
            aux1 = np.dot(g, tetraFibers[index, :, :].T)
            aux2 = np.dot(tetraFibers[index, :, :], aux1)
            aux3 = np.linalg.inv(aux2)
            aux4 = np.dot(edgeVEC[index, :], aux3)
            aux5 = np.dot(aux4, edgeVEC[index:index+1, :].T)
            navigationCosts[index] = np.sqrt(aux5)[0]
    # Build adjacentcy costs
    adjacentCost = numba.typed.List()
    for i in range(0, nodesXYZ.shape[0], 1):
        not_nan_neighbours = neighbours[i][neighbours[i]!=nan_value]
        adjacentCost.append(np.concatenate((unfoldedEdges[not_nan_neighbours][:, 1:2], np.expand_dims(navigationCosts[not_nan_neighbours%edges.shape[0]], -1)), axis=1))
    return adjacentCost, rootNodesIndexes


@numba.njit()
def np_apply_along_axis(func1d, axis, arr):
    assert arr.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        result = np.empty(arr.shape[1])
        for i in range(len(result)):
            result[i] = func1d(arr[:, i])
    else:
        result = np.empty(arr.shape[0])
        for i in range(len(result)):
            result[i] = func1d(arr[i, :])
    return result


@numba.njit()
def np_mean(array, axis):
    return np_apply_along_axis(np.mean, axis, array)


@numba.njit()
def np_std(array, axis):
    return np_apply_along_axis(np.std, axis, array)


def eikonal_ecg(population, rootNodesIndexes):
    # print(population.shape)
    # print(population)
    if population.shape[1] <= nlhsParam:
        population = np.concatenate((population, np.ones((population.shape[0], rootNodesIndexes.shape[0]))), axis=1)

    population_unique, unique_indexes = np.unique(population, return_inverse=True, axis=0)
    
    if experiment_output == 'ecg':
        max_len = 256
        prediction_list = pymp.shared.array((population_unique.shape[0], nb_leads, max_len), dtype=np.float64)
        prediction_list[:, :, :] = np.nan
    else:
        prediction_list = pymp.shared.array((population_unique.shape[0], nodesXYZ.shape[0]), dtype=np.float64)
        prediction_list[:, :] = np.nan
    with pymp.Parallel(min(threadsNum, prediction_list.shape[0])) as p1:
        for conf_i in p1.range(prediction_list.shape[0]):
            params = population_unique[conf_i, :]
            adjacentCost, eiknoal_rootNodes = eikonal_one_ecg_part1(params, rootNodesIndexes)

            ## Initialise variables
            activationTimes = np.zeros((nodesXYZ.shape[0],), np.float64)
            visitedNodes = np.zeros((nodesXYZ.shape[0],), dtype=np.bool_)
            cummCost = 1. # WARNING!! ROOT NODES HAVE A TIME OF 1 ms
            tempTimes = np.zeros((nodesXYZ.shape[0],), np.float64) + 1000

            ## Run the code for the root nodes
            visitedNodes[eiknoal_rootNodes] = True
            activationTimes[eiknoal_rootNodes] = cummCost
            nextNodes = (np.vstack([adjacentCost[rootNode] + np.array([0, cummCost]) for rootNode in eiknoal_rootNodes])).tolist()

            activeNode_i = eiknoal_rootNodes[0]
            sortSecond = lambda x : x[1]
            nextNodes.sort(key=sortSecond, reverse=True)

            while visitedNodes[activeNode_i]:
                nextEdge = nextNodes.pop()
                activeNode_i = int(nextEdge[0])
            cummCost = nextEdge[1]
            if nextNodes: # Check if the list is empty, which can happen while everything being Ok
                tempTimes[(np.array(nextNodes)[:, 0]).astype(np.int32)] = np.array(nextNodes)[:, 1]

            ## Run the whole algorithm
            for i in range(0, len(activationTimes)-(len(eiknoal_rootNodes)), 1):
                visitedNodes[activeNode_i] = True
                activationTimes[activeNode_i] = cummCost
                adjacents = (adjacentCost[activeNode_i] + np.array([0, cummCost])).tolist()
                for adjacent_i in range(0, len(adjacents), 1):
                    if (not visitedNodes[int(adjacents[adjacent_i][0])]
                    and (tempTimes[int(adjacents[adjacent_i][0])] >
                    adjacents[adjacent_i][1])):
                        insertSorted(nextNodes, adjacents[adjacent_i])
                        tempTimes[int(adjacents[adjacent_i][0])] = adjacents[adjacent_i][1]
                while visitedNodes[activeNode_i] and len(nextNodes) > 0:
                    nextEdge = nextNodes.pop()
                    activeNode_i = int(nextEdge[0])
                cummCost = nextEdge[1]

            # Clean Memory
            adjacentCost = None # Clear Mem
            visitedNodes = None # Clear Mem
            tempTimes = None # Clear Mem
            nextNodes = None # Clear Mem
            tempVisited = None # Clear Mem
            navigationCosts = None # Clear Mem

            activationTimes = np.round(activationTimes).astype(np.int32)

            if experiment_output == 'ecg':
                # Start ECG section ---------------
                nb_timesteps = min(max_len, np.max(activationTimes) + 1) # 1000 Hz is one evaluation every 1 ms
                ECG_aux = np.full((nb_leads, nb_timesteps), np.nan, dtype=np.float64)

                # Calculate voltage per timestep
                Vm = np.zeros((nb_timesteps, nodesXYZ.shape[0])) #+ innactivated_Vm
                for t in range(0, nb_timesteps, 1): # 1000 Hz is one evaluation every 1 ms
                    Vm[t:, activationTimes == t] = 1

                BSP = np.zeros((nb_bsp, nb_timesteps), dtype=np.float64)
                eleContrib = np.zeros((nb_bsp, tetrahedrons.shape[0]), dtype=np.float64)
                for timestep_i in range(1, nb_timesteps, 1):
                    activeNodes = np.nonzero(activationTimes==timestep_i)[0].astype(np.int32)
                    if not len(activeNodes) == 0:
                        activeEle = np.unique(
                            np.concatenate([elements[nodeID] for nodeID in activeNodes]))

                        bMVm = (Vm[timestep_i, tetrahedrons[activeEle, 0:3]]
                            - Vm[timestep_i, tetrahedrons[activeEle, 3]][:, np.newaxis])
                        bd_Vm = np.squeeze(
                        np.matmul(G_pseudo[activeEle, :, :], bMVm[:, :, np.newaxis]), axis=2)
                        eleContrib[:, activeEle] = np.sum(d_r[:, activeEle, :]*bd_Vm, axis=2)
                        BSP[:, timestep_i] = np.sum(eleContrib, axis=1)
                    else:
                        BSP[:, timestep_i] = BSP[:, timestep_i-1]

                # Clear Memory
                activationResults = None
                Vm = None
                eleContrib = None

                # Make 12-lead ECG
                ECG_aux[0, :] = (BSP[0, :] - BSP[1, :])
                ECG_aux[1, :] = (BSP[2, :] - BSP[1, :])
                BSPecg = BSP - np_mean(BSP[0:2, :], axis=0)
                BSP = None # Clear Memory
                ECG_aux[2:nb_leads, :] = BSPecg[4:nb_bsp, :]
                ECG_aux = signal.filtfilt(b_filtfilt, a_filtfilt, ECG_aux) # Filter ECG signal

                ECG_aux = ECG_aux - np_mean(ECG_aux, axis=1)[:, np.newaxis]
                ECG_aux = ECG_aux / np_std(ECG_aux, axis=1)[:, np.newaxis]

                prediction_list[conf_i, :, :ECG_aux.shape[1]] = ECG_aux - ECG_aux[:, 0:1] # align at zero
            else:
                prediction_list[conf_i, :] = activationTimes
    #print(np.sum(np.isnan(prediction_list)))
    return prediction_list[unique_indexes]


def trianglorgram(n_timestamps_1, n_timestamps_2, max_slope=2.):
    
    # Compute the slopes of the parallelogram bounds
    max_slope_ = max_slope
    min_slope_ = 1 / max_slope_
    scale_max = (n_timestamps_2 - 1) / (n_timestamps_1 - 2) # ORIGINAL
    max_slope_ *= scale_max
    # We take out this line because we want to consider values around the new diagonal, rather than the true diagonal
#     max_slope_ = max(1., max_slope_) # ORIGINAL, this would include the true diagonal if not included already

    scale_min = (n_timestamps_2 - 2) / (n_timestamps_1 - 1) # ORIGINAL
    min_slope_ *= scale_min
    # We take out this line because we want to consider values around the new diagonal, rather than the true diagonal
#     min_slope_ = min(1., min_slope_) # ORIGINAL, this would include the true diagonal if not included already
    
    # Little fix for max_slope = 1
    if max_slope == 1:
        # Now we create the piecewise linear functions defining the parallelogram
        centered_scale = np.arange(n_timestamps_1) - n_timestamps_1 + 1
        lower_bound = np.empty((2, n_timestamps_1))
        lower_bound[0] = min_slope_ * np.arange(n_timestamps_1)
        lower_bound[1] = max_slope_ * centered_scale + n_timestamps_2 - 1

        # take the max of the lower linear funcs
        lower_bound = np.round(lower_bound, 2)
        lower_bound = np.ceil(np.max(lower_bound, axis=0))

        upper_bound = np.empty((2, n_timestamps_1))
        upper_bound[0] = max_slope_ * np.arange(n_timestamps_1) + 1
        upper_bound[1] = min_slope_ * centered_scale + n_timestamps_2

        # take the min of the upper linear funcs
        upper_bound = np.round(upper_bound, 2)
        upper_bound = np.floor(np.min(upper_bound, axis=0))
        
        # This part makes that sometimes dtw(ecg1, ecg2) != dtw(ecg2, ecg1) for ecgs from DTI003, thus should be revised in the future
        if n_timestamps_2 > n_timestamps_1:
            upper_bound[:-1] = lower_bound[1:]
        else:
            upper_bound = lower_bound + 1
    else:
        # Now we create the piecewise linear functions defining the parallelogram
        centered_scale = np.arange(n_timestamps_1) - n_timestamps_1 + 1
        lower_bound = min_slope_ * np.arange(n_timestamps_1)

        # take the max of the lower linear funcs
        lower_bound = np.round(lower_bound, 2)
#         lower_bound = np.ceil(lower_bound) # ORIGINAL
        lower_bound = np.floor(lower_bound) # Enforces that at least one pixel is available when we take out the restriction that the true diagonal should be always available to the wraping path

        upper_bound = max_slope_ * np.arange(n_timestamps_1) + 1

        # take the min of the upper linear funcs
        upper_bound = np.round(upper_bound, 2)
#         upper_bound = np.floor(upper_bound) # ORIGINAL
        upper_bound = np.ceil(upper_bound) # Enforces that at least one pixel is available when we take out the restriction that the true diagonal should be always available to the wraping path

    region = np.asarray([lower_bound, upper_bound]).astype('int64')
    region = np.clip(region[:, :n_timestamps_1], 0, n_timestamps_2) # Project region on the feasible set
    return region


def dtw_trianglorgram(x, y, window=None, time_penalty=None, max_slope=1.5, w_max=10.):

    n_timestamps_1 = x.shape[0]
    n_timestamps_2 = y.shape[0]

    small_c = 0.05 * 171 / meshVolume
    
    # Computes the region (in-window area using a trianglogram)
    region = trianglorgram(n_timestamps_1, n_timestamps_2, max_slope)
    
    # Computes cost matrix from dtw input
    dist_ = lambda x, y : (x - y) ** 2

    region = check_array(region, dtype='int64')
    region_shape = region.shape
    if region_shape != (2, x.size):
        raise ValueError(
            "The shape of 'region' must be equal to (2, n_timestamps_1) "
            "(got {0}).".format(region_shape)
        )

    # Computs the cost matrix considering the window (0 inside, np.inf outside)
    cost_mat = np.full((n_timestamps_1, n_timestamps_2), np.inf)
    m = np.amax(cost_mat.shape)
    for i in numba.prange(n_timestamps_1):
        for j in numba.prange(region[0, i], region[1, i]):
            cost_mat[i, j] = dist_(x[i], y[j]) * (w_max * abs(i-j)/max(1., (i+j))+1.) # This new weight considers that wraping in time is cheaper the later it's done #* (w_max/(1+math.exp(-g*(abs(i-j)-m/2)))+1.) # + abs(i-j)*small_c # Weighted version of the DTW algorithm

    cost_mat = check_array(cost_mat, ensure_min_samples=2,
                           ensure_min_features=2, ensure_2d=True,
                           force_all_finite=False, dtype='float64')
    
    # Computes the accumulated cost matrix
    acc_cost_mat = np.ones((n_timestamps_1, n_timestamps_2)) * np.inf
    acc_cost_mat[0, 0: region[1, 0]] = np.cumsum(
        cost_mat[0, 0: region[1, 0]]
    )
    acc_cost_mat[0: region[1, 0], 0] = np.cumsum(
        cost_mat[0: region[1, 0], 0]
    )
    region_ = np.copy(region)

    region_[0] = np.maximum(region_[0], 1)
    ant_acc_min_i = -1
    acc_count = 0
    for i in range(1, n_timestamps_1):
        for j in range(region_[0, i], region_[1, i]):
            # Implementation of a Slope-constraint as a step-pattern:
            # This constraint will enforce that the algorithm can only take up to 2 consecutive steps along the time wraping directions.
            # I decided to make it symetric because in (Sakoe and Chiba, 1978) they state that symetric means that DTW(A, B) == DTW(B, A), although I am not convinced why it's not the case in the asymetric implementation.
            # Besides, the asymetric case has a bias towards the diagonal which I thought could be desirable in our case, that said, having DTW(A, B) == DTW(B, A) may prove even more important, especially in further
            # applications of this algorithm for ECG comparison.
            # This implementation is further explained in (Sakoe and Chiba, 1978) and correspondes to the one with P = 0.5, (P = n/m, where P is a rule being inforced, n is the number of steps in the diagonal
            # direction and m is the steps in the time wraping direction).
            acc_cost_mat[i, j] = min(
                acc_cost_mat[i - 1, j-3] + 2*cost_mat[i, j-2] + cost_mat[i, j-1] + cost_mat[i, j],
                acc_cost_mat[i - 1, j-2] + 2*cost_mat[i, j-1] + cost_mat[i, j],
                acc_cost_mat[i - 1, j - 1] + 2*cost_mat[i, j],
                acc_cost_mat[i - 2, j-1] + 2*cost_mat[i-1, j] + cost_mat[i, j],
                acc_cost_mat[i - 3, j-1] + 2*cost_mat[i-2, j] + cost_mat[i-1, j] + cost_mat[i, j]
            )
            

    
    dtw_dist = acc_cost_mat[-1, -1]/(n_timestamps_1 + n_timestamps_2) # Normalisation M+N according to (Sakoe and Chiba, 1978)
    
    dtw_dist = math.sqrt(dtw_dist)
    
    # Penalty for ECG-width differences
    ecg_width_cost = small_c * (n_timestamps_1-n_timestamps_2)**2 / min(n_timestamps_1,n_timestamps_2)
    
#     return (dtw_dist+ecg_width_cost, dtw_dist, cost_mat/(n_timestamps_1+n_timestamps_2), acc_cost_mat/(n_timestamps_1+n_timestamps_2), path)
    return dtw_dist, ecg_width_cost

    
def main(args):
    import multiprocessing
    run_iter = int(args[0])
    print(run_iter)
    target_type = args[1]
    target_snr_db = 20
    rootNodeResolution = "high"
    file_path_tag = "path"
    metric = "dtw"
    print(file_path_tag)
    load_target = False
    consistency_count = 3
    meshName_list = ["DTI001", "DTI024", "DTI004", "DTI003"]
    conduction_speeds_list = [
                [50, 32, 29, 150],
                [50, 32, 29, 120],
                [50, 32, 29, 179],
                [88, 49, 45, 179],
                [88, 49, 45, 120]
    ]
    npart = 512
    threadsNum = multiprocessing.cpu_count()
    keep_fraction =  (npart - threadsNum) / npart
    run_count = 0
    conduction_speeds = None
    consistency_iter = None
    for consistency in range(consistency_count):
        for conduction_speeds_aux in conduction_speeds_list:
            for meshName_val_aux in meshName_list:
                if run_count == run_iter:
                    conduction_speeds = conduction_speeds_aux
                    consistency_iter = consistency
                    meshName_val = meshName_val_aux
                run_count = run_count + 1
    if conduction_speeds is not None:
        # Volumes of the meshes
        if meshName_val == "DTI001":
            meshVolume_val = 74
        elif meshName_val == "DTI024":
            meshVolume_val = 76
        elif meshName_val == "DTI004":
            meshVolume_val = 107
        elif meshName_val == "DTI003":
            meshVolume_val = 171
        elif meshName_val == "DTI032":
            meshVolume_val = 139
        finalResultsPath = 'data/'+file_path_tag+'/'
        tmpResultsPath = 'data/'+file_path_tag+'/'
        population_fileName = (meshName_val + '_' + rootNodeResolution + '_' + str(conduction_speeds) + '_'
                    + rootNodeResolution + '_' + target_type + '_' + metric + '_' +
                     str(consistency_iter) + '_population.csv')
        if (not os.path.isfile(finalResultsPath + population_fileName)
            and (not os.path.isfile(tmpResultsPath + population_fileName))):
            target_fileName = None
            if (not load_target) or os.path.isfile(target_fileName):
                f = open(tmpResultsPath + population_fileName, "w")
                f.write("Experiment in progress")
                f.close()
                print(population_fileName)
                run_inference(
                    meshName_val=meshName_val,
                    meshVolume_val=meshVolume_val,
                    conduction_speeds=conduction_speeds,
                    final_path=finalResultsPath + population_fileName,
                    tmp_path=tmpResultsPath + population_fileName,
                    threadsNum_val=threadsNum,
                    target_type=target_type,
                    metric=metric,
                    npart=npart,
                    keep_fraction=keep_fraction,
                    rootNodeResolution=rootNodeResolution,
                    target_snr_db=target_snr_db,
                    load_target=load_target,
                    target_fileName=target_fileName
                )
            else:
                print('No target file: ' +target_fileName)
        else:
            print('Done by someone else')
    else:
        print('Nothing to be done')

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
