################################################################################
# Written by Joakim Halldin Stenlid, SUNCAT SLAC/Stanford Univeristy 2020-2021
# for paper:
#   'Assessing Catalytic Rates of Bimetallic Nanoparticles with Active Site Specificity - A
#    Case Study using NO Decomposition'
#    Authors: Joakim Halldin Stenlid, Verena Streibel, Tej Choksi, Frank Abild-Pedersen
#    currently under peer review (Oct 2021)
#
# Dependables: Atomic simulation evironment (ASE) and ASAP3
#
# Syntax: python3 str2alpha+BE.py [structure-file]
#
# OUTPUT: (Site, A_pt, A_ptm, A_mpt, BE) # for every site beloning to 7-7, 8-8, 7-9-9, 9-9-9 or C-7
#                                        # with A_x being the delta CN vector for pt in pt
#                                        # pt in m=alloying metal, and m in pt
#
################################################################################

from ase import Atoms
from ase import geometry
from ase.build import bulk, make_supercell
from ase.io import read, write
import numpy as np
import sys, math
from ase import neighborlist
from asap3 import *

### USER INPUTS ###
infile=sys.argv[1]          # Name of structure input file
eq_dist=2.772               # Metal equilibrium distance in favored coordination environment
tol=0.1                     # Max tolerated bond strain. Note: strain model not tested beyond 8%
eq_dist_tol=eq_dist*(1+tol) # Max bond distance to be considered a bond
M = 195.09                  # Molar mass Pt
NA = 6.022e23               # Avogadoros constant
clustertype = 'CUBO'        # NP cluster type - affects how NP area and volume are computed
include_corners = True      # Do you want to count C-7 corner sites as 7-7?
alloy = 'Au'                # If alloy, add guest atom here, else use 'none' (Pt always host)

### DO NOT CHANGE FROM HERE ####

#alpha parameters 
Pt_Pt = [-3.414, -0.228, -0.206, -0.157, -0.190, -0.116, -0.211, -0.111, -0.088, -0.056]
Pt_Pd = [-3.146, -0.317, -0.190, -0.188, -0.236, -0.154, -0.219, -0.166, -0.111, -0.056]
Pt_Au = [-3.144, -0.303, -0.135, -0.214, -0.175, -0.128, -0.230, -0.157, -0.125, -0.064]
Pt_Ag = [-3.071, -0.322, -0.124, -0.206, -0.221, -0.148, -0.224, -0.188, -0.139, -0.062]
Pt_Cu = [-3.286, -0.315, -0.184, -0.165, -0.239, -0.145, -0.206, -0.173, -0.106, -0.052]
Pd_Pt = [-2.049, -0.170, -0.103, -0.131, -0.130, -0.124, -0.128, -0.095, -0.114, -0.103]
Au_Pt = [-1.827, -0.099, -0.138, -0.099, -0.082, -0.065, -0.072, -0.016, -0.005, -0.011]
Ag_Pt = [-1.421, -0.081, -0.080, -0.093, -0.057, -0.077, -0.064, -0.035, -0.047, -0.045]
Cu_Pt = [-2.051, -0.148, -0.085, -0.153, -0.102, -0.096, -0.148, -0.142, -0.163, -0.106]

if alloy == 'none':
    Pt_alpha = Pt_Pt
    guest_alpha = Pt_Pt
elif alloy == 'Pd':
    Pt_alpha = Pt_Pd
    guest_alpha = Pd_Pt
elif alloy == 'Au':
    Pt_alpha = Pt_Au
    guest_alpha = Au_Pt
elif alloy == 'Ag':
    Pt_alpha = Pt_Ag
    guest_alpha = Ag_Pt
elif alloy == 'Cu':
    Pt_alpha = Pt_Cu
    guest_alpha = Cu_Pt

### Functions for analysis of A vectors (delta coordination vectors)

def alpha_bridge(nblist,site1,site2,elements): #(nb, nb_list1, index, nb1, cn_site, cn_nb):

    #get neighbors and CNs
    nb_list1 = nblist.get_neighbors(site1)[0]
    nb_list2 = nblist.get_neighbors(site2)[0]
    cn_site1 = len(nb_list1)
    cn_site2 = len(nb_list2)
    cn_nb1 = []
    for ii in range(cn_site1):
        neighbor = nb_list1[ii]
        cn_neighbor = len(nblist.get_neighbors(neighbor)[0])
        cn_nb1.append(cn_neighbor)
    cn_nb2 = []
    for ii in range(cn_site2):
        neighbor = nb_list2[ii]
        cn_neighbor = len(nblist.get_neighbors(neighbor)[0])
        cn_nb2.append(cn_neighbor)

    #add alphas of central atoms of the site
    alpha_pt = np.zeros(12,dtype='int')
    alpha_pt_g = np.zeros(12,dtype='int')
    alpha_guest = np.zeros(12,dtype='int')
    if list( map(elements.__getitem__,nb_list1)).count(alloy) > 0:
        alpha_pt_g[np.arange(cn_site1)] = alpha_pt_g[np.arange(cn_site1)] + 1
    else:
        alpha_pt[np.arange(cn_site1)] = alpha_pt[np.arange(cn_site1)] + 1
    if list( map(elements.__getitem__,nb_list2)).count(alloy) > 0:
        alpha_pt_g[np.arange(cn_site2-1)] = alpha_pt_g[np.arange(cn_site2-1)] + 1
    else:
        alpha_pt[np.arange(cn_site2-1)] = alpha_pt[np.arange(cn_site2-1)] + 1

    #add alphas for neighbors of first bridge site atom
    for iii in range(len(cn_nb1)):
        nb_nb_list = nblist.get_neighbors(nb_list1[iii])[0] 
        if elements[nb_list1[iii]] == 'Pt':
            if list( map(elements.__getitem__,nb_nb_list)).count(alloy) > 0:
                alpha_pt_g[cn_nb1[iii]-1] = alpha_pt_g[cn_nb1[iii]-1] + 1
            else:
                alpha_pt[cn_nb1[iii]-1] = alpha_pt[cn_nb1[iii]-1] + 1
        else:
            alpha_guest[cn_nb1[iii]-1] = alpha_guest[cn_nb1[iii]-1] + 1

    #add alphas for neighbors of second bridge site atom
    site1_index = np.where(np.asarray(nb_list2) == site1)[0]
    upd_cn_nb2 = np.delete(cn_nb2, site1_index)
    upd_nb_list2 = np.delete(nb_list2, site1_index)
    for iii in range(len(upd_cn_nb2)):
        nb_nb_list = nblist.get_neighbors(upd_nb_list2[iii])[0]
        if any(site == upd_nb_list2[iii] for site in nb_list1) == True :
            if elements[upd_nb_list2[iii]] == 'Pt':
                if list( map(elements.__getitem__,nb_nb_list)).count(alloy) > 0:
                    alpha_pt_g[upd_cn_nb2[iii]-2] = alpha_pt_g[upd_cn_nb2[iii]-2] + 1
                else:
                    alpha_pt[upd_cn_nb2[iii]-2] = alpha_pt[upd_cn_nb2[iii]-2] + 1
            else:
                alpha_guest[upd_cn_nb2[iii]-2] = alpha_guest[upd_cn_nb2[iii]-2] + 1
        else:
            if elements[upd_nb_list2[iii]] == 'Pt':
                if list( map(elements.__getitem__,nb_nb_list)).count(alloy) > 0:
                    alpha_pt_g[upd_cn_nb2[iii]-1] = alpha_pt_g[upd_cn_nb2[iii]-1] + 1
                else:
                    alpha_pt[upd_cn_nb2[iii]-1] = alpha_pt[upd_cn_nb2[iii]-1] + 1
            else:
                alpha_guest[upd_cn_nb2[iii]-1] = alpha_guest[upd_cn_nb2[iii]-1] + 1

    return np.delete(alpha_pt,[1,2]), np.delete(alpha_pt_g,[1,2]), np.delete(alpha_guest,[1,2])

def alpha_hollow(nblist,site1,site2,site3,elements): #(nb, nb_list1, index, nb1, cn_site, cn_nb):

    #get neighbors and CNs
    nb_list1 = nblist.get_neighbors(site1)[0]
    nb_list2 = nblist.get_neighbors(site2)[0]
    nb_list3 = nblist.get_neighbors(site3)[0]
    cn_site1 = len(nb_list1)
    cn_site2 = len(nb_list2)
    cn_site3 = len(nb_list3)
    cn_nb1 = []
    for ii in range(cn_site1):
        neighbor = nb_list1[ii]
        cn_neighbor = len(nblist.get_neighbors(neighbor)[0])
        cn_nb1.append(cn_neighbor)
    cn_nb2 = []
    for ii in range(cn_site2):
        neighbor = nb_list2[ii]
        cn_neighbor = len(nblist.get_neighbors(neighbor)[0])
        cn_nb2.append(cn_neighbor)
    cn_nb3 = []
    for ii in range(cn_site3):
        neighbor = nb_list3[ii]
        cn_neighbor = len(nblist.get_neighbors(neighbor)[0])
        cn_nb3.append(cn_neighbor)    

    #add alphas of central atoms of the site
    alpha_pt = np.zeros(12,dtype='int')
    alpha_pt_g = np.zeros(12,dtype='int')
    alpha_guest = np.zeros(12,dtype='int')
    if list( map(elements.__getitem__,nb_list1)).count(alloy) > 0:
        alpha_pt_g[np.arange(cn_site1)] = alpha_pt_g[np.arange(cn_site1)] + 1
    else:
        alpha_pt[np.arange(cn_site1)] = alpha_pt[np.arange(cn_site1)] + 1
    if list( map(elements.__getitem__,nb_list2)).count(alloy) > 0:
        alpha_pt_g[np.arange(cn_site2-1)] = alpha_pt_g[np.arange(cn_site2-1)] + 1
    else:
        alpha_pt[np.arange(cn_site2-1)] = alpha_pt[np.arange(cn_site2-1)] + 1
    if list( map(elements.__getitem__,nb_list3)).count(alloy) > 0:
        alpha_pt_g[np.arange(cn_site3-2)] = alpha_pt_g[np.arange(cn_site3-2)] + 1
    else:
        alpha_pt[np.arange(cn_site3-2)] = alpha_pt[np.arange(cn_site3-2)] + 1

    #add alphas for neighbors of first hollow site atom
    for iii in range(len(cn_nb1)):
        nb_nb_list = nblist.get_neighbors(nb_list1[iii])[0]
        if elements[nb_list1[iii]] == 'Pt':
            if list( map(elements.__getitem__,nb_nb_list)).count(alloy) > 0:
                alpha_pt_g[cn_nb1[iii]-1] = alpha_pt_g[cn_nb1[iii]-1] + 1
            else:
                alpha_pt[cn_nb1[iii]-1] = alpha_pt[cn_nb1[iii]-1] + 1
        else:
            alpha_guest[cn_nb1[iii]-1] = alpha_guest[cn_nb1[iii]-1] + 1

    #add alphas for neighbors of second hollow site atom
    site1_index = np.where(np.asarray(nb_list2) == site1)[0]
    upd_cn_nb2 = np.delete(cn_nb2, site1_index)
    upd_nb_list2 = np.delete(nb_list2, site1_index)
    for iii in range(len(upd_cn_nb2)):
        nb_nb_list = nblist.get_neighbors(upd_nb_list2[iii])[0]
        if any(site == upd_nb_list2[iii] for site in nb_list1) == True :
            if elements[upd_nb_list2[iii]] == 'Pt':
                if list( map(elements.__getitem__,nb_nb_list)).count(alloy) > 0:
                    alpha_pt_g[upd_cn_nb2[iii]-2] = alpha_pt_g[upd_cn_nb2[iii]-2] + 1
                else:
                    alpha_pt[upd_cn_nb2[iii]-2] = alpha_pt[upd_cn_nb2[iii]-2] + 1
            else:
                alpha_guest[upd_cn_nb2[iii]-2] = alpha_guest[upd_cn_nb2[iii]-2] + 1
        else:
            if elements[upd_nb_list2[iii]] == 'Pt':
                if list( map(elements.__getitem__,nb_nb_list)).count(alloy) > 0:
                    alpha_pt_g[upd_cn_nb2[iii]-1] = alpha_pt_g[upd_cn_nb2[iii]-1] + 1
                else:
                    alpha_pt[upd_cn_nb2[iii]-1] = alpha_pt[upd_cn_nb2[iii]-1] + 1
            else:
                alpha_guest[upd_cn_nb2[iii]-1] = alpha_guest[upd_cn_nb2[iii]-1] + 1

    #add alphas for neighbors of third bridge site atom
    site1_index = np.where(np.asarray(nb_list3) == site1)[0]
    site2_index = np.where(np.asarray(nb_list3) == site2)[0]
    upd_cn_nb3 = np.delete(cn_nb3, [site1_index, site2_index])
    upd_nb_list3 = np.delete(nb_list3, [site1_index, site2_index])
    for iii in range(len(upd_cn_nb3)):
        nb_nb_list = nblist.get_neighbors(upd_nb_list3[iii])[0]
        if any(site == upd_nb_list3[iii] for site in nb_list1) == True :
            if any(site == upd_nb_list3[iii] for site in upd_nb_list2) == True :
                if elements[upd_nb_list3[iii]] == 'Pt':
                    if list( map(elements.__getitem__,nb_nb_list)).count(alloy) > 0:
                        alpha_pt_g[upd_cn_nb3[iii]-3] = alpha_pt_g[upd_cn_nb3[iii]-3] + 1
                    else:
                        alpha_pt[upd_cn_nb3[iii]-3] = alpha_pt[upd_cn_nb3[iii]-3] + 1
                else:
                    alpha_guest[upd_cn_nb3[iii]-3] = alpha_guest[upd_cn_nb3[iii]-3] + 1
            else:
                if elements[upd_nb_list3[iii]] == 'Pt':
                    if list( map(elements.__getitem__,nb_nb_list)).count(alloy) > 0:
                        alpha_pt_g[upd_cn_nb3[iii]-2] = alpha_pt_g[upd_cn_nb3[iii]-2] + 1
                    else:
                        alpha_pt[upd_cn_nb3[iii]-2] = alpha_pt[upd_cn_nb3[iii]-2] + 1
                else:
                    alpha_guest[upd_cn_nb3[iii]-2] = alpha_guest[upd_cn_nb3[iii]-2] + 1
        elif any(site == upd_nb_list3[iii] for site in upd_nb_list2) == True :
            if elements[upd_nb_list3[iii]] == 'Pt':
                if list( map(elements.__getitem__,nb_nb_list)).count(alloy) > 0:
                    alpha_pt_g[upd_cn_nb3[iii]-2] = alpha_pt_g[upd_cn_nb3[iii]-2] + 1
                else:
                    alpha_pt[upd_cn_nb3[iii]-2] = alpha_pt[upd_cn_nb3[iii]-2] + 1
            else:
                alpha_guest[upd_cn_nb3[iii]-2] = alpha_guest[upd_cn_nb3[iii]-2] + 1
        else:
            if elements[upd_nb_list3[iii]] == 'Pt':
                if list( map(elements.__getitem__,nb_nb_list)).count(alloy) > 0:
                    alpha_pt_g[upd_cn_nb3[iii]-1] = alpha_pt_g[upd_cn_nb3[iii]-1] + 1
                else:
                    alpha_pt[upd_cn_nb3[iii]-1] = alpha_pt[upd_cn_nb3[iii]-1] + 1
            else:
                alpha_guest[upd_cn_nb3[iii]-1] = alpha_guest[upd_cn_nb3[iii]-1] + 1
                
    return np.delete(alpha_pt,[1,2]), np.delete(alpha_pt_g,[1,2]), np.delete(alpha_guest,[1,2])


###################################################################
################ HERE THE ANALYSIS STARTS! ########################
###################################################################

#### analyse surface structure ####
atoms=read(infile)
eq2=eq_dist**2              
N = atoms.get_number_of_atoms()
nblist = FullNeighborList(eq_dist_tol, atoms, driftfactor=0.05)
pos=atoms.get_positions()
cell=atoms.get_cell()
elements=atoms.get_chemical_symbols() 
count = 0
cat_sites_77 = []
cat_sites_C7 = []
cat_sites_799 = []
cat_sites_88 = []
cat_sites_999 = []
dmax=0

#specify corner sites
if include_corners == True:
    if clustertype == 'WULFF': # VARNING! Could vary with structure
        corner = 6
    elif clustertype == 'OCT':
        corner = 4
    elif clustertype == 'CUBO':
        corner = 5
#    elif clustertype == 'ICO': # Not well tested
#        corner = 6
#    elif clustertype == 'DECA': # Not well tested
#        corner = [7,6]
else:
    corner = 0

#read structure and loop over all atoms
for i in range(len(atoms)):
    nb, dvec, d2 = nblist.get_neighbors(i)
    #neigborlist, pairwise distance vector, and squared norm of distance vectors

    cn_site = len(nb)
    dists = atoms.get_distances(i, np.arange( len(atoms)),mic=False, vector=False)
    if np.max(dists) > dmax:
        dmax = np.max(dists)
    if cn_site < 7 and cn_site > 9: 
        continue
    else:
        count = count+1
    cn_nb = []
    for ii in range(cn_site):
        neighbor = nb[ii]
        cn_neighbor = len(nblist.get_neighbors(neighbor)[0])
        cn_nb.append(cn_neighbor)

### analysis of sites  ###
    
    sevens=len(np.where(np.asarray(cn_nb) == 7)[0])
    eights=len(np.where(np.asarray(cn_nb) == 8)[0])
    nines=len(np.where(np.asarray(cn_nb) == 9)[0])

    # 7-7 sites - not correct for very small NPs, but will be okay if corner sites are included
    if sevens == 2 and cn_site == 7 and elements[i] == 'Pt':
        pos_site = pos[i]
        nb1 = nb[np.where(np.asarray(cn_nb) == 7)[0][0]]
        nb2 = nb[np.where(np.asarray(cn_nb) == 7)[0][1]]
        if atoms.get_angle(nb1,i,nb2) < 170 :
            continue
        for n in range(sevens):
            nb1 = nb[np.where(np.asarray(cn_nb) == 7)[0][n]]
            if elements[nb1] != 'Pt':
                continue
            pos_nb1 = pos[nb1]
            site_cluster = Atoms("Pt2", positions = [pos_site, pos_nb1], cell=cell, pbc=[1,1,1])
            cat_site_pos = site_cluster.get_center_of_mass()
            
            if len(cat_sites_77) == 0:
                cat_sites_77.append(cat_site_pos)
                MPt, MPtG, MG = alpha_bridge(nblist, i, nb1, elements); print( '7-7', MPt, MPtG, MG, np.dot(Pt_Pt,MPt) + np.dot(Pt_alpha,MPtG) + np.dot(guest_alpha,MG))
            elif len(cat_sites_77) == 1:
                if np.sum(np.abs(cat_site_pos - cat_sites_77[0])) > 0.01:
                    cat_sites_77.append(cat_site_pos)
                    MPt, MPtG, MG = alpha_bridge(nblist, i, nb1, elements); print( '7-7', MPt, MPtG, MG, np.dot(Pt_Pt,MPt) + np.dot(Pt_alpha,MPtG) + np.dot(guest_alpha,MG))
            else:
                test = np.sqrt(np.sum(np.square(cat_sites_77 - cat_site_pos), axis=1))
                if any(p < float(0.01) for p in test) == False:
                    cat_sites_77.append(cat_site_pos)
                    MPt, MPtG, MG = alpha_bridge(nblist, i, nb1, elements); print( '7-7', MPt, MPtG, MG, np.dot(Pt_Pt,MPt) + np.dot(Pt_alpha,MPtG) + np.dot(guest_alpha,MG))
            
    # 8-8 sites  
    if eights >= 1 and cn_site == 8 and elements[i] == 'Pt':
        pos_site = pos[i]
        for n in range(eights):
            nb1 = nb[np.where(np.asarray(cn_nb) == 8)[0][n]]
            if elements[nb1] == 'Pt':
                pos_nb1 = pos[nb1]
                site_cluster = Atoms("Pt2", positions = [pos_site, pos_nb1], cell=cell, pbc=[1,1,1])
                cat_site_pos = site_cluster.get_center_of_mass()
                if len(cat_sites_88) == 0:
                    cat_sites_88.append(cat_site_pos)
                    MPt, MPtG, MG = alpha_bridge(nblist, i, nb1, elements); print( '8-8', MPt, MPtG, MG, np.dot(Pt_Pt,MPt) + np.dot(Pt_alpha,MPtG) + np.dot(guest_alpha,MG))
                elif len(cat_sites_88) == 1:
                    if np.sum(np.abs(cat_site_pos - cat_sites_88[0])) > 0.01:
                        cat_sites_88.append(cat_site_pos)
                        MPt, MPtG, MG = alpha_bridge(nblist, i, nb1, elements); print( '8-8', MPt, MPtG, MG, np.dot(Pt_Pt,MPt) + np.dot(Pt_alpha,MPtG) + np.dot(guest_alpha,MG))
                else:
                    test = np.sqrt(np.sum(np.square(cat_sites_88 - cat_site_pos), axis=1))
                    if any(p < float(0.01) for p in test) == False:
                        cat_sites_88.append(cat_site_pos)
                        MPt, MPtG, MG = alpha_bridge(nblist, i, nb1, elements); print( '8-8', MPt, MPtG, MG, np.dot(Pt_Pt,MPt) + np.dot(Pt_alpha,MPtG) + np.dot(guest_alpha,MG))
                        
    # 7-9-9 sites  
    if nines >= 2 and cn_site == 7 and elements[i] == 'Pt':
        pos_site = pos[i]
        for n in range(nines):
            nb1 = nb[np.where(np.asarray(cn_nb) == 9)[0][n]]
            for nn in range(n+1, nines):
                nb2 = nb[np.where(np.asarray(cn_nb) == 9)[0][nn]] 
                if atoms.get_distance(nb1, nb2) <= eq_dist_tol and elements[nb1] == 'Pt' and elements[nb2] == 'Pt': 
                    pos_nb1 = pos[nb1]
                    pos_nb2 = pos[nb2]
                    site_cluster = Atoms("Pt3", positions = [pos_site, pos_nb1, pos_nb2], cell=cell, pbc=[1,1,1])
                    cat_site_pos = site_cluster.get_center_of_mass()
                
                    if len(cat_sites_799) == 0:
                        cat_sites_799.append(cat_site_pos)
                        MPt, MPtG, MG = alpha_hollow(nblist, i, nb1, nb2, elements); print( '7-9-9', MPt, MPtG, MG, np.dot(Pt_Pt,MPt) + np.dot(Pt_alpha,MPtG) + np.dot(guest_alpha,MG))
                    elif len(cat_sites_799) == 1:
                        if np.sum(np.abs(cat_site_pos - cat_sites_799[0])) > 0.01:
                            cat_sites_799.append(cat_site_pos)
                            MPt, MPtG, MG = alpha_hollow(nblist, i, nb1, nb2, elements); print( '7-9-9', MPt, MPtG, MG, np.dot(Pt_Pt,MPt) + np.dot(Pt_alpha,MPtG) + np.dot(guest_alpha,MG))
                    else:
                        test = np.sqrt(np.sum(np.square(cat_sites_799 - cat_site_pos), axis=1))
                        if any(p < float(0.01) for p in test) == False:
                            cat_sites_799.append(cat_site_pos)
                            MPt, MPtG, MG = alpha_hollow(nblist, i, nb1, nb2, elements); print( '7-9-9', MPt, MPtG, MG, np.dot(Pt_Pt,MPt) + np.dot(Pt_alpha,MPtG) + np.dot(guest_alpha,MG))
         
    # 9-9-9 sites
    if nines >= 2 and cn_site == 9 and elements[i] == 'Pt':
        pos_site = pos[i]
        for n in range(nines):
            nb1 = nb[np.where(np.asarray(cn_nb) == 9)[0][n]]
            for nn in range(n+1, nines):
                nb2 = nb[np.where(np.asarray(cn_nb) == 9)[0][nn]] 
                if atoms.get_distance(nb1, nb2) <= eq_dist_tol and elements[nb1] == 'Pt' and elements[nb2] == 'Pt' :
                    pos_nb1 = pos[nb1]
                    pos_nb2 = pos[nb2]
                    site_cluster = Atoms("Pt3", positions = [pos_site, pos_nb1, pos_nb2], cell=cell, pbc=[1,1,1])
                    cat_site_pos = site_cluster.get_center_of_mass()
                    atoms_site = Atoms("".join(['O1','Pt9']),positions=np.vstack((cat_site_pos,pos[nb])), cell=cell, pbc=[1,1,1])
                    nblist_site = FullNeighborList(0.9*eq_dist, atoms_site, driftfactor=0.05)
                    hcp_fcc = len(nblist_site.get_neighbors(0)[0]) + 1
                    if hcp_fcc == 4:
                        continue
                    if len(cat_sites_999) == 0:
                        cat_sites_999.append(cat_site_pos)
                        MPt, MPtG, MG = alpha_hollow(nblist, i, nb1, nb2, elements); print( '9-9-9', MPt, MPtG, MG, np.dot(Pt_Pt,MPt) + np.dot(Pt_alpha,MPtG) + np.dot(guest_alpha,MG))
                    elif len(cat_sites_999) == 1:
                        if np.sum(np.abs(cat_site_pos - cat_sites_999[0])) > 0.01: 
                            cat_sites_999.append(cat_site_pos)
                            MPt, MPtG, MG = alpha_hollow(nblist, i, nb1, nb2, elements); print( '9-9-9', MPt, MPtG, MG, np.dot(Pt_Pt,MPt) + np.dot(Pt_alpha,MPtG) + np.dot(guest_alpha,MG))
                    else:
                        test = np.sqrt(np.sum(np.square(cat_sites_999 - cat_site_pos), axis=1))
                        if any(p < float(0.01) for p in test) == False:
                            cat_sites_999.append(cat_site_pos)
                            MPt, MPtG, MG = alpha_hollow(nblist, i, nb1, nb2, elements); print( '9-9-9', MPt, MPtG, MG, np.dot(Pt_Pt,MPt) + np.dot(Pt_alpha,MPtG) + np.dot(guest_alpha,MG))
                            
    # C-7 corner sites
    if sevens >= 1 and cn_site == corner and elements[i] == 'Pt':
        pos_site = pos[i]
        for n in range(sevens):
            nb1 = nb[np.where(np.asarray(cn_nb) == 7)[0][n]]
            pos_nb1 = pos[nb1]
            site_cluster = Atoms("Pt2", positions = [pos_site, pos_nb1], cell=cell, pbc=[1,1,1])
            cat_site_pos = site_cluster.get_center_of_mass()
            if len(cat_sites_C7) == 0:
                cat_sites_C7.append(cat_site_pos)
                MPt, MPtG, MG = alpha_bridge(nblist, i, nb1, elements); print( 'C-7', MPt, MPtG, MG, np.dot(Pt_Pt,MPt) + np.dot(Pt_alpha,MPtG) + np.dot(guest_alpha,MG))
            elif len(cat_sites_C7) == 1:
                if np.sum(np.abs(cat_site_pos - cat_sites_C7[0])) > 0.01:
                    cat_sites_C7.append(cat_site_pos)
                    MPt, MPtG, MG = alpha_bridge(nblist, i, nb1, elements); print( 'C-7', MPt, MPtG, MG, np.dot(Pt_Pt,MPt) + np.dot(Pt_alpha,MPtG) + np.dot(guest_alpha,MG))
            else:
                test = np.sqrt(np.sum(np.square(cat_sites_C7 - cat_site_pos), axis=1))
                if any(p < float(0.01) for p in test) == False:
                    cat_sites_C7.append(cat_site_pos)
                    MPt, MPtG, MG = alpha_bridge(nblist, i, nb1, elements); print( 'C-7', MPt, MPtG, MG, np.dot(Pt_Pt,MPt) + np.dot(Pt_alpha,MPtG) + np.dot(guest_alpha,MG))

                    
#################### OUTPUT #####################

NS_77 = len(cat_sites_77)
NS_88 = len(cat_sites_88)
NS_799 = len(cat_sites_799)          # currently both fcc and hcp sites included
NS_999 = len(cat_sites_999)          # currently only fcc sites included
NS_C7 = len(cat_sites_C7)

site_of_interest = 77                # for defining geometrical parameters of the particles

if site_of_interest == 77:
    NS = NS_77 + NS_C7
elif site_of_interest == 88:
    NS = NS_88
elif site_of_interest == 799:
    NS = NS_799
elif site_of_interest == 999:
    NS = NS_999
    
if clustertype == 'WULFF':
    a = dmax/2                        #radius of a sphere approximating size of the NP
    A = 4*math.pi*(a**2)              #Total Area of NP in A^2
    V = 4*math.pi*(a**3)/3            #Total volume of NP in A^3
    r = a                             #Radius of a sphere approximating the size of NP 

elif clustertype == 'OCT':            #note: the non-edge 7-7 sites are not included in N_77
    a = (NS_77/12 + 2)*eq_dist        #length edge in A
    A = 2*(3**0.5)*(a**2)             #Total Area of NP in A^2
    V = (2**0.5)*(a**3)/3             #Total volume of NP in A^3
    r = (V*3/(4*math.pi))**(0.333333) #Radius of a sphere with same volume as NP

elif clustertype == 'CUBO':           #note: the non-edge 7-7 sites are not included in N_77
    a = (NS_77/24 + 2)*eq_dist        #length edge in A
    A = (6 + 2*(3**0.5))*(a**2)       #Total Area of NP in A^2
    V = 5/3*(2**0.5)*(a**3)           #Total volume of NP in A^3
    r = (V*3/(4*math.pi))**(0.333333) # Radius of a sphere with same volume as NP

#print(N, NS, NS/A, NS/V, r, dmax, a) # uncomment if this information is needed
#print(NS_77+NS_C7, NS_88, NS_799, NS_999, A, V) # uncomment if this information is needed

