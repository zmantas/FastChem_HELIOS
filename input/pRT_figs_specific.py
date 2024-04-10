import matplotlib.colors as mcolors
from matplotlib.patches import Polygon
import struct
from scipy.interpolate import interp1d
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib
from petitRADTRANS import Radtrans
from petitRADTRANS import nat_cst as nc
import pickle
import os
import matplotlib.pyplot as plt
import sys
from astropy.io import ascii
import seaborn as sns
from matplotlib.ticker import StrMethodFormatter, NullFormatter
sns.set_theme()
sns.set_style("ticks")
import matplotlib.patheffects as pteffects
import concurrent.futures
import random
from matplotlib.patheffects import withStroke
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import csv
from matplotlib.colors import LogNorm
from matplotlib.ticker import ScalarFormatter, LogLocator



class Atmosphere(object): 
     
    def __init__(self,vul):        
        self.variab = vul['variable']
        self.atm = vul['atm']
        self.param = vul['parameter']
        self.P = self.atm['pco']
        self.y_ini = self.variab['y_ini']
        self.n_0 = self.atm['n_0']
        self.species = self.variab['species']       
        self.T = self.atm['Tco']
        self.VMR = self.variab['ymix']
        self.ms = self.atm['ms']    # Atomic weight of species
        self.MMW = self.atm['mu']

    def return_data(self):
        return[self.variab,self.atm,self.param]
         
    def massVMR(self):
        
        mass_frac = {}             
        for count,ele in enumerate(self.species):
            mass_frac[ele] = self.ms[count]*self.variab['y'][:,self.species.index(ele)]/np.sum(self.variab['y'],axis=1)/self.atm['mu']
            
            # Reversing for radtrans
            mass_frac[ele] = mass_frac[ele][::-1]     
            
            
            # Ion conversions
            if '_p' in ele or '_m' in ele:
                new_ele = ele[:-2] + '+' if '_p' in ele else ele[:-2] + '-'
                mass_frac[new_ele] = mass_frac[ele]
  
        mass_frac['e-'] = mass_frac['e']
        
        return mass_frac
    
 

def sort_output(vul_directory):
    
    listed_output = os.listdir(vul_directory)
    sorted_output = sorted(listed_output, key=lambda x: (x[:4], x))
    
    # Filter out runs without 'OP_' files
    sorted_output = [run for run in sorted_output if any(file.startswith('OP_') and file.endswith('.vul') for file in os.listdir(os.path.join(vul_directory, run)))]


    return sorted_output

def output():
        
    var, atm, param = [], [], []
    species, VMR, T = [], [], []
    P, plname, y_ini = [], [], []
    mass_frac = []
   
    sorted_output = sort_output(vul_directory)   
    
    for out in sorted_output:
        
        # Since some runs finish on different steps, checks the number of steps, that is the name of the file
        output_file = max((file for file in os.listdir(vul_directory + out) if file.startswith('OP_') and file.endswith('.vul')),
                          key=lambda file: int(file[3:-4]), default=None)

        vul = vul_directory + out + '/' + output_file

        # Unpack Pickle
        with open(vul, 'rb') as handle:
            vulcan = Atmosphere(pickle.load(handle))
            var.append(vulcan.return_data()[0])
            atm.append(vulcan.return_data()[1])
            param.append(vulcan.return_data()[2])
            species.append(var[-1]['species'])
            VMR.append(var[-1]['ymix'])
            #T.append(atm[-1]['Tco'])
            #P.append(atm[-1]['pco'])
            plname.append(out[:4])
            y_ini.append(var[-1]['y_ini'])
            mass_frac.append(vulcan.massVMR())

    return var, atm, param, species, VMR, plname, mass_frac, y_ini

def create_directory_if_not_exists(directory):
    os.makedirs(directory, exist_ok=True)

def save_data(file_path, header, data):
    with open(file_path, 'w+') as f:
        f.write(header)
        for n, wl in enumerate(data['wavelength']):
            f.write(f"{nc.c/wl/1e-4:<12}\t{data['flux'][n]*1e6:<12}\n")

def save_data_TP(file_path, header, data):
    with open(file_path, 'w+') as f:
        f.write(header)
        for n, wl in enumerate(data['temperature']):
            f.write(f"{wl:<12}\t{data['pressure'][n]:<12}\n")

def pt_empty(P,T,abundances,g,MMW):
    pt_empty_atm = Radtrans(line_species = ['O'],
                     wlen_bords_micron = [wlLow, wlMax],
                     do_scat_emis = False)
    pt_empty_atm.setup_opa_structure(P)
    pt_empty_atm.calc_flux(T, abundances, g, MMW, contribution=False)
    return pt_empty_atm

def pt_sp(P,T,abundances,g,MMW,sp):
    
    ray_sp = []
    if sp in ray_scatter:
        ray_sp.append(sp)
    pt_sp_atm = Radtrans(line_species = [sp],
                     wlen_bords_micron = [wlLow, wlMax],
                     rayleigh_species=ray_sp,
                     do_scat_emis = True)
    pt_sp_atm.setup_opa_structure(P)
    pt_sp_atm.calc_flux(T, abundances, g, MMW, contribution=False)
    return pt_sp_atm

def pt_continuum(P,T,abundances,g,MMW,sp):
    

    pt_sp_atm = Radtrans(line_species = ['O'],
                     wlen_bords_micron = [wlLow, wlMax],
                     continuum_opacities = [sp],
                     do_scat_emis = True)
    pt_sp_atm.setup_opa_structure(P)
    pt_sp_atm.calc_flux(T, abundances, g, MMW, contribution=False)
    return pt_sp_atm


def save_pt_sp(pt_sp_atm,starFL,sp,output_folder,file_number):
    
    dir_star_sp = output_folder + '/FpFs_species/' + sp
    create_directory_if_not_exists(dir_star_sp)
    data_sp = {'wavelength': pt_sp_atm.freq, 'flux': pt_sp_atm.flux / starFL * (Rpl / Rstar) ** 2}
    header_sp = '# Wavelength(microm)\tF_planet/F_star(ppm)\n'
    save_data(dir_star_sp + '/' + file_number + '.dat', header_sp, data_sp)

def save_pt_scatter(pt_sp_atm,starFL,sp,output_folder,file_number):
    
    dir_star_sp = output_folder + '/FpFs_species/' + sp
    create_directory_if_not_exists(dir_star_sp)
    data_sp = {'wavelength': pt_sp_atm.freq, 'flux': pt_sp_atm.flux / starFL * (Rpl / Rstar) ** 2}
    header_sp = '# Wavelength(microm)\tF_planet/F_star(ppm)\n'
    save_data(dir_star_sp + '/' + file_number + '.dat', header_sp, data_sp)

    
def emission():
    
    var, atm, param, species, VMR, plname, mass_frac, y_ini = output()
    sorted_output = sort_output(vul_directory) 
    
    for idx,abundances in enumerate(mass_frac):
        
        run = sorted_output[idx]
        print (f'Reading case {run}\n')
        
        P = atm[idx]['pco'][::-1]/1e6
        T = atm[idx]['Tco'][::-1]
        g = atm[idx]['gs'] 
        MMW = atm[idx]['mu'][::-1]
        
        
        phx = nc.get_PHOENIX_spec(stellarT)
        starWL = phx[:,0]
        starFL = phx[:,1] # in erg -s cm-3
        
        freq = nc.c / starWL
        planck = nc.b(T[-1], freq)
        planck = planck*np.pi

        # 55 Cancri e Spectra Renyu #
        R55 = np.loadtxt('input/55cncH.dat')
        wl55 = R55[:,0] # In cm
        sp55 = R55[:,1] # In erg s-1 cm-3 at stellar surface
        sp55 = sp55/nc.c*wl55**2 # to pRT format
        
        
        ptATM = Radtrans(line_species = opacities,
                         wlen_bords_micron = [wlLow, wlMax],
                         rayleigh_species=ray_scatter,
                         continuum_opacities = continuum,
                         do_scat_emis = True)
        
        ptATM.setup_opa_structure(P)          
        ptATM.calc_flux(T, abundances, g, MMW, contribution=True)
        
        # Interpolate new star flux
        #starFL = np.interp(nc.c/ptATM.freq/1e-4,starWL*1e4,starFL)
        starFL = np.interp(nc.c/ptATM.freq/1e-4,wl55*1e4,sp55) # 55 Cnc e
        planck = np.interp(nc.c/ptATM.freq/1e-4,starWL*1e4,planck)

        
        ### Save output ###
        file_number = run
        
        output_folder ='output/pRT/' + output_grid
        create_directory_if_not_exists(output_folder)
        
        
        dir_star = output_folder + '/FpFs/'
        dir_star_sp = output_folder+ '/FpFs_species/'
        dir_flux = output_folder + '/Fp/'
        dir_empty = output_folder + '/blank/'
        dir_contrib = output_folder + '/contribution/'
        dir_bb = output_folder + '/blackbody/'
        dir_TP = output_folder + '/TP/'
        dir_VMR = output_folder+ '/VMR/'
        
        
        
        create_directory_if_not_exists(dir_star)
        create_directory_if_not_exists(dir_star_sp)
        create_directory_if_not_exists(dir_flux)
        create_directory_if_not_exists(dir_empty)
        create_directory_if_not_exists(dir_bb)
        create_directory_if_not_exists(dir_contrib)
        create_directory_if_not_exists(dir_TP)
        create_directory_if_not_exists(dir_VMR)
        


        header_flux = '# Wavelength(micron)\tF_planet(10^-6 erg/cm^2/s/Hz)\n'
        header_star = '# Wavelength(microm)\tF_planet/F_star(ppm)\n'
        #header_star_sp = '# Wavelength(micron)\tF_planet(10^-6 erg/cm^2/s/Hz)\n'
        header_bb = '# Wavelength(microm)\tF_planet(BB)/F_star\n'
        header_TP = '# T(K)\tP(bar)\n'
        
        
        #Setting up empty and single species objects
        if analyse_spectrum:
            pt_empty_atm = pt_empty(P,T,abundances,g,MMW)
            for sp in abundances:
                if sp in opacities:
                    avg = np.mean(abundances[sp])
                    if avg > 1e-10:
                        print (avg)
                        pt_sp_atm = pt_sp(P,T,abundances,g,MMW,sp)
                        save_pt_sp(pt_sp_atm,starFL,sp,output_folder,file_number)
            for sp in continuum:
                pt_sp_atm = pt_continuum(P,T,abundances,g,MMW,sp)
                save_pt_sp(pt_sp_atm,starFL,sp,output_folder,file_number)
                
                
        data_flux = {'wavelength': ptATM.freq, 'flux': ptATM.flux}
        data_star = {'wavelength': ptATM.freq, 'flux': ptATM.flux / starFL * (Rpl / Rstar) ** 2}
        data_empty = {'wavelength': pt_empty_atm.freq, 'flux': pt_empty_atm.flux / starFL * (Rpl / Rstar) ** 2}
        data_bb = {'wavelength': ptATM.freq, 'flux': planck / starFL * (Rpl / Rstar) ** 2}
        data_TP = {'temperature': T, 'pressure': P}

        save_data(dir_flux + file_number + '.dat', header_flux, data_flux)
        save_data(dir_star + file_number + '.dat', header_star, data_star)
        save_data(dir_bb + file_number + '.dat', header_bb, data_bb)        
        save_data_TP(dir_TP + file_number + '.dat', header_TP, data_TP) 
        save_data(dir_empty + file_number + '.dat', header_star, data_empty)
        
        
        # Save VMR
        species_vmr = {}
        for sp_idx, sp in enumerate(species[idx]):
            species_vmr[sp] = VMR[idx][:, sp_idx]
        # Save all species VMR data to a single file
        np.save(f'output/pRT/{output_grid}/VMR/{run}_species_vmr.npy', species_vmr)

        
        
        
        # Save contribution function
        np.save(dir_contrib + file_number + '.npy', ptATM.contr_em) 


def calcChi(wl,flux,wl_jwst,flux_JWST,errorsDown,errorsUp,errorsLeft):
    
    # For NIRCAM leave alone
    # For miri average 
    
    wl_error = errorsLeft
    error_y = np.array((errorsDown+errorsUp)/2.)
    modelWL = wl
    modelFL = flux
    
    lowest_chi = float('inf')  # Initialize with positive infinity
    lowest_chi_nircam = float('inf')  # Initialize with positive infinity
    lowest_chi_miri = float('inf')  # Initialize with positive infinity
    lowest_chi_reduced = float('inf')  # Initialize with positive infinity
    
    offset_range = np.linspace(0,120,121)
    best_offset = 0
    best_offset_nircam = 0
    best_offset_miri = 0
    best_offset_reduced = 0
    
    for off in offset_range:
        
        offset_jwst_measurements = flux_JWST.copy()
        current_chi = 0.  # Initialize for the current offset
        current_chi_nircam = 0.
        current_chi_miri = 0.
        current_chi_reduced = 0.
        
        #print (f'offset_jwst {offset_jwst_measurements}')
        
        for i,errorx in enumerate(wl_jwst):
            
            if errorx < 6.0:
                # For nircam just take closest point
                closest_idx = np.argmin(np.abs(modelWL - errorx))
                model_flux_within_bin = modelFL[closest_idx]
                offset_jwst_measurements[i] += off
                
                # Nircam chi
                current_chi_nircam += np.sum(((offset_jwst_measurements[i] - model_flux_within_bin) / error_y[i]) ** 2)
                
            else:
                
                # For miri average of bin
                within_bin = (modelWL >= errorx - wl_error[i]) & (modelWL <= errorx + wl_error[i])
                model_flux_within_bin = np.mean(modelFL[within_bin])
                # MIRI chi
                current_chi_miri += np.sum(((offset_jwst_measurements[i] - model_flux_within_bin) / error_y[i]) ** 2)

                
            current_chi += np.sum(((offset_jwst_measurements[i] - model_flux_within_bin) / error_y[i]) ** 2)
            
            if i < len(wl_jwst) - 1:  # Not adding the last data point
                current_chi_reduced += np.sum(((offset_jwst_measurements[i] - model_flux_within_bin) / error_y[i]) ** 2)
            
            
        if current_chi < lowest_chi:          
            lowest_chi = current_chi     
            best_offset = off

        if current_chi_nircam < lowest_chi_nircam:          
            lowest_chi_nircam = current_chi_nircam     
            #best_offset_nircam = off

        if current_chi_miri < lowest_chi_miri:          
            lowest_chi_miri = current_chi_miri     
            #best_offset_miri = off

        if current_chi_reduced < lowest_chi_reduced:          
            lowest_chi_reduced = current_chi_reduced     
            #best_offset_reduced = off

        
    return lowest_chi,lowest_chi_nircam,lowest_chi_miri,lowest_chi_reduced, best_offset


def calcChi_select(wl,flux,wl_jwst,flux_JWST,errorsDown,errorsUp,errorsLeft):
    
    # For NIRCAM leave alone
    # For miri average 
    
    wl_error = errorsLeft
    error_y = np.array((errorsDown+errorsUp)/2.)
    modelWL = wl
    modelFL = flux
    
    lowest_chi = float('inf')  # Initialize with positive infinity
    lowest_chi_nircam = float('inf')  # Initialize with positive infinity
    lowest_chi_miri = float('inf')  # Initialize with positive infinity
    lowest_chi_reduced = float('inf')  # Initialize with positive infinity
    
    offset_range = [0,55,115]
    best_offset = 0
    best_offset_nircam = 0
    best_offset_miri = 0
    best_offset_reduced = 0
    
    chi_values = []
    for off in offset_range:
        
        offset_jwst_measurements = flux_JWST.copy()
        current_chi = 0.  # Initialize for the current offset
        current_chi_nircam = 0.
        current_chi_miri = 0.
        current_chi_reduced = 0.
        
        #print (f'offset_jwst {offset_jwst_measurements}')
        
        for i,errorx in enumerate(wl_jwst):
            
            if errorx < 6.0:
                # For nircam just take closest point
                closest_idx = np.argmin(np.abs(modelWL - errorx))
                model_flux_within_bin = modelFL[closest_idx]
                offset_jwst_measurements[i] += off
                
                # Nircam chi
                current_chi_nircam += np.sum(((offset_jwst_measurements[i] - model_flux_within_bin) / error_y[i]) ** 2)
                
            else:
                
                # For miri average of bin
                within_bin = (modelWL >= errorx - wl_error[i]) & (modelWL <= errorx + wl_error[i])
                model_flux_within_bin = np.mean(modelFL[within_bin])
                # MIRI chi
                current_chi_miri += np.sum(((offset_jwst_measurements[i] - model_flux_within_bin) / error_y[i]) ** 2)

                
            current_chi += np.sum(((offset_jwst_measurements[i] - model_flux_within_bin) / error_y[i]) ** 2)
            
            if i < len(wl_jwst) - 1:  # Not adding the last data point
                current_chi_reduced += np.sum(((offset_jwst_measurements[i] - model_flux_within_bin) / error_y[i]) ** 2)
            
        
        
        chi_values.append(current_chi) # Should return 3 values
# =============================================================================
#         if current_chi < lowest_chi:          
#             lowest_chi = current_chi     
#             best_offset = off
# 
#         if current_chi_nircam < lowest_chi_nircam:          
#             lowest_chi_nircam = current_chi_nircam     
#             #best_offset_nircam = off
# 
#         if current_chi_miri < lowest_chi_miri:          
#             lowest_chi_miri = current_chi_miri     
#             #best_offset_miri = off
# 
#         if current_chi_reduced < lowest_chi_reduced:          
#             lowest_chi_reduced = current_chi_reduced     
#             #best_offset_reduced = off
# =============================================================================

    return chi_values
    #return lowest_chi,lowest_chi_nircam,lowest_chi_miri,lowest_chi_reduced, best_offset


def create_figure(pRT_out):
    

    gs = gridspec.GridSpec(nrows=2, ncols=3, width_ratios=[1, 1, 1], height_ratios=[1, 1])   
    gs.update(wspace=0.15, hspace=0.15)
    
    fig = plt.figure(figsize=(21, 11))
    

    
    
    listed_output = os.listdir(pRT_out)
    sorted_output = sorted(os.path.join(pRT_out, file) for file in listed_output)

    # For atm type composition file
    x_min_0 = []
    x_min_55 = []
    x_min_115 = []
    x_min_nircam = []
    x_min_miri = []
    x_min_reduced = []
    x_min_run = []
    offset_min = []
    offset_nircam = []
    offset_miri = []
    offset_reduced = []

    for idx,case in enumerate(sorted_output):


        # Top panel (spans all three columns)
        ax_top = fig.add_subplot(gs[0, :])
        
        # Three panels below the top one
        ax_bottom_left = fig.add_subplot(gs[1, 0])
        ax_bottom_center = fig.add_subplot(gs[1, 1])
        ax_bottom_right = fig.add_subplot(gs[1, 2])


        all_opacities = opacities + continuum
        colorLen = np.linspace(0, 1, len(all_opacities))
        #colors = [matplotlib.cm.tab20(x) for x in colorLen]
        
        # Define a list of modern colors
        modern_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                         '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                         '#7fc97f', '#beaed4', '#fdc086', '#ffff99', '#386cb0',
                         '#f0027f', '#bf5b17', '#666666', '#e41a1c', '#377eb8']
        
        # Create a ListedColormap
        modern_colormap = mcolors.ListedColormap(modern_colors)
        colors = [modern_colormap(x) for x in colorLen]
        
        colors_iter_vmr = iter(colors)
        colors_iter_spectra = iter(colors)
        
            
        print (f'Generating figure {case}')
        
        file_number = os.path.basename(case)[:-4]
        
        out_file = np.loadtxt(case)
        wl = out_file[:,0]
        flux = out_file[:,1]
        
        vmr_file_path = f'output/pRT/{output_grid}/VMR/{file_number}_species_vmr.npy'
        if os.path.exists(vmr_file_path):
            vmr_data = np.load(vmr_file_path, allow_pickle=True).item()
            #print (vmr_data.keys())
        
        lw=2.0
        legC = '#eef8f8'
        
        
        #### Spectrum
        ax_top.plot(wl,flux,lw=lw,alpha=1.0,c='k',label='Total',zorder=100)
        
 
        
        if analyse_spectrum:
            file_empty = np.loadtxt(f'output/pRT/{output_grid}/blank/{file_number}.dat')
            ax_top.plot(file_empty[:,0],file_empty[:,1],lw=lw,alpha=1.0,c='r',label='Empty',zorder=100) 
            
            #Individual species
            #folders_sp = listed_output = os.listdir(pRT_out[:-1] + '_species/')
            for sp, color in zip(all_opacities, colors_iter_spectra):
                sp_file = pRT_out[:-1] + '_species/' + sp + '/' + file_number + '.dat'
                if os.path.exists(sp_file):
                    
                    file_sp = np.loadtxt(sp_file)
                    if np.any(np.abs(file_empty[:,1] - file_sp[:,1])  > 10):
                        ax_top.plot(file_sp[:,0],file_sp[:,1],lw=lw,alpha=1.0,c=color,label=sp)
                    

            
        ax_top.set(xlim=[wlLow,13],ylim=[0,300], xlabel=r'Wavelength (micron)',
               ylabel=r'$\rm F_{planet}/F_{star}$ ppm')          


        handles, labels = ax_top.get_legend_handles_labels()
        handles_red_patch, labels_red_patch = ax_top.get_legend_handles_labels()
        

        ### Planet Errors
        errors55 =  ascii.read('input/55cnce_formodelv2.txt')
        wl_jwst = errors55['Wavelength [um]']
        flux_NIRCAM = errors55['Depth [ppm]'][0:-9]
        flux_MIRI = errors55['Depth [ppm]'][-9:]
       
        errorsLeft = errors55['Wave_left']
        errorsRight = errors55['Wave_right']
        errorsUp = errors55['Depth_pos']   
        errorsDown = errors55['Depth_neg']     
        
        flux_JWST = list(flux_NIRCAM) + list(flux_MIRI)
        

        chi_values = calcChi_select(wl, flux, wl_jwst, flux_JWST, errorsDown, errorsUp, errorsLeft)
        #lowest_chi,lowest_chi_nircam,lowest_chi_miri,lowest_chi_reduced, best_offset = calcChi(wl,flux,wl_jwst,flux_JWST,errorsDown,errorsUp,errorsLeft)
        flux_NIRCAM_new = [value + 55 for value in flux_NIRCAM]

        flux_JWST_offset = list(flux_NIRCAM_new) + list(flux_MIRI) # Recalculate
        
        
        ax_top.errorbar(wl_jwst, flux_JWST_offset,xerr= [errorsLeft,errorsRight],
                      yerr=[errorsDown,errorsUp], markeredgecolor='white',
                      ecolor='mediumblue', fmt='s', markersize=2, capsize=1.5,
                      capthick=1, elinewidth=1, color='mediumblue', alpha=1,zorder=100)
        # Add a transparent shadow using fill_between
        ax_top.fill_between(wl_jwst, flux_JWST_offset - errorsDown, flux_JWST_offset + errorsUp,
                        color='mediumblue', alpha=0.1, zorder=49)

        
        # Appending values for csv saving file
        x_min_0.append(chi_values[0])
        x_min_55.append(chi_values[1])        
        x_min_115.append(chi_values[2]) 
# =============================================================================
#         x_min_nircam.append(lowest_chi_nircam)
#         x_min_miri.append(lowest_chi_miri)
#         x_min_reduced.append(lowest_chi_reduced)
# =============================================================================
        
        x_min_run.append(file_number)
        
# =============================================================================
#         offset_min.append(best_offset)
# =============================================================================
# =============================================================================
#         offset_nircam.append(best_offset_nircam)
#         offset_miri.append(best_offset_miri)
#         offset_reduced.append(best_offset_reduced)        
# =============================================================================
        
        
# =============================================================================
#         red_patch = mpatches.Patch(color='black', label=f'NIRCam offset = {best_offset:.2f}\n $\\chi^2$ = {lowest_chi:.2f}')
#         handles.append(red_patch)
#         
#         red_patch_nircam = mpatches.Patch(color='black', label=f'NIRCam only $\\chi^2$ = {lowest_chi_nircam:.2f}')    
#         handles.append(red_patch_nircam)
#         
#         red_patch_miri = mpatches.Patch(color='black', label=f'MIRI only $\\chi^2$ = {lowest_chi_nircam:.2f}')
#         handles.append(red_patch_miri)
#         
#         red_patch_reduced = mpatches.Patch(color='black', label=f'Reduced $\\chi^2$ = {lowest_chi_nircam:.2f}')
#         handles.append(red_patch_reduced)
#         
#         labels.append(f'NIRCam offset = {best_offset:.2f}\n $\\chi^2$ = {lowest_chi:.2f}')
#         labels.append(f'NIRCam only $\\chi^2$ = {lowest_chi_nircam:.2f}')
#         labels.append(f'MIRI only $\\chi^2$ = {lowest_chi_miri:.2f}')
#         labels.append(f'Reduced $\\chi^2$ = {lowest_chi_reduced:.2f}')
# =============================================================================
        combined_legend = ax_top.legend(handles, labels, loc='upper left', fontsize=7,
                                        handletextpad=0.3, markerscale=1, frameon=True, facecolor=legC, handlelength=1.5,bbox_to_anchor=(0.02, 1))
        combined_legend.get_frame().set_alpha(0.7)
        ax_top.add_artist(combined_legend)
        for line in combined_legend.get_lines():
            line.set_linewidth(3.0) 

# =============================================================================
#         for item in combined_legend.get_lines():
#             if item.get_label() == red_patch.get_label():
#                 item.set_visible(False)
# =============================================================================


            

        ax_top.tick_params(direction='in', which='both')      

        ### T-P
        TP_file_path = f'output/pRT/{output_grid}/TP/{file_number}.dat'
        TP = np.loadtxt(TP_file_path)
        ax_bottom_left.plot(TP[:,0],TP[:,1],c='k',lw=2)
        
        
        ax_bottom_left.set(yscale='log', xlabel=r'Temperature (K)', ylabel=r'Pressure (bar)',xlim=[1000,4000],ylim=[min(TP[:,1]),max(TP[:,1])])
        ax_bottom_left.tick_params(direction='in', which='both') 
        ax_bottom_left.invert_yaxis()       
        
        ### Chemistry

        species_labels = []  # To store species labels for adding to legend
        offset_increment = 0.1  # You can adjust this value based on the separation you want
        current_offset = 0
        label_x_positions = []
        
        chem_opacities = opacities + ['H2']
        for sp in chem_opacities:
             

            if '+' in sp:
                sp = sp[:-1] + '_p'
            if sp == 'e-':
                sp = 'e'
            if sp == 'H-':
                sp = 'H_m'

            
            species_label = f'{sp}'
            species_labels.append(species_label)
            current_offset += offset_increment
            c = next(colors_iter_vmr)
            ax_bottom_right.plot(vmr_data[sp][::-1],TP[:,1],lw=2,alpha=1.0,color=c,label=sp)
            

            # Plot the tiny label
            label_y = TP[:, 1][-1]/10**0.5

            if vmr_data[sp][0] > 1e-14:
                label_x_positions.append(vmr_data[sp][0])
                overlap_number = sum(0.5 <= vmr_data[sp][0] / item <= 2.0 and vmr_data[sp][0]/ item != 1.0 for item in label_x_positions)
                if overlap_number > 0.:
                    label_y = label_y / (10**(overlap_number/2))


                text = ax_bottom_right.text(vmr_data[sp][0], label_y, sp, fontsize=7, color=c, ha='left', va='center',rotation='vertical')
                
                # Add a white outline with reduced thickness
                text.set_path_effects([withStroke(linewidth=1, foreground='white')])

            
        ax_bottom_right.set(xlim=[1e-14,1],ylim=[min(TP[:,1]),max(TP[:,1])],
                            yscale='log',xscale='log', xlabel=r'Volume Mixing Ratio',
                            ylabel=r'Pressure (bar)')    
        
        
        ax_bottom_right.invert_yaxis() 

        ax_bottom_right.tick_params(direction='in', which='both')        
        

        # Contribution
        contr_file = np.load(f'output/pRT/{output_grid}/contribution/{file_number}.npy',allow_pickle=True)

        X,Y = np.meshgrid(wl,TP[:,1])
        ax_bottom_center.contourf(X,Y,contr_file,30,cmap=plt.cm.bone_r,zorder=-99)
        ax_bottom_center.set(yscale='log', xlabel=r'Wavelength (micron)',ylabel=r'Pressure (bar)',xlim=[min(wl),max(wl)],ylim=[min(TP[:,1]),max(TP[:,1])])
        ax_bottom_center.tick_params(direction='in', which='both') 
        ax_bottom_center.invert_yaxis()

        

        sns.despine()
        figures_folder = 'output/pRT/' + output_grid + '_figures_specific/'
        os.makedirs(figures_folder, exist_ok=True)
        #plt.savefig(os.path.join(figures_folder, file_number + '.png'),dpi=400,bbox_inches='tight')
        
        # Clear the figure to start a new one for the next iteration
        plt.clf()

    with open('output/x_min_specific.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['x_min_0','x_min_55','x_min_115','x_min_run'])
        csv_writer.writerows(list(zip(x_min_0,x_min_55,x_min_115,x_min_run)))
        
    
def plot_atm_comp(ax, grid, x_values, y_values, xlabel, ylabel, x_runs):
    im = ax.imshow(grid, origin='lower', cmap='gnuplot', norm=LogNorm(vmin=np.nanmin(grid), vmax=np.nanmax(grid)))

    ax.set_xticks(np.arange(len(y_values)))
    
    if xlabel == 'N + H':
        ax.set_xticklabels([f'{val:.4f}' for val in y_values], rotation='vertical', fontsize=8)  # Update this line
    else:
        ax.set_xticklabels(y_values, rotation='vertical',fontsize=8)
    ax.set_yticks(np.arange(len(x_values)))
    ax.set_yticklabels(x_values, rotation='horizontal',fontsize=8)


    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

# =============================================================================
#     # Add the lowest chi-squared value as text in each square outlined with white
#     for i, x_val in enumerate(x_values):
#         for j, y_val in enumerate(y_values):
#             lowest_chi = grid[i, j]
# 
#             if not np.isnan(lowest_chi):
#                 if xlabel == 'N + H':
#                     offset = x_offsets[i, j]
#                     run = x_runs[i, j]
#                     ax.text(j, i, f'{lowest_chi:.2f}\n({offset:.0f}, {run:.0f})', color='white', ha='center', va='center', fontsize=2)
#                 else:
#                     offset = x_offsets[i, j]
#                     run = x_runs[i, j]
#                     ax.text(j, i, f'{lowest_chi:.2f}\n({offset:.0f}, {run:.0f})', color='white', ha='center', va='center', fontsize=5)
# =============================================================================



def atm_comp_giga(comp_path, planetList_path, x_min_path,condition):
    comps = ascii.read(comp_path)
    planetList = ascii.read(planetList_path)
    x_min = ascii.read(x_min_path)
    runs = x_min['x_min_run']
    cases = ['x_min','x_min_nircam','x_min_miri','x_min_reduced']

    cases = ['x_min_0','x_min_55','x_min_115']
    for case in cases:
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=False, sharey=False)
        fig.subplots_adjust(wspace=0.4)
    
        elements_info = [
            ('N', 'H', 'N + H', 'N/H', 'NH.png'),
            ('S', 'P', 'S + P', 'S/P', 'SP.png'),
            ('C', 'O', 'C + O', 'C/O', 'CO.png')       
        ]
    
        for (element_x, element_y, xlabel, ylabel, output_filename), ax in zip(elements_info, axes):
            x = []
            y = []
            x_min_list = []
            x_offset_list = []
            x_run_list = []
    
            for idx, run in enumerate(runs):
                
            
                xx = x_min[case][idx]
                
                xx_offset = x_min['offset'][idx]
                xx_run = x_min['x_min_run'][idx]
                pl_list_idx = np.where(planetList['run'] == run)
                pl_comp = planetList['comp'][pl_list_idx][0]
                pl_P = planetList['TotalP'][pl_list_idx]
                pl_f = planetList['f'][pl_list_idx]
                x_val = comps[element_x][pl_comp] / comps[element_y][pl_comp]
                y_val = comps[element_x][pl_comp] + comps[element_y][pl_comp]
                
                if (condition[0] == pl_P or condition[0] == 'all') and (condition[1] == pl_f or condition[1] == 'all'):
                    x_min_list.append(xx)
                    x_offset_list.append(xx_offset)
                    x_run_list.append(xx_run)
                    x.append(x_val)
                    y.append(y_val)
                    
            x_values = np.unique(np.around(x, decimals=5))
            y_values = np.unique(y)
            

    
            grid = np.zeros((len(x_values), len(y_values)))

            
            offset_grid = np.zeros((len(x_values), len(y_values)))
            run_grid = np.zeros((len(x_values), len(y_values)))

            
            
            tolerance = 0.01
    
            # y_val is the ratio
            for i, x_val in enumerate(x_values):
                for j, y_val in enumerate(y_values):
                    if xlabel == 'N + H':
                        mask = (np.abs(x - x_val) <= tolerance) & (y == y_val)
                    else:
                        mask = (np.abs(x - x_val) <= tolerance) & (y == y_val)
            
                    x_array = np.array(x_min_list)

                    
                    
                    offset_array = np.array(x_offset_list)
                    run_array = np.array(x_run_list)
                    min_x = np.nanmin(x_array[mask]) if np.any(mask) else np.nan

            
                    if not np.all(np.isnan(x_array[mask])):
                        # Find the index where the minimum x value is located
                        min_x_index = np.unravel_index(np.nanargmin(x_array[mask]), x_array[mask].shape)

                
                        # Assign the correct offset and run values to the corresponding index
                        offset_grid[i, j] = offset_array[mask][min_x_index] if np.any(mask) else np.nan
                        run_grid[i, j] = run_array[mask][min_x_index] if np.any(mask) else np.nan

            
                    grid[i, j] = min_x

    
            plot_atm_comp(ax, grid, x_values, y_values, xlabel, ylabel, offset_grid, run_grid)
    
            
        # Add a shared colorbar
        cbar = fig.colorbar(axes[0].images[0], ax=axes, label='Min $\chi^2$',
                            ticks=LogLocator(subs=[1, 2, 3, 4, 5, 6, 7, 8, 9]), format=ScalarFormatter())
    
        fig.suptitle(case, fontsize=16)
        # Save the combined figure
        figures_folder = 'output/atm_type/'
        os.makedirs(figures_folder, exist_ok=True)
        plt.savefig(os.path.join(figures_folder, 'atm_' + case + '_' + str(condition[0]) + '_' + str(condition[1]) + '_.png'), dpi=500, bbox_inches='tight')
    
        plt.show()
        
def atm_comp_giga_specific(comp_path, planetList_path, x_min_path,condition):
    comps = ascii.read(comp_path)
    planetList = ascii.read(planetList_path)
    x_min = ascii.read(x_min_path)
    runs = x_min['x_min_run']
    cases = ['x_min','x_min_nircam','x_min_miri','x_min_reduced']

    #cases = ['x_min_0','x_min_55','x_min_115']
    
        
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=False, sharey=False)
    fig.subplots_adjust(wspace=0.4)

    elements_info = [
        ('C', 'O', 'C + O', 'C/O', '0.png'),
        ('C', 'O', 'C + O', 'C/O', '55.png'),
        ('C', 'O', 'C + O', 'C/O', '115.png')       
    ]

    for (element_x, element_y, xlabel, ylabel, output_filename), ax in zip(elements_info, axes):
        
        x = []
        y = []
        x_min_list = []
        x_offset_list = []
        x_run_list = []
        
        x_min_0_list = []
        x_min_55_list = []
        x_min_115_list = []

        f_list = []
        p_list = []
        run_list = []
        for idx, run in enumerate(runs):
            
            
            # x_min_0, x_min_55, x_min_115
            #xx = x_min[case][idx]
            
            #xx_offset = x_min['offset'][idx]
            xx_run = x_min['x_min_run'][idx]
            pl_list_idx = np.where(planetList['run'] == run)
            pl_comp = planetList['comp'][pl_list_idx][0]
            pl_P = planetList['TotalP'][pl_list_idx]
            pl_f = planetList['f'][pl_list_idx]
            #print (pl_f)
            #print (condition[1])
            x_val = comps[element_x][pl_comp] / comps[element_y][pl_comp]
            y_val = comps[element_x][pl_comp] + comps[element_y][pl_comp]

            if (condition[0] == pl_P or condition[0] == 'all') and (condition[1] == pl_f or condition[1] == 'all'):
                #x_min_list.append(x_min['x_min_0'][idx])
                #x_offset_list.append(xx_offset)
                x_min_0_list.append(x_min['x_min_0'][idx])
                x_min_55_list.append(x_min['x_min_55'][idx])
                x_min_115_list.append(x_min['x_min_115'][idx])

                x_run_list.append(xx_run)
                x.append(x_val) #C/O ratio
                y.append(y_val) #C+O
                f_list.append(pl_f.data[0])
                #print (pl_f)
                p_list.append(pl_P.data[0])
                run_list.append(xx_run)
        #print (f_list)        
        if (condition[0] == 'all') and (condition[1] == 'all'):
            with open('output_draft/x_min_specific_full.csv', 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['x_min_0','x_min_55','x_min_115','x_min_run','f','P','C/O','C+O'])
                csv_writer.writerows(list(zip(x_min_0_list,x_min_55_list,x_min_115_list,run_list,f_list,p_list,x,y)))
                                
        x_values = np.unique(np.around(x, decimals=5))
        y_values = np.unique(y)
    

        grid = np.zeros((len(x_values), len(y_values)))
        grid_55 = np.zeros((len(x_values), len(y_values)))
        grid_115 = np.zeros((len(x_values), len(y_values)))
        
        offset_grid = np.zeros((len(x_values), len(y_values)))
        run_grid = np.zeros((len(x_values), len(y_values)))

        
        
        tolerance = 0.01

        # y_val is the ratio
        for i, x_val in enumerate(x_values):
            for j, y_val in enumerate(y_values):
                mask = (np.abs(x - x_val) <= tolerance) & (y == y_val)
        
                x_0_array = np.array(x_min_0_list)
                x_55_array = np.array(x_min_55_list)                    
                x_115_array = np.array(x_min_115_list)                       
                
                #offset_array = np.array(x_offset_list)
                run_array = np.array(x_run_list)
                min_0_x = np.nanmin(x_0_array[mask]) if np.any(mask) else np.nan
                min_55_x = np.nanmin(x_55_array[mask]) if np.any(mask) else np.nan
                min_115_x = np.nanmin(x_115_array[mask]) if np.any(mask) else np.nan
        
# =============================================================================
#                     if not np.all(np.isnan(x_array[mask])):
#                         # Find the index where the minimum x value is located
#                         min_x_index = np.unravel_index(np.nanargmin(x_array[mask]), x_array[mask].shape)
#                         min_x_index = np.unravel_index(np.nanargmin(x_array[mask]), x_array[mask].shape)
#                 
#                         # Assign the correct offset and run values to the corresponding index
#                         #offset_grid[i, j] = offset_array[mask][min_x_index] if np.any(mask) else np.nan
#                         run_grid[i, j] = run_array[mask][min_x_index] if np.any(mask) else np.nan
# =============================================================================

                
                grid[i, j] = min_0_x
                grid_55[i, j] = min_55_x
                grid_115[i, j] = min_115_x

        #grid = [grid,grid55,grid_115]
        if output_filename == '0.png':
            plot_atm_comp(ax, grid, x_values, y_values, xlabel, ylabel, run_grid)
            ax.set_title('+ 0 ppm')
        if output_filename == '55.png':
            plot_atm_comp(ax, grid_55, x_values, y_values, xlabel, ylabel, run_grid)
            ax.set_title('+ 55 ppm')
        if output_filename == '115.png':
            plot_atm_comp(ax, grid_115, x_values, y_values, xlabel, ylabel, run_grid)  
            ax.set_title('+ 115 ppm')
        # Add a shared colorbar
# =============================================================================
#     cbar = fig.colorbar(axes[0].images[0], ax=axes, label='Min $\chi^2$',
#         ticks=LogLocator(subs=[1, 2, 3, 4, 5, 6, 7, 8, 9]), format=ScalarFormatter())
# =============================================================================
    
        #fig.suptitle(case, fontsize=16)
        # Save the combined figure
        
        
        #ax[0, 2].set_title('+ 115 ppm')
        figures_folder = 'output/atm_type_specific/'
        os.makedirs(figures_folder, exist_ok=True)
        plt.savefig(os.path.join(figures_folder, 'atm_' + str(condition[0]) + '_' + str(condition[1]) + '_.png'), dpi=500, bbox_inches='tight')
    
        plt.show()

### Function to calculate chi square values for +0, +55 and +115 NIRCam offsets


def draft_figure():
    

    #x_min = ascii.read('output/chi2_grid_draft.csv')
    #runs = x_min['x_min_run']

    data_03 = []
    data_667 = []
    data_all = []
    with open('output/chi2_grid_draft.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if float(row['f']) == 0.3:  # Filter based on 'f' value
                data_03.append({
                    'C/O': float(row['C/O']),
                    'C+O': float(row['C+O']),
                    'x_min_115': float(row['x_min_115']),
                    'x_min_55': float(row['x_min_55']),
                    'x_min': float(float(row['x_min'])),
                    'x_min_0': float(row['x_min_0'])})
            if float(row['f']) == 0.667:  # Filter based on 'f' value
                data_667.append({
                    'C/O': float(row['C/O']),
                    'C+O': float(row['C+O']),
                    'x_min_115': float(row['x_min_115']),
                    'x_min_55': float(row['x_min_55']),
                    'x_min': float(float(row['x_min'])),
                    'x_min_0': float(row['x_min_0'])})       
            data_all.append({
                'C/O': float(row['C/O']),
                'C+O': float(row['C+O']),
                'x_min_115': float(row['x_min_115']),
                'x_min_55': float(row['x_min_55']),
                'x_min': float(float(row['x_min'])),
                'x_min_0': float(row['x_min_0'])})  


    # Find unique combinations of C/O and C+O
    unique_combinations = set((d['C/O'], d['C+O']) for d in data_667)
    unique_combinations_03 = set((d['C/O'], d['C+O']) for d in data_03)
    unique_combinations_all = set((d['C/O'], d['C+O']) for d in data_all)
    # Find the minimum x_min_0 for each combination
    
    min_values = {}
    min_values_55 = {}
    min_values_115 = {}
    for combo in unique_combinations:
        min_values[combo] = min(d['x_min_0'] for d in data_667 if (d['C/O'], d['C+O']) == combo)
        min_values_55[combo] = min(d['x_min_55'] for d in data_667 if (d['C/O'], d['C+O']) == combo)
        min_values_115[combo] = min(d['x_min_115'] for d in data_667 if (d['C/O'], d['C+O']) == combo)
        
    min_values_03 = {}
    min_values_55_03 = {}
    min_values_115_03 = {}
    for combo in unique_combinations_03:
        min_values_03[combo] = min(d['x_min_0'] for d in data_03 if (d['C/O'], d['C+O']) == combo)
        min_values_55_03[combo] = min(d['x_min_55'] for d in data_03 if (d['C/O'], d['C+O']) == combo)
        min_values_115_03[combo] = min(d['x_min_115'] for d in data_03 if (d['C/O'], d['C+O']) == combo)
    
    min_values_all = {}
    min_values_all_03 = {}    
    min_values_all_667 = {}  
    for combo in unique_combinations_all:
        min_values_all[combo] = min(d['x_min'] for d in data_all if (d['C/O'], d['C+O']) == combo)  
        print(f"Minimum x_min for C/O={combo[0]}, C+O={combo[1]}: {min_values_all[combo]}")
        
    # Create a 2D array (grid) for plotting (f=2/3)
    unique_c_o = sorted(set(combo[0] for combo in unique_combinations))
    unique_c_plus_o = sorted(set(combo[1] for combo in unique_combinations))
    grid_667 = np.zeros((len(unique_c_o), len(unique_c_plus_o)))
    grid_667_55 = np.zeros((len(unique_c_o), len(unique_c_plus_o)))
    grid_667_115 = np.zeros((len(unique_c_o), len(unique_c_plus_o)))
    
    for i, c_o in enumerate(unique_c_o):
        for j, c_plus_o in enumerate(unique_c_plus_o):
            grid_667[i, j] = min_values[(c_o, c_plus_o)]
            grid_667_55[i, j] = min_values_55[(c_o, c_plus_o)]
            grid_667_115[i, j] = min_values_115[(c_o, c_plus_o)]


    # f=0.3 cases
    unique_c_o_03 = sorted(set(combo[0] for combo in unique_combinations_03))
    unique_c_plus_o_03 = sorted(set(combo[1] for combo in unique_combinations_03))
    grid_03 = np.zeros((len(unique_c_o_03), len(unique_c_plus_o_03)))
    grid_03_55 = np.zeros((len(unique_c_o_03), len(unique_c_plus_o_03)))
    grid_03_115 = np.zeros((len(unique_c_o_03), len(unique_c_plus_o_03)))
    
    for i, c_o in enumerate(unique_c_o_03):
        for j, c_plus_o in enumerate(unique_c_plus_o_03):
            grid_03[i, j] = min_values_03[(c_o, c_plus_o)]
            grid_03_55[i, j] = min_values_55_03[(c_o, c_plus_o)]
            grid_03_115[i, j] = min_values_115_03[(c_o, c_plus_o)]


    # Create a 2D array (grid) for plotting (f=2/3)
    unique_c_o_all = sorted(set(combo[0] for combo in unique_combinations_all))
    unique_c_plus_o_all = sorted(set(combo[1] for combo in unique_combinations_all))
    grid_all = np.zeros((len(unique_c_o_all), len(unique_c_plus_o_all)))
    for i, c_o in enumerate(unique_c_o_all):
        for j, c_plus_o in enumerate(unique_c_plus_o_all):
            grid_all[i, j] = min_values_all[(c_o, c_plus_o)]
            
            
    # Create a subplot figure with 4 rows and 3 columns
    fig = plt.figure(figsize=(15, 15))
    
    # Define the gridspec layout
    gs = gridspec.GridSpec(4, 10, width_ratios=[1,1,1,1,1,1,1,1,1,0.3], height_ratios=[2, 1.3, 1.3, 1.3])
    
    # Example data for each subplot
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = np.tan(x)
    y4 = np.exp(-0.1 * x) * np.sin(x)
    
    # Plot data on the first row's subplots
    ax0 = plt.subplot(gs[0, 0:6])
    ax0.plot(x, y1, label='Spectra')
    ax0.legend()
    
    ax1 = plt.subplot(gs[0, 6:10])  # Shifted the second subplot to the left
    ax1.plot(x, y2, label='TPs')
    ax1.legend()
    

    from matplotlib.colors import SymLogNorm
    from matplotlib.ticker import LogLocator, FixedLocator
    
    
    
    min_values = np.minimum(grid_667, grid_03)
    max_values = np.maximum(grid_667, grid_03)
    norm = LogNorm(vmin=np.nanmin([grid_667, grid_03]), vmax=1000)
    #norm=LogNorm(vmin=np.nanmin(grid_all), vmax=np.nanmax(grid_all)
    norm2 = LogNorm(vmin=np.nanmin([grid_667_55, grid_03_55,grid_all]), vmax=300)
    norm3 = LogNorm(vmin=np.nanmin([grid_667_115, grid_03_115]), vmax=500)
    # Create a color map with evenly spaced colors
    cmap = plt.get_cmap('viridis')
    
    # Plot data on the second row's subplots
    ax3 = plt.subplot(gs[1, 0:3])
    
    im = ax3.imshow(grid_667, origin='lower', cmap=cmap, norm=norm)

    # Set labels and title
    ax3.set_xticks(np.arange(len(unique_c_plus_o)))
    ax3.set_yticks(np.arange(len(unique_c_o)))
    ax3.set_xticklabels(unique_c_plus_o)
    ax3.set_yticklabels(unique_c_o)
    ax3.set_xlabel('C+O')
    ax3.set_ylabel('C/O')
    ax3.set_title('+0 ppm')


    ax4 = plt.subplot(gs[2, 0:3])
    im2 = ax4.imshow(grid_667_55, origin='lower', cmap=cmap, norm=norm2)

    # Set labels and title
    ax4.set_xticks(np.arange(len(unique_c_plus_o)))
    ax4.set_yticks(np.arange(len(unique_c_o)))
    ax4.set_xticklabels(unique_c_plus_o)
    ax4.set_yticklabels(unique_c_o)
    ax4.set_xlabel('C+O')
    ax4.set_ylabel('C/O')
    ax4.set_title('+55 ppm')
    
    ax5 = plt.subplot(gs[3, 0:3])
    im3 = ax5.imshow(grid_667_115, origin='lower', cmap=cmap, norm=norm3)

    # Set labels and title
    ax5.set_xticks(np.arange(len(unique_c_plus_o)))
    ax5.set_yticks(np.arange(len(unique_c_o)))
    ax5.set_xticklabels(unique_c_plus_o)
    ax5.set_yticklabels(unique_c_o)
    ax5.set_xlabel('C+O')
    ax5.set_ylabel('C/O')
    ax5.set_title('+115 ppm')
    
    # Plot data on the third row's subplots
    ax6 = plt.subplot(gs[1, 3:6])
    im = ax6.imshow(grid_03, origin='lower', cmap=cmap, norm=norm)

    # Set labels and title
    ax6.set_xticks(np.arange(len(unique_c_plus_o_03)))
    ax6.set_yticks(np.arange(len(unique_c_o_03)))
    ax6.set_xticklabels(unique_c_plus_o_03)
    ax6.set_yticklabels(unique_c_o_03)
    ax6.set_xlabel('C+O')
    ax6.set_ylabel('C/O')
    ax6.set_title('+0 ppm')
    
    ax7 = plt.subplot(gs[2, 3:6])
    im2 = ax7.imshow(grid_03_55, origin='lower', cmap=cmap, norm=norm2)

    # Set labels and title
    ax7.set_xticks(np.arange(len(unique_c_plus_o_03)))
    ax7.set_yticks(np.arange(len(unique_c_o_03)))
    ax7.set_xticklabels(unique_c_plus_o_03)
    ax7.set_yticklabels(unique_c_o_03)
    ax7.set_xlabel('C+O')
    ax7.set_ylabel('C/O')
    ax7.set_title('+55 ppm')
    
    ax8 = plt.subplot(gs[3, 3:6])
    
    im3 = ax8.imshow(grid_03_115, origin='lower', cmap=cmap, norm=norm3)

    # Set labels and title
    ax8.set_xticks(np.arange(len(unique_c_plus_o_03)))
    ax8.set_yticks(np.arange(len(unique_c_o_03)))
    ax8.set_xticklabels(unique_c_plus_o_03)
    ax8.set_yticklabels(unique_c_o_03)
    ax8.set_xlabel('C+O')
    ax8.set_ylabel('C/O')
    ax8.set_title('+115 ppm')
    
    # Plot data on the single subplot in the fourth row
    ax9 = plt.subplot(gs[2, 6:9])
    #norm=LogNorm(vmin=np.nanmin(grid), vmax=np.nanmax(grid))
    
    im4 = ax9.imshow(grid_all, origin='lower', cmap=cmap, norm=LogNorm(vmin=np.nanmin(grid_all), vmax=np.nanmax(grid_all)))

    # Set labels and title
    ax9.set_xticks(np.arange(len(unique_c_plus_o_all)))
    ax9.set_yticks(np.arange(len(unique_c_o_all)))
    ax9.set_xticklabels(unique_c_plus_o_all)
    ax9.set_yticklabels(unique_c_o_all)
    ax9.set_xlabel('C+O')
    ax9.set_ylabel('C/O')
    ax9.set_title('Variable offset')

    # Create a color bar axis
    cax = plt.subplot(gs[1, 9:10])
    cax2 = plt.subplot(gs[2, 9:10])
    cax3 = plt.subplot(gs[3, 9:10])
    # Add a shared colorbar
    # Create a color bar with log-scale ticks
    #additional_ticks = FixedLocator([40, 50, 60, 70, 80, 90, 100,200, 300, 500, 800, 1200 ,1500])
    #cbar = plt.colorbar(im3, cax=cax, label='Min $\chi^2$',ticks=[tick for tick in additional_ticks()],format=ScalarFormatter())
    cbar = plt.colorbar(im, cax=cax, label='Min $\chi^2$',ticks=LogLocator(subs=[1, 2, 3, 4]), format=ScalarFormatter())
    cbar = plt.colorbar(im2, cax=cax2, label='Min $\chi^2$',ticks=LogLocator(subs=[1, 2, 3, 4, 5]), format=ScalarFormatter())
    cbar = plt.colorbar(im3, cax=cax3, label='Min $\chi^2$',ticks=LogLocator(subs=[1, 2, 3, 4, 5, 6]), format=ScalarFormatter())
# =============================================================================
#     cbar = fig.colorbar(axes[0].images[0], ax=axes, label='Min $\chi^2$',
#         ticks=LogLocator(subs=[1, 2, 3, 4, 5, 6, 7, 8, 9]), format=ScalarFormatter())
# =============================================================================

    
    # Adjust layout
    plt.tight_layout()
    
    figures_folder = 'output_draft/'
    os.makedirs(figures_folder, exist_ok=True)
    plt.savefig(os.path.join('atm_full.png'), dpi=500, bbox_inches='tight')
    
    # Show the plot
    plt.show()


if __name__ == "__main__":      
    output_grid = 'grid_01'
    vul_directory = '../output/55_cnce/' + output_grid + '/' 
    
    opacities = ['Al','Al+','AlO','Ca','Ca+','CaO','Fe','Fe+','K','Mg','MgO','Mg+','Na',
                 'O','O2','O3','Si','Si+','Ti','Ti+','C2','SiH4','FeH','SiS','H3O+','OH+','H2+','NaOH',
                 'CaOH','SiH2','PO','H2CO','H2O2','HNO3','PH','CH',
                 'NH','PC','PN','H2O','CO','CH4','NH3','S','PS','CO2',
                 'HCN','OH','CH3','CN','HS','PH3',
                 'C2H2','C2H4','NS','CS','SO2','SO3','NaH','SiH','H2S','MgH','AlH','CaH','TiH','SiO','SiO2','TiO',
                 'KOH','N2','N2O','NaO','NO','SiN','SO']


    ray_scatter = ['H2','H2O','O2','CO2','N2','CO','CH4']
    
    continuum = ['H2-H2', 'H2-He','O2-O2','CO2-CO2','N2-H2','N2-N2','N2-He','H-']
    
    analyse_spectrum = True
    
    # Spectrum wavelength limits in micron
    wlLow = 2
    wlMax = 20
    
    ### Planet ###
    stellarT = 5214
    Rpl = 1.95*nc.r_earth
    Rstar = 0.98*nc.r_sun
    
    condition = [1,0.3] #first option pressure, second redistribution
    conds_all = [['all','all'],[1,'all'],[100,'all'],['all',0.3],['all',0.667],[1,0.3],[100,0.3],[1,0.667],[100,0.667]]
    #emission()
   # create_figure('output/pRT/' + output_grid + '/FpFs/')
    
# =============================================================================
#     for combo in conds_all:
#         print (combo)
#         atm_comp_giga_specific('output/compositions.csv','output/planetList.csv','output/x_min_specific.csv',combo)
# 
# =============================================================================
    draft_figure()