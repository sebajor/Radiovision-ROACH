# imports
import itertools
import calandigital as cd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from plot_spectra import get_specdata, write_phasor_reg

class Beamformer():
    """
    Represents a multi-beamformer implemented in the ROACH2. Controls the
    beams position, reads the beams data, and make spectra and colormap plots
    of the data. This script was implemented to be used with model 
    mbf_64beams.slx.
    """
    def __init__(self):
        # communication parameters
        self.roach_ip = "192.168.1.13"

        # model parameters
        self.bandwidth = 140
        
        # calibration model parameters
        self.cal_acclen_reg = "cal_acc_len"
        self.cal_cntrst_reg = "cal_cnt_rst"
        self.cal_phase_regs = ['cal_phase_re', 'cal_phase_im']
        self.cal_addr_regs  = ['cal_phase_addr']
        self.cal_nbits      = 18
        self.cal_binpt      = 17
        self.cal_we_reg     = 'cal_phase_we'
        self.cal_awidth     = 8   # bits
        self.cal_dwidth     = 128 # bits
        self.cal_pow_dtype  = '>u8'
        self.cal_xab_dtype  = '>i8'
        self.cal_pow_brams  = ["cal_probe0_xpow_pow0", "cal_probe0_xpow_pow1",
                               "cal_probe1_xpow_pow0", "cal_probe1_xpow_pow1",
                               "cal_probe2_xpow_pow0", "cal_probe2_xpow_pow1",
                               "cal_probe3_xpow_pow0", "cal_probe3_xpow_pow1"]
        self.cal_xab_brams  = ["cal_probe0_xab_ab0", "cal_probe0_xab_ab1",
                               "cal_probe0_xab_ab2", "cal_probe0_xab_ab3",
                               "cal_probe1_xab_ab0", "cal_probe1_xab_ab1",
                               "cal_probe1_xab_ab2", "cal_probe1_xab_ab3",
                               "cal_probe2_xab_ab0", "cal_probe2_xab_ab1",
                               "cal_probe2_xab_ab2", "cal_probe2_xab_ab3",
                               "cal_probe3_xab_ab0", "cal_probe3_xab_ab1",
                               "cal_probe3_xab_ab2", "cal_probe3_xab_ab3"]

        # beamforming model parameters
        self.bf_acclen_reg = "bf_acc_len"
        self.bf_cntrst_reg = "bf_cnt_rst"
        self.bf_phase_regs = ['bf_phase_re', 'bf_phase_im']
        self.bf_addr_regs  = ['bf_addr_x', 'bf_addr_y', 
                              'bf_addr_t', 'bf_addr_sub']
        self.bf_we_reg     = 'bf_phase_we'
        self.bf_nbits      = 18
        self.bf_binpt      = 17
        self.bf_awidth     = 10 # bits
        self.bf_dwidth     = 64 # bits
        self.bf_dtype      = '>u8'
        self.bf_brams      = ["bf_probe0_xpow_s0",  "bf_probe0_xpow_s1", 
                              "bf_probe0_xpow_s2",  "bf_probe0_xpow_s3",
                              "bf_probe0_xpow_s4",  "bf_probe0_xpow_s5", 
                              "bf_probe0_xpow_s6",  "bf_probe0_xpow_s7",
                              "bf_probe0_xpow_s8",  "bf_probe0_xpow_s9",  
                              "bf_probe0_xpow_s10", "bf_probe0_xpow_s11",
                              "bf_probe0_xpow_s12", "bf_probe0_xpow_s13",
                              "bf_probe0_xpow_s14", "bf_probe0_xpow_s15"]

        # front-end parameters
        self.ninputs = 16
        # element positions: a 2D array with the XYZ position of each element 
        # in the array. The position is given in wavelength units. The 
        # coordinate system origin of the center of the array.
        self.elpos = [[(0.75, 0,  0.75), (0.25, 0,  0.75), (-0.25, 0,  0.75), (-0.75, 0,  0.75)],
                      [(0.75, 0,  0.25), (0.25, 0,  0.25), (-0.25, 0,  0.25), (-0.75, 0,  0.25)],
                      [(0.75, 0, -0.25), (0.25, 0, -0.25), (-0.25, 0, -0.25), (-0.75, 0, -0.25)],
                      [(0.75, 0, -0.75), (0.25, 0, -0.75), (-0.25, 0, -0.75), (-0.75, 0, -0.75)]]

        # derivative parameters
        self.nchannels = 2**self.cal_awidth
        self.freqs     = np.linspace(0, self.bandwidth, self.nchannels, endpoint=False)
        self.dBFS      = 6.02*8 + 1.76 + 10*np.log10(self.nchannels/2) # Hard-coded 8-bits ADC
    
    def initialize(self, cal_acclen, bf_acclen):
        """
        Create roach communication object, initialize model accumulators and 
        set initial input calibration to ideal.
        :param cal_acclen: calibration accumulation length.
        :param bf_acclen: beam forming accumulation length.
        """
        # initialize roach
        self.roach = cd.initialize_roach(self.roach_ip, boffile=None)

        # initial setting of registers
        print("Setting and resetting registers...")
        self.roach.write_int(self.cal_acclen_reg, cal_acclen)
        self.roach.write_int(self.cal_cntrst_reg, 1)
        self.roach.write_int(self.cal_cntrst_reg, 0)
        self.roach.write_int(self.bf_acclen_reg, bf_acclen)
        self.roach.write_int(self.bf_cntrst_reg, 1)
        self.roach.write_int(self.bf_cntrst_reg, 0)
        print("done")
        
        # writing unitary constants into the calibration phase bank
        print("Setting ideal constants for bf initialization...")
        for addr in range(self.ninputs):
            write_phasor_reg(self.roach, 1+0j, [addr], self.cal_phase_regs, 
                self.cal_addr_regs, self.cal_we_reg, 32, 17)
        print("done")
    
    def calibrate_inputs(self, chnl):
        """
        Calibrates magnitude and phase imbalances at the ADCs inputs.
        It assumes that a test tone is being injeted at all the inputs at a
        certain channel of the spectrometers, with no external imbalances.
        :param chnl: channel used for the calibration.
        """
        # create figure
        fig, ax = create_phasor_figure()

        # get calibration data
        specdata = get_specdata(self.roach, self.cal_pow_brams, 2, 
            self.cal_awidth, self.cal_dwidth, self.cal_pow_dtype)
        xabdata  = get_xabdata(self.roach, self.cal_xab_brams, 
            self.cal_awidth, self.cal_dwidth, self.cal_xab_dtype)

        # compute ratios
        print("Computed imbalances:")
        cal_ratios = compute_ratios(specdata, xabdata, chnl)

        # plot calibration data
        plot_calibration_phasors(fig, ax, cal_ratios, [])

        # load correction constants
        for cal_ratio, addr in zip(cal_ratios, range(self.ninputs)):
            write_phasor_reg(self.roach, cal_ratio, [addr], self.cal_phase_regs, 
                self.cal_addr_regs, self.cal_we_reg, 32, 27)

        # compute new ratios
        print("Calibrated imbalances:")
        cal_ratios_new = compute_ratios(specdata, xabdata, chnl)

        # plot calibrated data
        plot_calibration_phasors(fig, ax, cal_ratios, cal_ratios_new)

        print("Close plots to finish.")
        plt.show()

    def steer_beam(self, addrs_list, az, el, chnl):
        """
        Given a phase array antenna that outputs into the ROACH,
        set the appropiate phasor constants so that the array points
        to a given direction (in azimuth and elevation angles). The exact 
        phasors are computed using the ang2phasors() function, the 
        self.elpos (element positions) parameter of the beamformer and the 
        frequency of the expected signal (downconverted).
        :param addrs_list: list of addresses from the phase bank where to set 
            the constants.
        :param az: azimuth angle to point the beam in degrees.
        :param el: elevation angle to point the beam in degrees.
        :param chnl: frequency channel of the expected input signal.
        """
        freq = self.freqs[chnl] # input frequency
        phasor_list = angs2phasors(self.elpos, freq, el, az)
        
        for phasor, addrs in zip(phasor_list, addrs_list):
            write_phasor_reg(self.roach, phasor, addrs, self.bf_phase_regs, 
                self.bf_addr_regs, self.bf_we_reg, 18, 17)

    def steer_beams(self, az_list, el_list, chnl):
        """
        Steer all beams to appropriate locations in a grid for the 64 beams 
        beamformer to create image.
        :param az_list: list of azimuth angles for the beams. Must be length 8.
        :param az_list: list of elevation angles for the beams. Must be length 8.
        :param chnl: frequency channel to use when computing the positions.
        """
        angle_pairs = itertools.product(el_list, az_list)

        print("Steering the beams for every beamformer...")
        for i in range(4):
            for j in range(4):
                for t in range(4):
                    addrs = [[i,j,t,input] for input in range(self.ninputs)]
                    el, az = angle_pairs.next()
                    print("Steering beam az:" + str(az).rjust(3) + 
                                      ", el:" + str(el).rjust(3))
                    self.steer_beam(addrs, az, el, chnl)

        print("done")

    def get_bf_data(self):
        """
        Get the data of all of the 64 beams of the beamformer.
        :return: spectral data for each beam of the beamformer.
        """
        bf_data_list = []
        # get data from bram
        for bf_bram in self.bf_brams:
            bf_data = cd.read_data(self.roach, bf_bram, self.bf_awidth, 
                self.bf_dwidth, self.bf_dtype)
            
            # split data into each individual (4) beamformers
            bf_data = np.split(bf_data, 4)

            # add data to list
            bf_data_list += bf_data

        return bf_data_list

    def plot_colormap(self, chnl, extent, acclen):
        """
        Plot a colormap of the beams power on a given channel.
        :param chnl: channel to plot.
        :param extent: limits for the colormap ploting as defined by the imshow
            function of matplotlib.
        :param acclen: accumulation length for spectral data scaling.
        """
        # create figure
        fig, ax, img, cbar = create_colormap_fig(extent)

        # animation definition
        def animate(_):
            # get spectral data from beamformers
            bf_data = self.get_bf_data()

            # extract the data from the specified channel only
            bf_data = np.array(bf_data)[:, chnl]

            # convert into dBFS
            bf_data = cd.scale_and_dBFS_specdata(bf_data, acclen, self.dBFS)

            # reshape the data into a matrix
            bf_data = np.reshape(bf_data, (8,8))

            # update the plot grid and colormap
            img.set_data(bf_data)
            img.set_clim(vmin=np.min(bf_data), vmax=np.max(bf_data))
            cbar.update_normal(img)

            return []

        ani = FuncAnimation(fig, animate, blit=True)
        plt.show()

#################################
### calibrate input functions ###
#################################
def create_phasor_figure():
    """
    Creates figure to plot the phasors on a unit circle. Used to check the
    calibration of the beamformer.
    """
    fig, ax = plt.subplots(1,1)
    fig.show()
    fig.canvas.draw()
    
    # set axis properties
    ax.set_xlim([-1.5,1.5])
    ax.set_ylim([-1.5,1.5])
    ax.grid()
    ax.set_axisbelow(True)
    ax.set_aspect('equal')

    # draw unit circle
    ax.add_artist(plt.Circle((0,0), 1, color='g', fill=False))

    # draw x at (1,0)
    ax.scatter([1], [0], color='k', marker='x', s=50)
    
    # make legend
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:2]
    rects = [plt.Rectangle((0,0),1,1,color=color,ec="k") for color in colors]
    ax.legend(rects, ['uncalibrated', 'calibrated'])

    return fig, ax

def plot_calibration_phasors(fig, ax, uncal_phasors, cal_phasors):
    """
    Plot a list of uncalibrated and calibrated phasors (complex numbers)
    a arrows in a unit circle.
    """
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:2]
    for color, phasor_list in zip(colors, [uncal_phasors, cal_phasors]):
        for i, phasor in enumerate(phasor_list):
            phr = np.real(phasor)
            phi = np.imag(phasor)
            arrow = plt.Arrow(0, 0, phr, phi, width=0.05, color=color)
            text = plt.Text(phr, phi, 'a'+str(i), color=color)
            ax.add_artist(arrow)
            ax.add_artist(text)
    
    fig.canvas.draw()

def get_xabdata(roach, xabbrams, awidth, dwidth, dtype):
    """
    Get correlated (xab) data from mbf model. Notice that the real and 
    imaginary part of the cross spectrum is interleaved, so it must be
    separated to create the compex data.
    :param roach: FpgaClient object to get the data.
    :param xabbrams: list of brams where to get the data.
    :param awidth: brams address width.
    :param dwidth: brams data width.
    :param dtype: bram data type.
    """
    xabdata_list = []
    for bram in xabbrams:
        xabdata_reim = cd.read_deinterleave_data(roach, bram, 2, awidth, 
            dwidth, dtype)
        xabdata = xabdata_reim[0] + 1j*xabdata_reim[1]
        xabdata_list.append(xabdata)

    return xabdata_list

def compute_ratios(specdata, xabdata, chnl):
    """
    Given a list of spectral data and cross-spectral data,
    compute the complex ratio (=magnitude ratio and phase difference)
    of the data for a single channel chnl. It is assumed that the
    cross-spectrum was computed as: reference x conj(signal).
    :param specdata: list of power spectral data.
    :param xabdata: list of crosspower spectral data.
    :param chnl: frequency channel of the data in which compute the
        complex ratio.
    :return: array of the complex ratios of the data. The size of
        this list is equal to the number of spectrum arrays in the
        pow_data and xab_data lists.
    """
    cal_ratios = []
    for xpow, xab in zip(specdata, xabdata):
        cal_ratios.append(xab[chnl] / xpow[chnl])

    # print computed ratios
    for i, cal_ratio in enumerate(cal_ratios):
        print("Port " + str(i).zfill(2) + \
            ": mag: " + "%0.4f" % np.abs(cal_ratio) + \
            ", ang: " + "%0.4f" % np.angle(cal_ratio, deg=True) + "[deg]")
    print("")

    return np.array(cal_ratios)

##########################
### colormap functions ###
##########################
def angs2phasors(elpos_w, freq, theta, phi): 
    """
    Computes the phasor constants for every element of
    a 2D phase array antenna in order to make it point to the
    direction given by the theta (elevation) and phi (azimuth) 
    angles. For this the relative position of all the element of 
    the array is needed as well as the frequency of the input signal. 
    :param elpos_w: relative position of the elements of the array in 
        wavelength units.
    :param freq: operation frequency of the array in MHz.
    :param theta: elevation angle to point the beam in degrees.
    :param phi: azimuth angle to point the beam in degrees.
    :return: phase constants for every element in the array to
        set in order to properly point the array to the desired direction.
    """
    c = 3e8 # speed of light
    wavelength = c / (freq*1e6)

    # get array element positions in meters
    elpos_m = wavelength * np.array(elpos_w)

    # convert angles into standard ISO shperical coordinates
    # (instead of (0,0) being the array perpendicular direction)
    theta = 90 - theta 
    phi = 90 - phi

    # convert angles into radians
    theta = np.radians(theta)
    phi = np.radians(phi)
  
    # direction of arrival
    a = np.array([-np.sin(theta) * np.cos(phi), -np.sin(theta) * np.sin(phi), -np.cos(theta)])
    k = 2 * np.pi / wavelength * a  # wave-number vector

    # Calculate array manifold vector
    v = np.exp(-1j * np.dot(elpos_m, k))

    return list(np.conj(v).flatten())

def create_colormap_fig(extent):
    """
    Creates figure and image to plot a colormap.
    """
    fig  = plt.figure()
    ax   = fig.gca()
    img  = ax.imshow([[0,0],[0,0]], origin='lower', aspect='equal', 
        interpolation='spline36', extent=extent)
    cbar = fig.colorbar(img)

    ax.set_xlabel('Azimuth ($\phi$) [$^\circ$]')
    ax.set_ylabel('Elevation ($\\theta$) [$^\circ$]')
    cbar.set_label("Power [dB]")

    return fig, ax, img, cbar

if __name__ == '__main__':
    chnl = 20 # chnl:20 => 10.9375MHz at BW=140MHz
    cal_acclen = 2**16
    bf_acclen  = 2**16
    bf = Beamformer()
    bf.initialize(cal_acclen, bf_acclen)
    #bf.calibrate_inputs(chnl)

    # beams position for testing the colormap
    az_start = -56; el_start = -56
    az_stop  =  56; el_stop  =  56
    az_step  =  16; el_step  =  16
    az_range = range(az_start, az_stop+1, az_step)
    el_range = range(el_start, el_stop+1, el_step)
    bf.steer_beams(az_range, el_range, chnl)
    
    extent = [az_start-az_step/2.0, az_stop+az_step/2.0,
              el_start-el_step/2.0, el_stop+el_step/2.0]
    bf.plot_colormap(chnl, extent, bf_acclen)
