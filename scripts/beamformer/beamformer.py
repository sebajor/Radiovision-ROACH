# imports
import calandigital as cd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from ..plot_spectra import get_specdata, write_phasor_reg

class Beamformer():
    """
    Represents a multi-beamformer implemented in the ROACH2. Controls the
    beams position, reads the beams data, and make spectra and colormap plots
    of the data.
    """
    def __init__(self):
        # communication parameters
        roach_ip = "192.168.1.13"

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
        self.cal_xab_dtype  = '>u8'
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
        self.bf_addr_regs  = ['bf_phase_addr']
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
        self.elpos = [[(0.75, 0,  0.75), (0.25, 0,  0.75), (-0.25, 0,  0.75), (-0,75, 0,  0.75)],
                      [(0.75, 0,  0.25), (0.25, 0,  0.25), (-0.25, 0,  0.25), (-0,75, 0,  0.25)],
                      [(0.75, 0, -0.25), (0.25, 0, -0.25), (-0.25, 0, -0.25), (-0,75, 0, -0.25)],
                      [(0.75, 0, -0.75), (0.25, 0, -0.75), (-0.25, 0, -0.75), (-0,75, 0, -0.75)]]

        # derivative parameters
        self.nbrams    = len(specbrams)
        self.nchannels = 2**bram_addr_width
        self.freqs     = np.linspace(0, bandwidth, nchannels, endpoint=False)
        self.dBFS      = 6.02*8 + 1.76 + 10*np.log10(nchannels/2) # Hard-coded 8-bits ADC
    
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
        roach.write_int(self.cal_acclen_reg, cal_acclen)
        roach.write_int(self.cal_cntrst_reg, 1)
        roach.write_int(self.cal_cntrst_reg, 0)
        roach.write_int(self.bf_acclen_reg, bf_acclen)
        roach.write_int(self.bf_cntrst_reg, 1)
        roach.write_int(self.bf_cntrst_reg, 0)
        print("done")
        
        # writing unitary constants into the calibration phase bank
        for addr in range(self.ninputs):
            write_phasor_reg(self.roach, 1+0j, [addr], self.cal_phase_regs, 
                self.cal_addr_regs, self.cal_we_reg, 32, 27)
    
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
        print "Computed imbalances:"
        cal_ratios = compute_ratios(sepcdata, xabdata, chnl)

        # plot calibration data
        plot_calibration_phasors(fig, ax, cal_ratios, [])

        # load correction constants
        for cal_ratio, addr in zip(cal_ratios, range(self.ninputs)):
            write_phasor_reg(self.roach, cal_ratio, [addr], self.cal_phase_regs, 
                self.cal_addr_regs, self.cal_we_reg, 32, 27)


        # compute new ratios
        print "Calibrated imbalances:"
        cal_ratios_new = compute_ratios(sepcdata, xabdata, chnl)

        # plot calibrated data
        plot_calibration_phasors(fig, ax, cal_ratios, cal_ratios_new)

#################################
### calibrate input functions ###
#################################
def create_phasor_figure():
    """
    Creates figure to plot the phasors on a unit circle. Used to check the
    calibration of the beamformer.
    """
    fig, ax = plt.subplots(1,1,1)
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
    plt.legend(rects, legends)

def plot_calibration_data(ax, uncal_phasors, cal_phasors):
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
        print "Port " + str(i).zfill(2) + \
            ": mag: " + "%0.4f" % np.abs(cal_ratio) + \
            ", ang: " + "%0.4f" % np.angle(cal_ratio, deg=True) + "[deg]"
    print ""

    return np.array(cal_ratios)


if __name__ == '__main__':
    Beamformer().plot_specs()
