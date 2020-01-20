#!/usr/bin/python
# Script to plot spectra of each ADC, animated.

# imports
import calandigital as cd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# communication parameters
roach_ip = "192.168.1.13"

# model parameters
bandwidth       = 140
acc_len_reg     = "cal_acc_len"
cnt_rst_reg     = "cal_cnt_rst"
bram_addr_width = 8   # bits
bram_word_width = 128 # bits
bram_data_type  = '>u8'
specbrams = ["cal_probe0_xpow_pow0", "cal_probe0_xpow_pow1",
             "cal_probe1_xpow_pow0", "cal_probe1_xpow_pow1",
             "cal_probe2_xpow_pow0", "cal_probe2_xpow_pow1",
             "cal_probe3_xpow_pow0", "cal_probe3_xpow_pow1"]

# experiment parameters
acc_len = 2**16

# derivative parameters
nbrams    = len(specbrams)
nchannels = 2**bram_addr_width
freqs     = np.linspace(0, bandwidth, nchannels, endpoint=False)
dBFS      = 6.02*8 + 1.76 + 10*np.log10(nchannels) # Hard-coded 8-bits ADC

def main():
    # initialize roach
    roach = cd.initialize_roach(roach_ip, boffile=None, rver=2)

    # create figure
    fig, lines = create_figure(16, bandwidth, dBFS)
    
    # initial setting of registers
    print("Setting and resetting registers...")
    roach.write_int(acc_len_reg, acc_len)
    roach.write_int(cnt_rst_reg, 1)
    roach.write_int(cnt_rst_reg, 0)
    print("done")

    # animation definition
    def animate(_):
        # get spectral data
        specdata_list = get_specdata(roach, specbrams, 
            bram_addr_width, bram_word_width, bram_data_type)
        for line, specdata in zip(lines, specdata_list):
            specdata = cd.scale_and_dBFS_specdata(specdata, acc_len, dBFS)
            line.set_data(freqs, specdata)
        return lines

    ani = FuncAnimation(fig, animate, blit=True)
    plt.show()

def create_figure(nspecs, bandwidth, dBFS):
    """
    Create figure with the proper axes settings for plotting spectra.
    """
    axmap = {1 : (1,1), 2 : (1,2), 4 : (2,2), 16 : (4,4)}

    fig, axes = plt.subplots(*axmap[nspecs], squeeze=False)
    fig.set_tight_layout(True)

    lines = []
    adcs = ['a1', 'a2', 'a3', 'a4', 'b1', 'b2', 'b3', 'b4',
            'a1', 'a2', 'a3', 'a4', 'b1', 'b2', 'b3', 'b4']
    for adc, ax in zip(adcs, axes.flatten()):
        ax.set_xlim(0, bandwidth)
        ax.set_ylim(-dBFS-2, 0)
        ax.set_xlabel('Frequency [MHz]')
        ax.set_ylabel('Power [dBFS]')
        ax.set_title(adc)
        ax.grid()

        line, = ax.plot([], [], animated=True)
        lines.append(line)

    return fig, lines

def get_specdata(roach, specbrams, awidth, dwidth, dtype):
    """
    Get spectral data from mbf model. Notice that the data from different
    spectrum is interleaved into the same bram, so it must be deinterleaved.
    :param roach: FpgaClient object to get the data.
    :param specbrams: list of brams where to get the data.
    :param awidth: brams address width.
    :param dwidth: brams data width.
    :param dtype: bram data type.
    """
    specdata_list = []
    for bram in specbrams:
        specdata = cd.read_deinterleave_data(roach, bram, 2, awidth, dwidth, dtype)
        specdata_list += specdata

    return specdata_list

if __name__ == '__main__':
    main()
