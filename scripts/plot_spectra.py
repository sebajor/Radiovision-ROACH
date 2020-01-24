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
dBFS      = 6.02*8 + 1.76 + 10*np.log10(nchannels/2) # Hard-coded 8-bits ADC

def main():
    # initialize roach
    roach = cd.initialize_roach(roach_ip, boffile=None)

    # create figure
    fig, lines = create_figure(16, bandwidth, dBFS)
    
    # initial setting of registers
    print("Setting and resetting registers...")
    roach.write_int(acc_len_reg, acc_len)
    roach.write_int(cnt_rst_reg, 1)
    roach.write_int(cnt_rst_reg, 0)
    print("done")

    # writing unitary constants into the calibration phase bank
    for i in range(16):
        write_phasor_reg(roach, 1+0j, [i], ['cal_phase_re', 'cal_phase_im'], 
            ['cal_phase_addr'], 'cal_phase_we', 32, 27)

    # animation definition
    def animate(_):
        # get spectral data
        specdata_list = get_specdata(roach, specbrams, 2, 
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
            'c1', 'c2', 'c3', 'c4', 'd1', 'd2', 'd3', 'd4']
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

def get_specdata(roach, specbrams, dfactor, awidth, dwidth, dtype):
    """
    Get spectral data from mbf model. Notice that the data from different
    spectrum is interleaved into the same bram, so it must be deinterleaved.
    :param roach: FpgaClient object to get the data.
    :param specbrams: list of brams where to get the data.
    :param dfactor: deinterleave factor.
    :param awidth: brams address width.
    :param dwidth: brams data width.
    :param dtype: bram data type.
    """
    specdata_list = []
    for bram in specbrams:
        specdata = cd.read_deinterleave_data(roach, bram, dfactor, awidth, 
            dwidth, dtype)
        specdata_list += specdata

    return specdata_list

def write_phasor_reg(roach, phasor, addrs, phasor_regs, addr_regs, we_reg, nbits, binpt):
    """
    Writes a phasor (complex) constant into a register from a register bank
    in the FPGA. The method to write the phasor is:

        1. Write the complex value into software registers, one for the real 
            and other for the imaginary part.
        2. Write the appropate value(s) into the address register(s). 
            This/these value(s) select the register in the register bank. 
            In some cases it also select the appropate bank if you have more 
            than one register bank.
        3. Create a positive edge (0->1) in  the we (write enable register). 
            This register is reseted to 0 before anything else in order to 
            avoid tampering with the rest of the bank.

    :param roach: FpgaClinet object for communication.
    :param phasor: complex constant to write in the register bank.
    :param addrs: list of addresses to set in the address registers to 
        properly select the register in the register bank. The number of 
        addresses must coincide with the number of address registers.
    :param phasor_regs: list of two registers for the real and imaginary part 
        of the phasor constant. E.g.: ['real_reg', 'imag_reg'].
    :param addr_regs: list of registers for the addresses in the bank.
    :param we_reg: write enable register for the bank.
    :param nbits: number of bits of the fixed point representation in the model.
    :param binpt: binary point for the fixed point representation in the model.
    """
    # 1. write phasor registers
    phasor_re = cd.float2fixed(nbits, binpt, np.real([phasor]))
    phasor_im = cd.float2fixed(nbits, binpt, np.imag([phasor]))
    roach.write_int('phasor_regs'][0], phasor_re)
    roach.write_int('phasor_regs'][1], phasor_im)

    # 2. write address registers
    for addr_reg, addr in zip(addr_regs, addrs):
        roach.write_int(addr_reg, addr)
            
    # 3. posedge in we register
    roach.write_int(we_reg, 1)
    roach.write_int(we_reg, 0)

if __name__ == '__main__':
    main()
