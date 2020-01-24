#!/usr/bin/python
# Script to plot the total power at each ADC, animated.

# imports
import calandigital as cd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# communication parameters
roach_ip = "192.168.1.13"

# model parameters
acc_len_reg = "pow_acc_len"
cnt_rst_reg = "pow_rst"
pow_regs    = ["power_adc_a1", "power_adc_a2", "power_adc_a3", "power_adc_a4",
               "power_adc_b1", "power_adc_b2", "power_adc_b3", "power_adc_b4",
               "power_adc_c1", "power_adc_c2", "power_adc_c3", "power_adc_c4",
               "power_adc_d1", "power_adc_d2", "power_adc_d3", "power_adc_d4"]

# experiment parameters
acc_len = 2**16

def main():
    # initialize roach communication
    roach = cd.initialize_roach(roach_ip, boffile=None)
    
    # create figure
    fig, rects = create_figure(pow_regs)

    # initial setting of registers
    print("Setting and resetting registers...")
    roach.write_int(acc_len_reg, acc_len)
    roach.write_int(cnt_rst_reg, 1)
    roach.write_int(cnt_rst_reg, 0)
    print("done")

    # animation definition
    def animate(_):
        powdata = get_powdata(roach, pow_regs)
        for rect, pow in zip(rects, powdata):
            rect.set_height(pow)
        return rects

    ani = FuncAnimation(fig, animate, blit=True)
    plt.show()

def create_figure(regs):
    """
    Create figure with the proper axes for plotting.
    """
    fig, axis = plt.subplots(1, 1)
    fig.set_tight_layout(True)
    rects = axis.bar(regs, len(regs)*[0], align='center', width=1)

    axis.set_ylim((5, -80)) # Harcoded 8-bit ADC
    axis.set_ylabel('Full Bandwidth Power [dBFS]')
    axis.grid()

    # rotate ticks labels for readability
    for tick in axis.get_xticklabels():
        tick.set_rotation(90)

    return fig, rects

def get_powdata(roach, pow_regs):
    """
    Read the power data from the registers and convert it to dBFS.
    :param roach: corr's FpgaClinet object used to read the data.
    :param pow_regs: list of the registers to read.
    :return: power data in dBFS.
    """
    # read the data
    powdata = [roach.read_int(reg) for reg in pow_regs]
    powdata = np.array(powdata) # convert into np.array for easy manipulation

    powdata = powdata / float(acc_len)  # divide by accumulation
    powdata = 10*np.log10(powdata)      # convert to dB
    powdata = powdata - (6.02*8 + 1.76) # convert to dBFS (Harcoded 8-bit ADC)

    return powdata

if __name__ == '__main__':
    main()
