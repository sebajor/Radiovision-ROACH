#!/bin/sh
plot_snapshots.py \
    --ip         192.168.1.13 \
    --bof        ../models/mbf_64beams.bof.gz \
    --rver       2 \
    --snapnames  snap_adc_a1 snap_adc_a2 snap_adc_a3 snap_adc_a4 \
                 snap_adc_b1 snap_adc_b2 snap_adc_b3 snap_adc_b4 \
                 snap_adc_c1 snap_adc_c2 snap_adc_c3 snap_adc_c4 \
                 snap_adc_d1 snap_adc_d2 snap_adc_d3 snap_adc_d4 \
    --dtype      ">i1" \
    --nsamples   200
