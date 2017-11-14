#!/usr/bin/env bash

export PREPROC_OUTPUT_DIR=/home/nrafidi/preproc_output/

for experiment in krns2 #PassAct2
do
    for subject in A I #B C D E F G H # A B C
    do
        /share/volume1/sharedapps/mne-wrapper/run_preproc.sh --subject $subject --experiment $experiment --settings /share/volume1/sharedapps/mne-wrapper/settings/default_settings_lp-150.yaml --begin-with "trans-D_nsb-5_cb-0"
    done
done