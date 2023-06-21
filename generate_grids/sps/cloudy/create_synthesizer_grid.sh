
GRID="bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-c17.03"
SYNTHESIZER_DATA_DIR="/research/astrodata/highz/synthesizer/"
INCLUDE_SPECTRA=True

python create_synthesizer_grid.py -synthesizer_data_dir $SYNTHESIZER_DATA_DIR -grid $GRID -include_spectra $INCLUDE_SPECTRA
