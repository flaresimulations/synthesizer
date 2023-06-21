
SPS_GRID="bpass-2.2.1-bin_chabrier03-0.1,300.0"
CLOUDY_GRID="c17.03"

MACHINE="apollo"
SYNTHESIZER_DATA_DIR="/research/astrodata/highz/synthesizer/"

# CLOUDY_PATH="/Users/sw376/Dropbox/Research/software/cloudy"
# SYNTHESIZER_DATA_DIR="/Users/sw376/Dropbox/Research/data/synthesizer"

python create_cloudy_input_grid.py -machine $MACHINE -synthesizer_data_dir $SYNTHESIZER_DATA_DIR -sps_grid $SPS_GRID -cloudy_grid $CLOUDY_GRID -cloudy_path $CLOUDY_PATH

