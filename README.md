# spectra-shift
Use Fourier Transform to shift energy of reference spectra (I0) wrt signal spectra (I).

There are two utilities.

### `absorption_shift_gui.py`
Allows the user to plot and (independently) shift multiple different spectra.

### `absorption_single_gui.py`
Allows the user to plot and shift one spectra at a time.

## Usage

In the directory `./example` there is a sample set of raw, `.fits` spectra files (Au) which can be used to familiarise with the utility. (To save on filespace, only a select few of the full experimental run are included.)

First, generate a `.csv` datafile using the utility `fits_to_csv.py`. If available, this utility will also use a `DataInfo.txt` file to map the filename to the corresponding time delay. This is based on Eric Doyle's convention.

If experimental data is of a different file type/convention, a utility can be built to convert that data into a similar `.csv` file which will be compatable with the shifting utility.

The shifting utility is then run on this `.csv` and opens an interactive gui which allows the user to select a time delay (or filename if `DataInfo.txt` unavailable) and shift the reference or signal measurement by a fraction of a pixel.

TODO:
* pixel to energy mapping
* check any hardcoded variables may need updated for different experimental runs/setups.
