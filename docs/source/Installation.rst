Installation
============

wass2s requires Python 3.9 or later. The recommended approach is to install
it inside a dedicated conda environment so that all binary dependencies
(GDAL, Cartopy, Dask, etc.) are resolved consistently.

1. Create and activate the environment
--------------------------------------

Pre-built environment files are provided for both platforms.

**Linux / macOS**

Download the environment file from the repository:

.. code-block:: bash

   wget https://raw.githubusercontent.com/hmandela/WASS2S/main/WAS_S2S_linux.yml

Then create and activate the environment:

.. code-block:: bash

   conda env create -f WAS_S2S_linux.yml
   conda activate WASS2S

**Windows**

Download the Windows-specific file from the repository
(`WAS_S2S_windows.yml <https://github.com/hmandela/WASS2S/blob/main/WAS_S2S_windows.yml>`_)
and run:

.. code-block:: bash

   conda env create -f WAS_S2S_windows.yml
   conda activate WASS2S

2. Install or upgrade wass2s
-----------------------------

With the environment active, install the latest stable release from PyPI:

.. code-block:: bash

   pip install wass2s

To upgrade an existing installation:

.. code-block:: bash

   pip install --upgrade wass2s

3. Configure the CDS API (required for ERA5 and C3S data)
----------------------------------------------------------

Downloading ERA5 reanalysis and Copernicus C3S seasonal forecast data
requires a free account on the `Copernicus Climate Data Store <https://cds.climate.copernicus.eu>`_.

Once registered, create a credentials file at ``~/.cdsapirc``:

.. code-block:: text

   url: https://cds.climate.copernicus.eu/api/v2
   key: <YOUR_UID>:<YOUR_API_KEY>

Your UID and API key are available on your CDS profile page.
See the `CDS API How-to <https://cds.climate.copernicus.eu/api-how-to>`_ for
step-by-step instructions.

.. note::
   NMME data (NCEP CFS, GFDL, NASA, NCAR, CMC) is downloaded directly from
   the IRI Data Library via HTTP — no credentials are required for this source.

4. Verify the installation
---------------------------

.. code-block:: python

   import wass2s
   print(wass2s.__version__)
