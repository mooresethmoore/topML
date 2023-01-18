
* topMOF.ipynb
	* Initial data exploration of DFT data / .cube files
	* failESP.pkl contains MOFS that have been removed from post-combustion-vsa-2.csv, they do not have readable cifs or have negative workingCap

	* Topology conversions -- near bottom of notebook
		* tFunc is calling cubicalripser directly on voxel
			* see persistCube(...)
		* tThresh0 and -05 are spatial topologies, with different threshold values
			* see persistGeoVoxel

* mofMLexplore.ipynb
	* Create frequency(2D histogram) projection of B1 for tThresh0 data
		* Reverse hash map found in phMOFmap-25_50.json
			* Also in csv form for individual MOF projection
				* phDF_tThresh0_B1.csv

* mofML.ipynb
	* Train:Test split without bias on optimization variable
		* ctrl-F trainBins
		* 
	* Naive CNN/DT
	* 



## Goal:

- Use B1 tThresh0 data --- contained in phDF_tThresh0_B1.csv, subset out points where
	- d-b<.05 or some epsilon
	- extract out subset triangular region -- approx from [-8:10]
		- Hopefully convolutions within this smaller region should eliminate correlations in dat
