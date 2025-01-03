# California Climate Risk and Adaptation Index

The Cal-CRAI evaluates both community capacity—the ability to adapt to and recover from climate events—and hazard risk, which examines exposure to specific climate hazards such as wildfire, sea level rise, drought, extreme heat, and inland flooding, along with historical losses from these events. Using a comprehensive set of socioeconomic, built, and natural environment indicators, it generates a composite resilience index score that integrates these assessments.

DOI: [![DOI:10.5281/zenodo.13840187](http://zenodo.org/badge/doi/10.5281/zenodo.13840187.svg)](https://doi.org/10.5281/zenodo.13840187) <br>
Victoria Ford, Jesse Espinoza, Beth McClenny

This repository contains the code (via Jupyter Notebooks and scripts) associated with the data processing in calculating input metrics for the California Climate Risk and Adaptation Index (Cal-CRAI). 
* **data_pull**: Scripts to retrieve/scrape/manually upload datasets.
* **data_subset**: Scripts to subset datasets to California domain, if necessary. Applicable for federal level scale datasets.
* **data_reproject**: Scripts to reproject datasets to California domain, following ACS census tracts.
* **data_reproject**: Scripts to calculate data metrics.
* **index_method**: Scripts to calculate the Cal-CRAI.
