# California Climate Risk and Adaptation Index (Cal-CRAI)
[![DOI:10.5281/zenodo.13840187](http://zenodo.org/badge/doi/10.5281/zenodo.13840187.svg)](https://doi.org/10.5281/zenodo.13840187) <a href="https://www.f-uji.net/view/4090"><img src="https://www.f-uji.net/badge/plastic/4090" alt="FAIR level 2.25, score 70.83 (2025-07-16 17:36:49)"></a> <br>
Victoria Ford, Jesse Espinoza, Beth McClenny

The California Climate Risk and Adaptation Index (Cal-CRAI) is a co-developed measure to quantitatively evaluate connections between California-specific climate risk needs, resilience indicators, and adaptation efforts undertaken by communities. Cal-CRAI evaluates both *community capacity* (the ability to adapt to and recover from climate events) and *hazard risk* (exposure to specific climate hazards and historical losses from relevant climate events). The Cal-CRAI evaluates the following hazards: wildfire, extreme heat, in-land flooding, drought, and sea level rise. Using a comprehensive set of socioeconomic, built environment, and natural environment indicators, it generates a composite resilience index score that integrates these assessments. A core tenant of this work is to ensure the index is transparent and accessible. 

This repository contains the code (via Jupyter Notebooks and scripts) associated with the data processing in calculating input metrics for the Cal-CRAI. 
* **data_pull**: Scripts to retrieve/scrape/manually upload datasets.
* **data_subset**: Scripts to subset datasets to California domain, if necessary. Applicable for federal level scale datasets.
* **data_reproject**: Scripts to reproject datasets to California domain, following ACS census tracts.
* **index_method**: Scripts to calculate indicators, hazard scores, and Cal-CRAI.

**History**<br>
**v1.0.0** -- Cal-CRAI v1 metrics completed. Released 12.19.2024
