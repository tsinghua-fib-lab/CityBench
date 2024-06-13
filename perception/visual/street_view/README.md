# Prepare street view crawling package
You could first get package needed for street view images download from the link below:

```bash
pip install pycitysim==1.16.1
```

# Generate random points of latitude and longitude in the city you want to obtain street view images
Generate random points of latitude and longitude using Randompoints_Gen.py

# Now scrape streetview images
Run streetview_batch.py with your own Randompoints path. This file is modified based on https://github.com/tsinghua-fib-lab/pycitysim/blob/main/examples/streetview.py

# Multi-modal LLM testing
For Open-Source models, just deploy with their official guidance, then you can use the .py files we provided to test their capability.
For Closed-Source models, apply for APIs then you can run in batch with our code.

# For evaluation, simply use Results_Analysis.ipynb
Just in case, the street view names from different sources (Google, Baidu) might have different order of latitude and longitude.

