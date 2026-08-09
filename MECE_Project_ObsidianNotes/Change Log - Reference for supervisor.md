
# Data Pipeline

I ran an initial EDA on the universal-diff-si-array and universal-diff-si-link datasets from TUHH SI/PI database and after the initial EDA, I decided to extract the ports as 4x4 s-parameter matrix, where I treated each simulation directory as a singular data point and I extracted the N-port touchstone file with general touchstone port ogic of (N+1)/2. the extracted matrix represented a single differential pair. The same was done for the Link dataset as well.

This approach affected the training of the forward model with severe methodological flaws. (explained more in the forward model section)
major flaws identified during the ablation study are,
1. Data Starvation: The major issue was the data starvation. The SI/PI database dataset had 1912 simulations and 1096 simulations for array and link datasets respectively. Extracting only one differential pair per simulation accounted for only 1912 and 1096 datapoints only, and that was not enough for the model to learn the trend. this made a severe overfitting and the average MAE was worse with more than 5dB. The predicted output did not match the ground truth curve and missed the resonance dips by a huge margin.
2. Ignoring the physics:  The coupling of the surrounding vias play a major role in the behavior of the via pair, and extracting only the 4 ports from 16 ports or 32 ports simulation violated the microwave network theory, resulting in the forward model to violate physics and provide inaccurate results. 
3. Limitation of datapoints in the dataset: The dataset contain simulations varying from 4 ports to 80 ports (in link dataset)

To overcome all these, I modified the data pipeline where to make the model learn the structure and stackup for every simulation, the pipeline now parse the via_array.txt and stackup.txt files present in each simulation. The script was now modified to extract the exact physical measurements and formulate them into three distinct feature vectors,

- Local Features: Parameter Intrinsic to the via (via radius, anti pad radius, drill size etc.)
- Global Features: These are the parameter governing the entire structure (Trace length, Array Pitch, etc.)
- I have made the script to include the contextual features, which are the topological identifiers defining the spatial location of the pair within the array (This is where I make use of the via_array.txt and stackup.txt)

Data Augmentation:
This is the major change made in the data pipeline. The initial model was struggling to learn with average MAE of 4dB+. So script was modified to iterate through the entire matrix and extract all the valid geometrical pair within the array. We use the parsed via_array.txt and stackup.txt to extract the differential pairs. If a single `.s16p` simulation contains 4 valid differential pairs, the script now dynamically slices the matrix to extract the $S_{dd}$ parameters for Pair 1, Pair 2, Pair 3, and Pair 4. Additionally, it extracts the crosstalk profiles (FEXT and NEXT) _between_ these pairs. This method increased the number of datapoints to around 5x times more and the data starvation was slightly improved where the average MAE coming down to 1.9dB where the curve was tracked with better accuracy. 

## Forward Model

The forward model was initially 