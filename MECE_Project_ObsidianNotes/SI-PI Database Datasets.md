Did some basic data analysis on the parameters.csv that was on the exported dataset. 

Below are the results

DataFrame shape: (1916, 16) 
DataFrame columns: ['SIM_ID', 'PERMITTIVITY', 'CONDUCTIVITY', 'LOSSTANGENT', 'TDIEL', 'TMET', 'LAYER_AMOUNT', 'VIAS_X_AMOUNT', 'VIAS_Y_AMOUNT', 'VIA_RADIUS', 'SIMULATION', 'ANTIPAD_RADIUS', 'PITCH', 'SIGNAL_AMOUNT', 'GROUND_AMOUNT', 'POWER_AMOUNT'] 
DataFrame info: <class 'pandas.DataFrame'> RangeIndex: 1916 entries, 0 to 1915 
dtypes: float64(14), int64(1), str(1) memory usage: 239.6 KB

also there was no null values or duplicated.  The TUHH SI/PI dataset is of high quality.

The maximum and minimum values for each parameters ,

Maximum values for each parameter:
SIM_ID          |            9999
PERMITTIVITY    |        4.699781
CONDUCTIVITY    | 57986261.822743
LOSSTANGENT     |        0.029995
TDIEL           |       79.977632
TMET            |        4.098909
LAYER_AMOUNT    |            48.0
VIAS_X_AMOUNT   |             7.0
VIAS_Y_AMOUNT   |             7.0
VIA_RADIUS      |       19.979403
SIMULATION      |    sim_pkg_9999
ANTIPAD_RADIUS  |       38.670885
PITCH           |        78.17221
SIGNAL_AMOUNT   |            28.0
GROUND_AMOUNT   |            32.0
POWER_AMOUNT    |            10.0
dtype: object   |

Minimum values for each parameter:
SIM_ID         |              17
PERMITTIVITY   |        2.500231
CONDUCTIVITY   | 40005179.157498
LOSSTANGENT    |        0.000155
TDIEL          |        3.293455
...            |
SIGNAL_AMOUNT  |             2.0
GROUND_AMOUNT  |             3.0
POWER_AMOUNT   |             0.0

Also need to scale the values in the dataset using min-max scaling

We can also notice from the minimum and maximum values that min. signal power is 2 and max. signal power is 28. My understanding is via array with 16 signal vias is 32 port file and via array with 8 signal vias is 16 port? so we need to find a way to handle this for pytorch to handle it as we cannot feed different matrix sizes

so we can use domain based feature extraction to handle this problem

so ig i successfully vibe coded the python code for parsing the files

Verifying the parsed data --> via_array_dataset.pt file 

Generated dataset features: Number of extracted features: 14 

Names of extracted features: ['PERMITTIVITY', 'CONDUCTIVITY', 'LOSSTANGENT', 'TDIEL', 'TMET', 'LAYER_AMOUNT', 'VIAS_X_AMOUNT', 'VIAS_Y_AMOUNT', 'VIA_RADIUS', 'ANTIPAD_RADIUS', 'PITCH', 'SIGNAL_AMOUNT', 'GROUND_AMOUNT', 'POWER_AMOUNT']

Frequency range
Frequency range of the dataset: Total frequency points: 400 
Sweep range: 0.25 GHz to 100.00 GHz

Verifying the extracted 4x4 S-parameters

Got the results for the simulation of sample set. 

- **Reason to use 4x4 matrix structure**:

When analyzed the raw touchstone files I found that there were simulations for different ports. Wrote a small script to find the various ports available in the dataset and found the below,

Analyzing raw ports for dataset: Universal-Diff-SI-Array
Found 1916 simulation folders in variations.
--------------------------------------------------
Port count summary:
4 ports: 4 simulations
8 ports: 84 simulations
12 ports: 173 simulations
16 ports: 258 simulations
20 ports: 274 simulations
24 ports: 292 simulations
28 ports: 260 simulations
32 ports: 175 simulations
36 ports: 146 simulations
40 ports: 125 simulations
44 ports: 43 simulations
48 ports: 50 simulations
52 ports: 20 simulations
56 ports: 12 simulations
--------------------------------------------------
Analyzing raw ports for dataset: Universal-Diff-SI-Link
Found 1073 simulation folders in variations.
--------------------------------------------------
Port count summary:
4 ports: 1 simulations
8 ports: 34 simulations
12 ports: 100 simulations
16 ports: 144 simulations
20 ports: 187 simulations
24 ports: 166 simulations
28 ports: 120 simulations
32 ports: 93 simulations
40 ports: 70 simulations
48 ports: 59 simulations
56 ports: 39 simulations
64 ports: 41 simulations
72 ports: 7 simulations
80 ports: 12 simulations

--------------------------------------------------
In the array dataset the ports varied from 4 ports to 56 ports, and in link dataset it varied form 4 ports to 80 ports. I know that input for the neural network need to have a uniform tensor and it cannot have different input structure in various batches (it need to be same). This is required to calculate the loss and update weights during the batch training. In case we feed a 56x56 matrix in one batch and 8x8 matric in another batch to the same input layer, the pytorch compiler may crash.

I did have a conversation with Gemini to look for the best way to handle this - i did ask it to provide me options which I later analyzed if they were suitable for our project. It suggested me few ways to handle it - 

1. **Filtering the dataset**: It suggested me that we can filter the raw dataset and have the touchstone files of only particular ports eg.  Filter the database to extract only the files with a specific number of vias (e.g., only the 16-port / 8-via configurations). and use this for training the model. But if we filter out the dataset like this - we wont have enough data to train our model accurately 
2. Zero padding to the maximum size: It the recommended zero-padding, which works fine with computer vision or image processing applications but if we use zero padding in our data - it changes the entire physics of the simulation? It will for sure destroy the data quality because S-parameter of 0 implies perfect isolation right? and also zero reflection?
3. Using Graph Neural Network: It then recommended using graph neural network - which sounded like the right method to use in our case. But in our project we have proposed using rational layer to extract poles and residues and also use Jacobian loss function, using a graph neural network - we cannot cleanly extract the continuous poles and residues and another major concern is it can destabilize the Jacobian. 
4. Domain Specific feature extraction: This sounded like the perfect approach for our project . If we use domain specific feature extraction - So in our situation the ports vary drastically and it may cause dimensional issue. I went through Differential pairs and Differential Impedance chapter in book (SIGNAL AND POWER INTEGRITY–SIMPLIFIED by Eric Bogatin) and also the documentation of the TUHH SI/PI database where they had specifically mentioned "Two signal vias are always placed adjacent to each other to enable differential signaling through post-processing". From this i assumed that slicing the core 4x4 sub matrix for those specific active vias is the originally intended post-processing step. Since the simulation were run on physics based-3D solver dropping the other ports wont severely affect the physics of the environment on the core port? I mean the impedance loading, parasitic capacitance, and reflections from the surrounding ground planes and victim vias are already permanently there in the insertion loss ($S_{21}$) and return loss ($S_{11}$) of that core $4 \times 4$ matrix.
This was the reason I wrote a pipeline to extract those 4x4 matrix and then separating them into real and imaginary tensors to preserve the phase causality of the neural network.
