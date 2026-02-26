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



