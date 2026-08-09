The Latent TTO is now built and complete and generated designs that are physically constraint. To build the Yield Aware optimization - the initial deliverable is defined below.

Deliverable - The Yield Aware Optimization is used to search, measure, and sort the designs that considering the yield in factory (I am using IPC to consider the factory deviation/wobble) and the optimizer actively hunts for a geometry that survives the factory wobble and ranks the designs based on the yield. 


# Try 1: Lambda 0 - no yield optimization - check with inverse model generated design.
![[Pasted image 20260729211140.png]]
![[Pasted image 20260729211156.png]]
			fig1: direct fit - close match with no yield push
For the sample 1260 (one from validation dataset), the fit-only design with lambda 0 (yield Push is off) - here it is clear that the cVAE model chase the design with best fit (0.53dB) and passes the gate. But with no yield optimization (just yield measure here with lambda =0) the manufactured boards with design from this model will fail in the factory with slight wobbles with a yield of only 13.2% . This establishes the fit-only baseline that yield-aware optimization must improve upon.
![[Pasted image 20260729212629.png]]
		 Fig2: Geometry of  Top design with 13.2% yield 

The near perfect design with matching the exact targets will leave no margin for the model and hence making it fragile with slight manufacturing variance. 

## Try 2: with Jacobian (proposed in the initial plan) Lambda = 1.0

![[Pasted image 20260729213037.png]]
			fig3:  Jacobian method yield optimization
![[Pasted image 20260729213112.png]]
			Fig4: Geometry of design with lambda =1.0

Applying the original Jacobian penalty proposal did not produce any improvement to the design - it slightly reduced the sensitivity but the design still stays the same as without the yield push.  

### Try 3: Worst case Method (Wang, F., Lazarov, B.S. & Sigmund, O. On projection methods, convergence and robust formulations in topology optimization. _Struct Multidisc Optim_ **43**, 767–784 (2011). https://doi.org/10.1007/s00158-010-0602-y)

![[Pasted image 20260730000514.png]]
			fig 5: Worst case yield optimization for sample 5490
![[Pasted image 20260730000614.png]]
			fig 6: geometry of best design for sample 5490

![[Pasted image 20260730000915.png]]
			fig 7: Worst case method for sample 1260 where the yield is better but the fit is sacrificed

I tried the worst case yield aware optimization on the validation set - which is inspired from the worst case penalty from Wang et al. the optimizer nudges each geometry dimension to its ±1σ manufacturing extreme (the ±0.5 mil drill wobble, ±3% material variation from the IPC tolerance model) and ensures that even the _worst_ of these perturbed corners still meets spec. this will result in the design that is robust to the manufacturing tolerances. In this method few samples had a trade off in the actual fit with the target with 2.5dB - but with 80% yield and few samples had perfect match and with 98 to 100% yield. It totally depends on the complexity. This study was useful to find whether the robust manufacturing design exists for a particular target and the trade off between the manufacturability (yield) and the fit. I am yet to run the openEMS simulation for the validation samples run though worst-case yield optimization method. i will share the eye diagram once the run is complete in openEMS.

**************************************

Yield sensitivity study

# normal tolerances 
python3 tto_yield_aware_inverse.py --samples 5490 --mode worstcase --fit-gate 1.5

Result:
  
![[Pasted image 20260801223341.png]]

# HALF the tolerances

python3 tto_yield_aware_inverse.py --samples 5490 --mode worstcase --fit-gate 1.5 --tol-scale 0.5

Results:

![[Pasted image 20260801223621.png]]

# DOUBLE the tolerances

![[Pasted image 20260801224023.png]]

==========================================================================
  SUMMARY over 50 random validation samples
==========================================================================

  STAGE 1 -- RECONSTRUCTION ACCURACY (inverse model quality):
    median fit = 1.29 dB   mean = 1.46 dB   best = 0.42   worst = 3.28
    reconstructed within 1.5 dB : 34/50 = 68%

  STAGE 2 -- MANUFACTURABILITY (worst-case yield):
    median yield = 94.0%   mean = 81.6%
    achieved yield >= 50.0% : 45/50 = 90%
    geometry buildable         : 50/50 = 100%
    (stricter) yield >= 70%    : 42/50 = 84%

  COMBINED:
    FULLY successful (both criteria): 32/50 = 64%
    (approx 95% CI on full-success rate: 51% - 77%)

  per-sample CSV written: validation_eval.csv
==========================================================================


openems evaluation

32    7063  sim_pkg_5844         100.0      0.763 --done  good design

![[Pasted image 20260808175043.png]]

