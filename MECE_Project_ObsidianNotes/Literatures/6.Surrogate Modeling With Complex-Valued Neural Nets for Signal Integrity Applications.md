This paper argue that standard real-valued neural networks struggle to capture the phase and magnitude relationship in the s parameter, so they propose a deep complex dense network (CDNet) that operated in complex domain.

Complex Dense Block:
In this paper they use complex multiplication instead of simple multiplication in order for the coupling to happen mathematically. They define the operation for complex input z = a +jb and a complex  weight w as,

$$w * z = (a \cdot \Re(w) - b \cdot \Im(w)) + j(a \cdot \Im(w) + b \cdot \Re(w))$$
here the Real output and Imaginary output depends on Real(a) and imaginary(b) inputs. This will make sure if the phase change - changing the ratio of a and b, the whole network reacts together.

Complex Activation:


To get around Liouville's Theorem, they use split complex-activations (CReLU - complex ReLU). They make this possible as below,
- They take the complex number z=a+jb
- They apply non-linearity to the Real and imaginary part separately.
- They combine them together.
    --> CReLU(z) = ReLU(a)+ jReLU(b)
    -->  CLeakyReLU(z) = LeakyReLU(a) + jLeakyReLU(b)

Causality/ Passivity Enforcer:
The s parameter has to be passive (energy can be lost by heat or radiation) so the gain of the system cannot exceed 1. So in this paper they hard coded layer at the end. The condition for passivity is that the largest singular value $\sigma_1$ must be less than or equal to 1,
$$\sigma_1(f) \le 1 \quad \forall f \in B$$ in the model, it first calculates the singular values of the output, and if ay value is >1 it applies minimum-phase filter to bring it back to 1.

And in causality- it is made sure that the output signal cannot happen before the input signal arrives. 

Algorithm 1: In the paper they apply minimum-phase filter enforcing both passivity and also by definition, the minimum phase filter has all its poles and zeros inside the unit circle so that the system is stable and causal

To construct this filter, the algorithm needs to figure out the correct phase that corresponds to the magnitude, and this is done using the Hilbert transform
$$\phi(f_i) = \mathcal{H}\{\log|\Sigma(f_i)|\}$$
this formula takes the logarithm of the magnitude and used Hilbert Tranform to calculate the corresponding phase. By deriving the phase directly from the magnitude using this transform, the system is mathematically guaranteed to be **minimum-phase**. A Minimum-phase is inherently causal so in a time-domain simulation, the output signal wont appear to start before the input signal arrives.

Forward Model workflow-

Input --> CDNet--> Passivity/Causal Layer --> Output (valid S-parameters)

This is how they have modeled the forward model

Inverse  Optimization:

They do not rain the inverse loop separately but instead re-use the forward model by freezing the weights so the network doesn't change.

Freeze the weights --> treat the input as trainable -->add constraints--> backpropagation to input

They use gradient descent to update the inputs to minimize the difference between the predicted s-params and the target s-params. As the CDNet creates smooth gradients it is useful in inverse optimization. 



