# TODO LIST

### General
1. Making the step-like diffusion inspired model with Masking
2. Making the step-like diffuion inspired model with also re-harmonizing task via precise maksing
3. Make the BachVAE capable of generating midi
4. Make a Se2Seq model capable of generating melodies (then it can harmonize them with the point 2)
5. To Make a Recurrent diffusion i should firstly try with a VAE; The idea is to pass to pass as input of the VAE not only the $x$ that i whant to recreate but also the $x_{t-1}$, maybe i can stack them together in two channels. This would be actually different from using attention or lstm inside the vae because the time dependence would not be between columns but between different pianorolls. O course inside the VAE i can try to use LSTM or Attention anyway to capture the local time dependencies too.

---

## 1.
- Convolution as I did are not really generative, so I need to better study this approach

## 3.
- The BachVAE is finally generating music! the KL is essential, and the MSE helps reducing the noise; I shoul probably try using a L1 loss instead tho