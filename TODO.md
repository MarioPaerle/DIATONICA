# TODO LIST

### General
1. Making the step-like diffusion inspired model with Masking
2. Making the step-like diffuion inspired model with also re-harmonizing task via precise maksing
3. Make the BachVAE capable of generating midi
4. Make a Se2Seq model capable of generating melodies (then it can harmonize them with the point 2)
---
# 1.
- Make the BachNet capable of generating new notes, by masking in a more organized way, starting from squares to completly ereasing voices as in the "harmon" dataset
- Make it less noisy, maybe a more specific loss could help? (L1 + L2 (?))
- I Should Try Adding another skip connection
