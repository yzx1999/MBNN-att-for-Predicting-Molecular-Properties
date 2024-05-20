# MBNN-att-for-Predicting-Molecular-Properties
Many-Body Function Neural Network with Atomic Attention Mechanism (MBNN-att) for Predicting Molecular Properties

We utilize the DScribe library to compute the ACSF, MBTR, PTSD, SOAP descriptors. 

For the global descriptor MBTR, each structure generates a corresponding structural descriptor. We then process these structural descriptors using a single feed-forward neural network (NN), based on Python with version 3.9.16, Pytorch with version 2.0.0.

For the local descriptor ACSF and SOAP, each atom in the structure generates its own atomic descriptor. Since different types of atoms should be distinguished, we adopt the shared-weight neural network (shared-weight NN) proposed by Pearson et al. to process these atomic structural descriptors, based on Python with version 3.9.16, Pytorch with version 2.0.0, Pytorch Geometric with version 2.3.1, DScribe with version 2.0.1.

For the PTSD, we implement the calculation of PTSD through the LASP program.

For MBNN, we have provided an example for each of the extensive and intensive properties: U0 and HOMO. In the folder, there are corresponding Lasp program for training, specifically lasp-Exten and lasp-Inten.
