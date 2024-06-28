# MBNN-att-for-Predicting-Molecular-Properties

## Many-Body Function Neural Network with Atomic Attention Mechanism (MBNN-att) for Predicting Molecular Properties

## Introduction
This project includes a series of explicit function-type molecular structure descriptors, as well as corresponding neural networks for modeling molecular properties. The molecular structure descriptors include: MBTR, ACSF, SOAP, and PTSD. It also provides an example of MBNN-att for both extensive and intensive properties: U0 and HOMO. The folder contains the corresponding LASP programs for training, specifically lasp-Exten and lasp-Inten.

For the global descriptor MBTR, each structure generates a corresponding structural descriptor, which is then processed using a single feed-forward neural network (NN).

For the local descriptors ACSF and SOAP, each atom in the structure generates its own atomic descriptor. Since different types of atoms should be distinguished, the project adopts the shared-weight neural network (shared-weight NN) proposed by Pearson et al. to process these atomic structural descriptors.

For the PTSD, the project implements the calculation of PTSD through the LASP program.

The project provides a comprehensive framework for predicting molecular properties using various molecular structure descriptors and neural network models, including the novel MBNN-att approach. This can be a valuable resource for researchers and practitioners working on molecular property prediction tasks.

## Requirements
1. Pytorch with version 2.0.0 (https://github.com/pytorch/pytorch)
2. Pytorch Geometric with version 2.3.1 (https://github.com/pyg-team/pytorch_geometric)
3. DScribe with version 2.0.1 (https://github.com/SINGROUP/dscribe)
4. LASP 3.4.0 (http://www.lasphub.com/#/lasp/download)

## Usage

