**ESP32 On Device Training Demo**
================================

This demo showcases the training of a neural network using the AIfES Library (Artificial Intelligence for Embedded Systems) on an ESP32 board. The network is trained to learn the XOR (Exclusive OR) gate function.

**Overview**
------------

This demo consists of three main parts:

1. **Setup the Training Data & Model**: Define the XOR function inputs and expected outputs as tensors.
2. **Train the Model**: Train the model using the Adam optimizer and calculate the loss at each epoch.
3. **Evaluate the Model**: Tests the model you just trained and prints out the results

Getting Started
---------------

1. Open the project in PlatformIO (VSCode extension).
2. Go to `src/main.cpp`: The C++ file that implements the demo.
3. Upload the code to an ESP32 board.
4. Open the serial monitor to see the training process and results.

AIfES Library Dual License
---------------

GNU Affero General Public License (AGPL) v3

> For private projects or developers of Free Open Source Software (FOSS), AIfES can be used free of charge under the GNU Affero General Public License (AGPL) v3.

Commercial License

> For commercial applications, a separate license agreement with Fraunhofer IMS is required. This applies if AIfES is to be combined and distributed with commercially licensed software and/or if you do not wish to distribute the AIfES source code under the AGPL v3. For more information and contact, refer to our homepage.

**About AIfES**
---------------

AIfES (Artificial Intelligence for Embedded Systems) is a platform-independent and standalone AI software framework optimized for embedded systems. It allows for the creation, training, and modification of Feedforward Neural Networks (FNN) and Convolutional Neural Networks (CNN) directly on the device.

**Resources**
------------

* [AIfES Website](www.aifes.ai)
* [AIfES GitHub Repository](https://github.com/Fraunhofer-IMS/AIfES_for_Arduino)


*Copyright (C) 2020-2023 Fraunhofer Institute for Microelectronic Circuits and Systems, 2024 Chris Dirks.*
