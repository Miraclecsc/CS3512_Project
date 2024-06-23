# CS3512_Project
### Project Structure

The project is organized into several directories and files, as shown below:

CS3512_PROJECT
├── audio
│   ├── asr.py
│   └── librispeech.py
├── data
│   └── sample2.flac
├── language
│   ├── chat_with_TinyLlama_AWQ.py
│   ├── eval_awq.py
│   └── llama.py
├── demo.mp4
├── main.py
└── README.md


### Directory and File Descriptions

- **audio/**
  - `asr.py`: Inference code for transcribing audio files using the Whisper model.
  - `librispeech.py`: Tests the transcription performance (inference speed and accuracy) of the Whisper model series on the LibriSpeech test dataset.dataset.

- **data/**
  - `sample2.flac`: Sample audio file used for asr-testing.

- **language/**
  - `chat_with_TinyLlama_AWQ.py`: Script for interacting with the TinyLlama model using AWQ (accelerated weight quantization).
  - `eval_awq.py`: Script for evaluating the AWQ model.
  - `llama.py`: Script for evaluating TinyLlama model with llama.cpp.
  - `AWQ.py`: Script to use AWQ to quantize language model.
  - `GPTQ.py`: Script to use GPTQ to quantize language model.

- **`demo.mp4`**: Demo video file showcasing the project.

- **`main.py`**: Main script for running the demo project on **RaspberryPi**.

- **`README.md`**: This file containing the project documentation.

### Getting Started

To get started with this project, follow the instructions below.

#### Prerequisites

- Python 3.8+
- PyTorch
- Transformers library from Hugging Face

#### Installation

Clone the repository:
   ```sh
   git clone https://github.com/Miraclecsc/CS3512_Project.git
   cd CS3512_Project
