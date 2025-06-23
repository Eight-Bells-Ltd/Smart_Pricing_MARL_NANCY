  

# 🏆 Reverse Auction Environment


![Python Version](https://img.shields.io/badge/python-3.8.11-blue.svg)

This project implements a reverse auction environment using the PettingZoo library and trains agents using Ray RLlib. The environment simulates a reverse auction where multiple agents compete by submitting bids, with the goal of offering the lowest price while maintaining a profitable position.


docker 8000 prt exposed here https://nancy-smart-pricing.8bellsresearch.com/

## API Example Request
Send a sample request using curl:

```bash
curl -X POST http://localhost:8000/price_calculation \
  -H "Content-Type: application/json" \
  -d '{
    "services": [
      {
        "provider_id": "provider_1",
        "minprice": 100.0,
        "maxprice": 200.0,
        "availability": 0.9,
        "service_id": "service_a"
      },
      {
        "provider_id": "provider_2",
        "minprice": 150.0,
        "maxprice": 300.0,
        "availability": 0.7,
        "service_id": "service_b"
      }
    ]
  }'
```

## 🌟 Features

- Custom Reverse Auction Environment

- Training with Proximal Policy Optimization (PPO)

- Comprehensive agent evaluation

- Dynamic auction round visualization

- Automatic video generation of auction processes

## 🚀 Quick Start

### Prerequisites

- Anaconda or Miniconda or pip

- Git

### Installation

1. Clone the repository:

```bash

git clone https://github.com/Eight-Bells-Ltd/Smart_Pricing_MARL_NANCY.git

```

2. Create and activate a virtual environment with Python 3.11.8:

```bash

conda create -n reverse_auction python=3.11.8

conda activate reverse_auction

cd Smart_Pricing_MARL_NANCY

```

3. Install the required packages:

```bash

pip install -r requirements.txt

```

## 🏃‍♂️ Usage

### Training

Initiate the training process:

```bash

python  main.py  --mode  train

```

This command launches the PPO algorithm to train your agents. The trained model will be saved in the `models` directory.

### Evaluation

Evaluate your trained agents:

```bash

python  main.py  --mode  evaluate

```

This loads the most recent model and runs evaluation episodes, printing results to the console.

### Configuration

You can adjust training parameters and environment settings via the `config.yml` file, such as learning rate, number of agents, and auction rounds.

## 🧪 Environment Details

The `ReverseAuctionEnv` is a custom implementation using PettingZoo's `ParallelEnv`. Key features include:

- Multiple bidding agents

- Discrete action space for bid adjustments

- Observations including current rank, previous rank, and current round

- Rewards based on rank improvements, bid values, and final positions

### Example Environment State

| Agent | Current Bid | Rank | Previous Rank | Round |
|-------|-------------|------|---------------|-------|
| A1    | $50         | 1    | 2             | 3     |
| A2    | $55         | 2    | 1             | 3     |
| A3    | $60         | 3    | 3             | 3     |


## 📊 Visualization

The environment includes a rendering function that creates plots of each auction round. These plots are saved as PNG files in the `outputs/` directory.
  

After evaluation, a video is automatically generated from these PNG files, providing a visual representation of the auction process. This video is saved in the `outputs` directory.

## ⚙️ Customization

Fine-tune various parameters of the environment and training/evaluation process by editing the `config.yml` file.

## 📦 Pretrained Model
A pretrained PPO model is included for immediate evaluation or fine-tuning in `models/`.

To retrain from scratch or fine-tune, modify config.yml or pass CLI arguments as needed.

## 📄 License
This project is licensed under the GNU General Public License v3.0 `license.txt`.

## ✉️ Contact

For inquiries, please reach out to [ilias.theodoropoulos@8bellsresearch.com](ilias.theodoropoulos@8bellsresearch.com) or [stratos.vamvourelis@8bellsresearch.com](stratos.vamvourelis@8bellsresearch.com).

## 🙏 Acknowledgements

This project uses the following open-source libraries:

- [PettingZoo](https://github.com/PettingZoo-Team/PettingZoo)
- [Ray](https://github.com/ray-project/ray)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [OpenCV](https://opencv.org/)

The project has received funding from the Smart Networks and Services Joint Undertaking (SNS JU) under the European Union’s Horizon Europe research and innovation programme under Grant Agreement No 101096456.
![](https://github.com/Eight-Bells-Ltd/Smart_Pricing_MARL_NANCY/blob/RLlib/eu_co_funded_en.jpg?raw=true) ![](https://github.com/Eight-Bells-Ltd/Smart_Pricing_MARL_NANCY/blob/RLlib/SNS-logo-colour-web-Trimmed.png?raw=true)
