<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="">
    <img src="https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">MNIST Neural Network</h3>

  <p align="center">
    This is a three layer neural network that accurately identifies the digits in the MNIST dataset!
    <br />
    <a href="https://github.com/jkat-codes/MNIST-Neural-Network"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    ·
    <a href="https://github.com/jkat-cdoes/MNIST-Neural-Network/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    ·
    <a href="https://github.com/jkat-codes/MNIST-Neural-Network/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->

## About The Project

This is a 3 layer neural network, built from scratch (raw Python), that is trained on the MNIST dataset. It is able to achieve a 94% accuracy on the test/dev set.

Here's why:

- I implemented L2 Regularization in order to minimize overfitting
- I implemented mini batch gradient descent in order to more effectively fine tune the model
- I ran numerous training loops on a wide range of hyperparameters in order to find the ones that yield the highest accuracy

Follow the `README.md` to get started!

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

- [![Python][python.io]][python-url]
- [![Pandas][pandas.io]][pandas-url]
- [![Numpy][numpy.io]][numpy-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->

## Getting Started

- Basic usage --> Run train.py to train, then run test_model.py to test the model on the test set
- Complex usage --> Utilize the given functions to find the best parameters to train on :)

### Installation

_Below is an example of how you can instruct your audience on installing and setting up your app. This template doesn't rely on any external dependencies or services._

1. Download the MNIST dataset from Kaggle [https://www.kaggle.com/c/digit-recognizer](https://www.kaggle.com/c/digit-recognizer)
2. Clone the repo
   ```sh
   git clone https://github.com/jkat-codes/MNIST-Neural-Network.git
   ```
3. Create a Python virtual environment
   ```sh
   python -m venv venv
   ```
4. Install required dependencies
   ```
   pip install numpy && pip install pandas && pip install matplotlib
   ```
5. Change the data path to your specific path
   ```
   data = read_csv('YOUR PATH')
   ```
6. Run train.py to train the model
   ```sh
   python train.py
   ```
7. After training is complete, run test_model.py
   ```
   python test_model.py
   ```
8. You can also run find_best_params and it will find the best combination of params that you specify

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ROADMAP -->

## Roadmap

- [x] Add L2 Regularization
- [x] Add Mini Batch Gradient Descent
- [ ] Add Nth Layer Functionality
- [ ] Add different optimization algoritms (ADAMW)
- [ ] Add different activation functions
- [ ] Achieve 97%+ testing accuracy

See the [open issues](https://github.com/jkat-codes/MNIST-Neural-Network/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->

## Acknowledgments

- [Andrew NG's Coursera](https://www.coursera.org/specializations/deep-learning?utm_medium=sem&utm_source=gg&utm_campaign=B2C_NAMER_deep-learning_deeplearning-ai_FTCOF_specializations_pmax-nonNRL-within-14d&campaignid=20131140422&adgroupid=6490643383&device=c&keyword=&matchtype=&network=x&devicemodel=&adposition=&creativeid=6490643383&hide_mobile_promo&gad_source=1&gclid=Cj0KCQjwyL24BhCtARIsALo0fSCBvbPXyK-UNDgza3bn1VCtlNM43x1vqCe5jxeBLt6PB_Exz8ULKc0aAjLsEALw_wcB)
- [DeepLearningAI YouTube Channel](https://www.youtube.com/@Deeplearningai)
- [Samson Zhang's Neural Network from Scratch](https://www.youtube.com/watch?v=w8yWXqWQYmU)
- [ReLu and Leaky ReLU Functions](https://www.digitalocean.com/community/tutorials/relu-function-in-python)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[python.io]: https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[python-url]: https://www.python.org/
[pandas.io]: https://img.shields.io/badge/-Pandas-333333?style=flat&logo=pandas
[pandas-url]: https://pandas.pydata.org/
[numpy.io]: https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white
[numpy-url]: https://numpy.org/
