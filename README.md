# Online Machine Learning

[Online Machine Learning](https://en.wikipedia.org/wiki/Online_machine_learning) is a method of machine learning in which data becomes available in a sequential order as a stream of data and is used to update the best predictor for future data at each step, as opposed to batch learning techniques which generate the best predictor by learning on the entire training data set at once. In contrast to the more traditional batch learning, online learning methods update themselves incrementally with one data point at a time. It is also used in situations where it is necessary for the algorithm to dynamically adapt to new patterns in the data, or when the data itself is generated as a function of time, e.g., stock price prediction.

(see also [Incremental Learning](https://en.wikipedia.org/wiki/Incremental_learning), [Streaming Algorithms](https://en.wikipedia.org/wiki/Streaming_algorithm), [Online Algorithm](https://en.wikipedia.org/wiki/Online_algorithm), [Prophet Inequality](https://en.wikipedia.org/wiki/Prophet_inequality), [Sequential Algorithm](https://en.wikipedia.org/wiki/Sequential_algorithm))

- [Courses and Books](#courses-and-books)
- [Blog Posts](#blog-posts)
- [Software](#software)
  - [Modelling](#modelling)
  - [Deployment](#deployment)
- [Papers](#papers)
  - [Linear Models](#linear-models)
  - [Support Vector Machines](#support-vector-machines)
  - [Neural Networks](#neural-networks)
  - [Decision Trees](#decision-trees)
  - [Unsupervised Learning](#unsupervised-learning)
  - [Time Series](#time-series)
  - [Drift Detection](#drift-detection)
  - [Anomaly Detection](#anomaly-detection)
  - [Metric Learning](#metric-learning)
  - [Ensemble Models](#ensemble-models)
  - [Expert Learning](#expert-learning)
  - [Miscellaneous](#miscellaneous)
  - [Surveys](#surveys)
  - [General-Purpose Algorithms](#general-purpose-algorithms)
  - [Hyperparameter Tuning](#hyperparameter-tuning)

## Courses and Books

- [IE 498: Online Learning and Decision Making](https://yuanz.web.illinois.edu/teaching/IE498fa19/)
- [Introduction to Continual Learning](https://deeplearning.neuromatch.io/tutorials/W3D4_ContinualLearning/student/W3D4_Tutorial1.html)
- [Introduction to Online Learning](https://parameterfree.com/lecture-notes-on-online-learning/)
- [Machine Learning the Feature](http://www.hunch.net/~mltf/) — Insights into the inner workings of Vowpal Wabbit (see [slides on online linear learning](http://www.hunch.net/~mltf/online_linear.pdf)).
- [Machine learning for data streams with practical examples in MOA](https://www.cms.waikato.ac.nz/~abifet/book/contents.html)
- [Online Methods in Machine Learning (MIT)](http://www.mit.edu/~rakhlin/6.883/)
- [Streaming 101: The world beyond batch](https://www.oreilly.com/ideas/the-world-beyond-batch-streaming-101)
- [Prediction, Learning, and Games](http://www.ii.uni.wroc.pl/~lukstafi/pmwiki/uploads/AGT/Prediction_Learning_and_Games.pdf)
- [Introduction to Online Convex Optimization](https://ocobook.cs.princeton.edu/OCObook.pdf)
- [Reinforcement Learning and Stochastic Optimization: A unified framework for sequential decisions](https://castlelab.princeton.edu/RLSO/) — The entire book builds upon Online Learning paradigm in applied learning/optimization problems, *Chapter 3  Online learning* being the reference.

## Software & Infrastructure

### Modelling

- [River](https://github.com/creme-ml/creme/) — A Python library for general purpose online machine learning.
- [dask](https://ml.dask.org/incremental.html) — Dask is a flexible library for parallel computing in Python, providing tools for Incremental Learning.
- [Jubatus](http://jubat.us/en/index.html) — Jubatus is a distributed processing framework and streaming machine learning library, including methods for Online Machine Learning.
- [LIBFFM](https://www.csie.ntu.edu.tw/~cjlin/libffm/) — A Library for Field-aware Factorization Machines.
- [LIBLINEAR](https://www.csie.ntu.edu.tw/~cjlin/liblinear/) — A Library for Large Linear Classification.
- [MOA](https://moa.cms.waikato.ac.nz/documentation/) — Popular open source framework for data stream mining, including methods for Online Machine Learning.
- [scikit-multiflow](https://scikit-multiflow.github.io/) — A machine learning package for streaming data in Python. ([scikit-learn](https://scikit-learn.org/stable/)
- [Apache Spark](https://spark.apache.org/docs/latest/streaming-programming-guide.html) — Doesn't do online learning per say, but instead mini-batches the data into fixed intervals of time.
- [Apache Flink](https://flink.apache.org/) — Stateful computations over data streams ('data stream' alternative to [Apache Sparks](https://spark.apache.org/docs/latest/streaming-programming-guide.html) 'simulated streams').
- [Apache Kafka](https://kafka.apache.org/documentation/streams/) — Apache Kafka is a distributed event store and stream-processing platform. Kafka Streams is a client library for building applications and microservices, where the input and output data are stored in Kafka clusters.
- [StreamDM](https://github.com/huawei-noah/streamDM) — A machine learning library on top of Spark Streaming.
- [Vowpal Wabbit](https://github.com/VowpalWabbit/vowpal_wabbit) — Vowpal Wabbit is a Machine Learning System which pushes the frontier of Machine Learning with techniques such as online, hashing, allreduce, reductions, learning2search, active, and interactive learning.

#### Deprecated
- [LIBOL](https://github.com/LIBOL) — A collection of Online Linear Models trained with first and second order Gradient Descent Methods.
- [SofiaML](https://code.google.com/archive/p/sofia-ml/) (see also [here](https://github.com/glycerine/sofia-ml))
- [Tornado](https://github.com/alipsgh/tornado) — The Tornado framework, designed and implemented for Adaptive Online Learning and Data Stream Mining in Python.
- [VFML](http://www.cs.washington.edu/dm/vfml/) (see also [here](https://github.com/ulmangt/vfml))

### Deployment

- [KappaML](https://www.kappaml.com/) (see also [here](https://github.com/KappaML/kappaml-core))
- [django-river-ml](https://github.com/vsoch/django-river-ml) — a Django plugin for deploying River models (see also [here](https://vsoch.github.io/django-river-ml/))
- [chantilly](https://github.com/online-ml/chantilly) — A prototype depolyment tool meant to be compatible with [River](https://github.com/creme-ml/creme/) (previously *Creme*)

## Research Papers

### Linear Models

- [Solving Large Scale Linear Prediction Problems Using Stochastic Gradient Descent Algorithms (2004)](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.58.7377)
- [Online Learning with Kernels (2004)](https://alex.smola.org/papers/2004/KivSmoWil04.pdf)
- [A Second-Order Perceptron Algorithm (2005)](http://www.datascienceassn.org/sites/default/files/Second-order%20Perception%20Algorithm.pdf)
- [Online Passive-Aggressive Algorithms (2006)](http://jmlr.csail.mit.edu/papers/volume7/crammer06a/crammer06a.pdf)
- [Logarithmic Regret Algorithms forOnline Convex Optimization (2007)](https://www.cs.princeton.edu/~ehazan/papers/log-journal.pdf)
- [Confidence-Weighted Linear Classification (2008)](https://www.cs.jhu.edu/~mdredze/publications/icml_variance.pdf)
- [Exact Convex Confidence-Weighted Learning (2008)](https://www.cs.jhu.edu/~mdredze/publications/cw_nips_08.pdf)
- [Adaptive Regularization of Weight Vectors (2009)](https://papers.nips.cc/paper/3848-adaptive-regularization-of-weight-vectors.pdf)
- [Stochastic Gradient Descent Training forL1-regularized Log-linear Models with Cumulative Penalty (2009)](https://www.aclweb.org/anthology/P09-1054)
- [Dual Averaging Methods for Regularized Stochastic Learning andOnline Optimization (2010)](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/xiao10JMLR.pdf)
- [Towards Optimal One Pass Large Scale Learning with Averaged Stochastic Gradient Descent (2011)](https://arxiv.org/abs/1107.2490)
- [Ad Click Prediction: a View from the Trenches (2013)](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41159.pdf)
- [Normalized Online Learning (2013)](https://arxiv.org/abs/1305.6646)
- [Practical Lessons from Predicting Clicks on Ads at Facebook (2014)](https://research.fb.com/wp-content/uploads/2016/11/practical-lessons-from-predicting-clicks-on-ads-at-facebook.pdf)
- [Field-aware Factorization Machines for CTR Prediction (2016)](https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf)

### Support Vector Machines

- [The Relaxed Online Maximum Margin Algorithm (2000)](https://papers.nips.cc/paper/1727-the-relaxed-online-maximum-margin-algorithm.pdf)
- [A New Approximate Maximal Margin Classification Algorithm (2001)](http://www.jmlr.org/papers/volume2/gentile01a/gentile01a.pdf)
- [Pegasos: Primal Estimated sub-GrAdient SOlver for SVM (2007)](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.74.8513)

### Neural Networks

- [Three Scenarios for Continual Learning (2019)](https://arxiv.org/pdf/1904.07734.pdf)

### Decision Trees

- [Mining High-Speed Data Streams (2000)](https://homes.cs.washington.edu/~pedrod/papers/kdd00.pdf)
- [Mondrian Forests: Efficient Online Random Forests (2014)](https://arxiv.org/abs/1406.2673)
- [AMF: Aggregated Mondrian Forests for Online Learning (2019)](https://arxiv.org/abs/1906.10529)

### Unsupervised Learning

- [BIRCH: an efficient data clustering method for very large databases (1996)](https://www2.cs.sfu.ca/CourseCentral/459/han/papers/zhang96.pdf)
- [Knowledge Acquisition Via Incremental Conceptual Clustering (2004)](http://www.inf.ufrgs.br/~engel/data/media/file/Aprendizagem/Cobweb.pdf)
- [Online and Batch Learning of Pseudo-Metrics (2004)](https://ai.stanford.edu/~ang/papers/icml04-onlinemetric.pdf)
- [Density-Based Clustering over an Evolving Data Stream with Noise (2006)](https://archive.siam.org/meetings/sdm06/proceedings/030caof.pdf)
- [Online Dictionary Learning For Sparse Coding (2009)](https://www.di.ens.fr/sierra/pdfs/icml09.pdf)
- [Web-Scale K-Means Clustering (2010)](https://www.eecs.tufts.edu/~dsculley/papers/fastkmeans.pdf)
- [Online Latent Dirichlet Allocation with Infinite Vocabulary (2013)](http://proceedings.mlr.press/v28/zhai13.pdf)
- [DeepWalk: Online Learning of Social Representations (2014)](https://arxiv.org/pdf/1403.6652.pdf)
- [Online Learning with Random Representations (2014)](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.127.2742&rep=rep1&type=pdf)
- [Online hierarchical clustering approximations (2019)](https://arxiv.org/pdf/1909.09667.pdf)

### Time Series

- [Online Learning for Time Series Prediction (2013)](https://arxiv.org/pdf/1302.6927.pdf)
- [Robust Online Time Series Prediction with Recurrent Neural Networks (2016)](https://core.ac.uk/download/pdf/148028169.pdf)
- [Learning Fast and Slow for Online Time Series Forecasting (2022)](https://arxiv.org/pdf/2202.11672.pdf)

### Drift Detection

- [A Survey on Concept Drift Adaptation (2014)](http://eprints.bournemouth.ac.uk/22491/1/ACM%20computing%20surveys.pdf)

### Anomaly Detection

- [Fast Anomaly Detection for Streaming Data (2011)](https://www.ijcai.org/Proceedings/11/Papers/254.pdf)
- [Interpretable Anomaly Detection with Mondrian Pólya Forests on Data Streams (2020)](https://arxiv.org/pdf/2008.01505.pdf)
- [Leveraging the Christoffel-Darboux Kernel for Online Outlier Detection (2022)](https://hal.laas.fr/hal-03562614/document)

### Metric Learning

- [Online and Batch Learning of Pseudo-Metrics (2004)](https://ai.stanford.edu/~ang/papers/icml04-onlinemetric.pdf)
- [Information-Theoretic Metric Learning (2007)](http://www.cs.utexas.edu/users/pjain/pubs/metriclearning_icml.pdf)
- [Online Metric Learning and Fast Similarity Search (2009)](http://people.bu.edu/bkulis/pubs/nips_online.pdf)

### Ensemble Models

- [A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting (1997)](http://www.face-rec.org/algorithms/Boosting-Ensemble/decision-theoretic_generalization.pdf)
- [Online Bagging and Boosting (2001)](https://ti.arc.nasa.gov/m/profile/oza/files/ozru01a.pdf)
- [Optimal and Adaptive Algorithms for Online Boosting (2015)](http://proceedings.mlr.press/v37/beygelzimer15.pdf) — An implementation is available [here](https://github.com/VowpalWabbit/vowpal_wabbit/blob/master/vowpalwabbit/boosting.cc)
- [Adaptive XGBoost for Evolving Data Streams (2020)](https://arxiv.org/abs/2005.07353) — An implementation is available [here](https://github.com/jacobmontiel/AdaptiveXGBoostClassifier)

### Expert Learning

- [On the optimality of the Hedge Algorithm in the Stochastic Regime](https://arxiv.org/pdf/1809.01382.pdf)

### Miscellaneous

- [Online EM Algorithm for Latent Data Models (2007)](https://arxiv.org/abs/0712.4273) — Source code is available [here](https://www.di.ens.fr/~cappe/Code/OnlineEM/)
- [A Complete Recipe for Stochastic Gradient MCMC (2015)](https://arxiv.org/abs/1506.04696)
- [Multi-Output Chain Models and their Application in Data Streams (2019)](https://jmread.github.io/talks/2019_03_08-Imperial_Stats_Seminar.pdf)

### Surveys

- [Online Learning and Stochastic Approximations (1998)](https://leon.bottou.org/publications/pdf/online-1998.pdf)
- [Incremental Gradient, Subgradient, and Proximal Methods for Convex Optimization: A Survey (2011)](https://arxiv.org/abs/1507.01030)
- [Batch-Incremental versus Instance-Incremental Learning in Dynamic and Evolving Data (2013)](http://albertbifet.com/wp-content/uploads/2013/10/IDA2012.pdf)
- [Incremental learning algorithms and applications (2016)](https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2016-19.pdf)
- [Online Learning: A Comprehensive Survey (2018)](https://arxiv.org/abs/1802.02871)
- [Online Machine Learning in Big Data Streams (2018)](https://arxiv.org/abs/1802.05872v1)

### General-Purpose Algorithms

- [The Sliding DFT (2003)](https://pdfs.semanticscholar.org/525f/b581f9afe17b6ec21d6cb58ed42d1100943f.pdf) — An online variant of the Fourier Transform, a concise explanation is available [here](https://www.comm.utoronto.ca/~dimitris/ece431/slidingdft.pdf)
- [Maintaining Sliding Window Skylines on Data Streams (2006)](http://www.cs.ust.hk/~dimitris/PAPERS/TKDE06-Sky.pdf)
- [Sketching Algorithms for Big Data](https://www.sketchingbigdata.org/)

### Hyperparameter Tuning & AutoML

- [ChaCha for Online AutoML (2021)](https://arxiv.org/pdf/2106.04815.pdf)
- [Online AutoML: An adaptive AutoML framework for Online Learning (2022)](https://arxiv.org/pdf/2201.09750.pdf)

## Related Threads & Articles

- [How to use different batch sizes when training and predicting with LSTMs](https://machinelearningmastery.com/use-different-batch-sizes-training-predicting-python-keras/)
- [Online-Learning with a LSTM?](https://stats.stackexchange.com/questions/223829/online-learning-in-lstm) (see [here](https://www.quora.com/What-is-the-batch-size-in-LSTM) for batch size discussion for LSTM)
- [Online/Incremental Learning with Keras and Creme Image Recognition](https://pyimagesearch.com/2019/06/17/online-incremental-learning-with-keras-and-creme/))

## Videos

- [Andrew NG: Large Scale Machine Learning & Online Learning](https://www.youtube.com/watch?v=dnCzy_XKGbA)

## Blog Posts

- [What Is It and Who Needs It (Data Science Central, 2015)](https://www.datasciencecentral.com/profiles/blogs/stream-processing-what-is-it-and-who-needs-it)
- [What is Online Machine Learning? (Max Pagels, 2018)](https://medium.com/value-stream-design/online-machine-learning-515556ff72c5)
- [Machine Learning is going real-time (Chip Huyen, 2020)](https://huyenchip.com/2020/12/27/real-time-machine-learning.html)
- [The correct way to evaluate Online Machine Learning Models (Max Halford, 2020)](https://maxhalford.github.io/blog/online-learning-evaluation/)
- [Anomalies detection using River (Matias Aravena Gamboa, 2021)](https://medium.com/spikelab/anomalies-detection-using-river-398544d3536)
- [Introdução (não-extensiva) a Online Machine Learning (Saulo Mastelini, 2021)](https://medium.com/@saulomastelini/introdu%C3%A7%C3%A3o-a-online-machine-learning-874bd6b7c3c8)
- [Anomaly Detection with Bytewax & Redpanda (Bytewax, 2022)](https://www.bytewax.io/blog/anomaly-detection-bw-rpk/)
- [The Online Machine Learning predict/fit switcheroo (Max Halford, 2022)](https://maxhalford.github.io/blog/predict-fit-switcheroo/)
- [Real-time Machine Learning: Challenges and Solutions (Chip Huyen, 2022)](https://huyenchip.com/2022/01/02/real-time-machine-learning-challenges-and-solutions.html)
