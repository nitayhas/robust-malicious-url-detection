# Robust Malicious URLs Detection

#### Thesis abstract
Malicious URLs are becoming increasingly common and pose a severe cybersecurity threat. Many types of current attacks using URL for the attack communications for example C&C, Phishing and Spear Phishing. Even though recent works report progress in detecting these attacks, there are serious problems and in particular the robustness of these attacks, that should be addressed. We propose a new  malicious URLs detection methodology using a robust set of features that resistant to Adversarial Example attacks. Our contributions are twofold: first, our mechanism results with high performance based on data collected from ~5000 benign active URLs and ~1350 malicious active (attacks) URLs. Second, we provide a hybrid features set that combined widely used features that showed resistant to Adversarial Examples attacks and novel engineered features. While we show that our features set improve the performance of the classifier (an increase in the model's F1-Score from 90.2% to 98.4%), we also demonstrated the effectiveness of constructing a model that is solely trained based on our novel features, resulting in an F1-Score of 95.2%.


### Explanation of the code
- **Datasets**
	This folder contains all the data collected during the research (divided into folders) e.g. urls, DNS traffic, VirusTotal data etc...
- **DatasetsCollectors**
	In the DatasetsCollectors folder you can find the data collectors.
	Inside the Tools folder (That inside DatasetsCollectors folder) you can find the Classes that handle the communication in order to get the appropriate data.
- **Models**
	This folder includes the necessary model classes.
- **Tests**
	The Tests folder contains both the test files that check each model specifically and the feature robustness and feature permutations tests.
- **Tools**
	The two classes that handle the feature extraction and the communication ratio tables phases.



## Acknowledgment
I would like to thank my supervisors, Dr.Amit Dvir and Dr.Chen Hajaj, who assisted and guided me throughout this research and degree work.
