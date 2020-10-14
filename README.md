---
Azure Machine Learning Hackathon
---

#Objective:


Objective of this hackathon is to familiarize you with Azure ML looking at a
basic linear regression problem. The first day we will familiar ourselves with
the tools using Azure ML R SDK. On the 2nd day we will move towards using the
python SDK, and look at Auto ML.

The R SDK is in preview, which means it is not intended for production use,
however, this hackathon will provide an introduction into Azure ML leveraging R,
and then applying same concepts in python showcasing AutoML capabilities.

\-Connecting to a Workspace

\-Working with Datasets

\-Leveraging AML Compute

#References:


<https://github.com/Azure/MachineLearningNotebooks/tree/master/how-to-use-azureml>

<https://azure.github.io/azureml-sdk-for-r/reference/index.html>

<https://pypi.org/project/azureml-sdk/>

Steps after this hackathon:
---------------------------

\-Work with your team to setup a dev environment with custom roles based on your
business needs and the roles within your teams.

\-Productionizing Code

\-Building Azure ML Pipelines

Day 1:   R & Azure ML Workspace 9-2
-----------------------------------

>   1. Introductions 9-9:30

>   2. Workspace overview Import Notebook 9:30-10:00

>   2. Connecting to workspace in RStudio 10:00-10:30

>   3. Loading Data 10:30-11:00

>   4. Creating Compute Resources 11:00-11:30

>   *Lunch 11:30-12:30*

>   5. Training – Experiments & Runs 12:30-1:00

>   6. Deploying & Testing a model 1:00-2:00

 

Day 2: Python & Auto ML
-----------------------

1.  Auto ML through Designer 9:00-10:00

2.  Connecting to workspace in Jupyter Notebook, Loading Data, Creating Compute
    Resources 10:00-11:30

>   *Lunch 11:30-12:30*

1.  Deploying & Testing best model

2.  Training Auto ML Notebook– AutoML

![](media/c69b9bbc02702811bede4c5301949ad0.png)

The Workspace

![](media/4a13c87585ac2cca3e2ce26ad4f1de9d.png)

1.  Let’s get into the workspace with the link provided.

    ![](media/4ddfa3cb3ee53c076bcd24d38a749703.png)

    ![](media/e156bb54942d9e34fd77be2795b024bd.png)

    Let’s click on the create button.

    ![](media/6b6aea23628081572dab1f106eba0888.png)

Let’s get the source files.

If you navigate to the notebooks, we are able to leverage git and clone the
resources which will then be available to our compute cluster.

Click on the git icon and let’s clone our notebooks



![](media/4ae2c83ab98f1e823f49d4e2820cad73.png)

```
Git clone https://github.com/memasanz/AMLHack.git
```
