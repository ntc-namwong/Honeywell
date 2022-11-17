# Honeywell

## Background

Control Performance Monitoring (CPM) is a tool to analyze and diagnose control performance for both regulatory control and advance process control. Even it is DCS-vendor and APC-vendor independence, it still be hard to configure for 3rd party APC vendor, especially Yokogawa.

This toolkit has been developed to enable Honeywell engineers to have a good experience (with minimum of manual steps) in the CPM configuration task if they were implementing SMOC or PACE models in the CPM software. Please be noted that this toolkit is not developed to replace CPM but to fulfill the gap of converting the Yokogawa APC model to the Generic APC model.

![Figure 1](Figures/Figure%2001.png)

## Prerequisite

This toolkit has been developed based on Python programming language. Below Python library is prerequisite.

| Python Library | Purpose |
| --- | --- |
| collections | To source direction from MV/DV to CV/POV by using a graph algorithm |
| control | To convert Laplace domain to time-series domain for plotting step response graph |
| matplotlib | To plot step response graph |
| numpy | To process array-wise data |
| os | To find current path directory |
| pandas | To read Excel, CSV and HTML files and process table-wise data |
| tkinter | To build a Graphic User Interface |
| xml | To read XML files |

## Input File Preparation

Before using the toolkit, two types of files including model file and tag mapping file need to be prepared. It is easy to generate those two files from Yokogawa APC system as below procedure.

### SMOC

#### Model File

1. Open workspace file (.WSP)
2. Select **Controller** folder 
3. Select the **Controller Name** and press **Select**
4. Select **Report** under the **Controller** folder
5. Go to **File** > **Save Report File** > Select file type as **HTM**

![Figure 2](Figures/Figure%2002.png)
![Figure 3](Figures/Figure%2003.png)

#### Tag Mapping File

1. Go to **UAPC Output** > **Variables**
2. Copy all data from **MV Details**, **DV Details**, **CV Details**, **POV Details**, and **SubController** tabs then paste into Excel and rename the sheet as **MV Detail**, **DV Detail**, **CV Detail**, **POV Detail**, and **Sub Controller** respectively

![Figure 4](Figures/Figure%2004.png)
![Figure 5](Figures/Figure%2005.png)

### PACE

#### Model File

1. Open workspace file (.QLX)
2. Select **Workspace** > **Export XML**
3. Browse **Save As** location
4. Check all **Workspace** data
5. Click **Export**

![Figure 6](Figures/Figure%2006.png)

#### Tag Mapping File

1. Go to **Modeling** (Bottom left) > **Model** (Top left)
2. Right click on the controller and select **Open Model**
3. Go to **Variables** tab
4. Select all rows in **Model Inputs** and **Model Outputs** then paste into Excel and rename the sheet as **Input** and **Output** respectively
5. Go to **Deployment** (Bottom left) > **Data Link** (Top left)
6. Select all rows and paste into Excel and rename the sheet as **Datalink**

![Figure 7](Figures/Figure%2007.png)
![Figure 8](Figures/Figure%2008.png)
![Figure 9](Figures/Figure%2009.png)

## Model Aggregation Concept

Yokogawa APC model whether SMOC or PACE can be illustrated as a flowchart below which shows a clear relationship from inputs to outputs. From left to right, the input tag (782FIC42.SV) can be either manipulated variable (MV) or a disturbance variable (DV). Through the Single Input Single Output (SISO) box, the output is called process output variable (POV), i.e., 782TI31A.PV, 782LIC47.MV, G2C2INSGS.PV, and MINC2INSGS. Some POVs can be classified into control variables (CV) as per design during the project phase. However, with only the flowchart, it cannot identify which input is MV or DV and which output is CV or POV.

Typically, the Yokogawa APC model consists of multiple intermediate models. For example, considering MV of 782FIC42.SV and CV of G2C2INSGS.PV, according to the figure below, there are 2 related models including the model between 782FIC42.SV – 782TI31A.PV and 782TI31A.PV – G2C2INSGS.PV. This concept is completely distinct from Honeywell Forge Advance Process Control (APC) which considers the model of 782FIC42.SV – G2C2INSGS.PV directly. Like the Honeywell Forge APC, CPM applies the same concept. Therefore, it is required to convert multiple intermediate models in SMOC and PACE to a direct model.

![Figure 10](Figures/Figure%2010.png)

Theoretically, a result of putting the transfer function in series is the multiplication of those two transfer functions. Regarding the above example, model of 782FIC42.SV – G2C2INSGS.PV can be calculated in figure below. The model result can be separately explained in 3 main points:

1.	**Gain:** An increase in 782FIC42.SV 1 unit leads to a decrease in 782TI31A.PV 0.4796 units. Also, an increase in 782TI31A.PV 1 unit leads to an increase in G2C2INSGS.PV by 0.2386 units. So, if 782FIC42.SV is increased by 1 unit, the G2C2INSGS.PV would be decreased by 0.4796 x 0.2386 = 0.1144 units.
2.	**Delay:** If there is a delay after adjusting 782FIC42.SV of 1.469 minutes and there is no delay after adjusting 782TI31A.PV, it would, hence, have a delay between 782FIC42.SV and G2C2INSGS.PV of 1.469 minutes.
3.	**Response order:** If both of responses between 782FIC42.SV – 782TI31A.PV and 782TI31A.PV – G2C2INSGS.PV are first order, it would show second-order response between 782FIC42.SV and G2C2INSGS.PV.

The toolkit applies this concept to transform the Yokogawa POV model to the final CPM Generic APC model.

![Figure 11](Figures/Figure%2011.png)

## User Guide

The toolkit contains 3 main tabs including Input, Tag Mapping, and Output tabs. In the beginning, the user must select the controller brand on the top-left toggle as shown in figure below. Currently, it supports only Yokogawa APC which is either SMOC or PACE.

![Figure 12](Figures/Figure%2012.png)

### Input Tab

According to [Input File Preparation](#input-file-preparation), two input files including a model file (HTM for SMOC and XML for PACE) and a tag mapping file (XLSX for both SMOC and PACE) must be prepared before using the toolkit.

Before clicking Fetch Data, there is a check box to exclude or include economic function (EF) variables in this fetching. By default, EF variables, such as MINC2INSGS in Figure 10, are excluded because these EF variables are not required in CPM. CPM is software to identify APC performance. Hence, only true CV, MV, and DV should be configured and other latent variables whether POV or EF variables should be excluded.

After fetching data, the controller name will be extracted from the file name. Moreover, MV, DV, CV, and POV are listed on the right-hand side. Also, the tag mapping table is loaded in the [Tag Mapping Tab](#tag-mapping-tab) as well.

In the [Input Tab](#input-tab), the toolkit allows users to move MV to DV, CV to POV, and POV to DV freely. Sometimes, the user might need to adjust the MV, DV, CV, and POV list manually. However, the moved variables cannot further move to other categories.

For example, CV1 is moved to POV7. The POV7 cannot further move to DV1 but it can return to the CV list. Ordering in the MV, DV, CV, and POV list can be manually changed by using up and down arrow icons in each variable type box.

When the variable lists are settled, the next step is to update tag mapping if requested.

![Figure 13](Figures/Figure%2013.png)

### Tag Mapping Tab

Tag Mapping Tab shows all tag lists in each MV, CV, and DV. Please be noted that the POV list would not be processed in Tag Mapping Tab and Output Tab. There is an option to move POV to CV or DV in the [Input Tab](#input-tab).

A list of the columns that the tag should be available is summarized in below table. If the tag is not displayed by default, please verify it in SMOC or PACE configuration. By the way, this tag mapping can also be manually modified in the toolkit by selecting the variable list, modifying the tag, and clicking the Update button.

| Variable Type | Key Name | Description | Possible Values |
| --- | --- | --- | --- |
| Controller | Mode | **Controller Mode** <br> Is the controller ON or OFF? | 0 = OFF <br> Any other value = ON <br> (Default = 1) |
| Controller | Opt Mode | **Controller optimizer mode** <br> Is the controller optimization ON or OFF? | 0 = OFF <br> Any other value = ON <br> (Default = 1) |
| CV | Tag Name | **CV tag naem** | N/A |
| CV | Mode | **CV mode** <br> Is the CV ON or OFF? | 0 = OFF <br> Any other value = ON <br> (Default = 1) |
| CV | High | **CV high constraint** | Real number <br> (Default = 0) |
| CV | Low | **CV low constraint** | Real number <br> (Default = 0) |
| CV | SST | **Steady-state target** <br> What does the controller expect the CV to be at <br> the end of the Prediction Horizon? | N/A |
| CV | PredErr | **Prediction error** <br> The error between the current CV and the unbiased <br> prediction of what CV would be the now last iteration <br> of the controller. <br> Please be noted that unbiased prediction also <br> can be configured. | N/A |
| CV | Opt Mode | **CV optimizer mode** <br> Is the CV Optimizer ON or OFF? | 0 = OFF <br> Any other value = ON <br> (Default = 1) |
| CV | Opt Coeff | **CV optimizer coefficient** | Real number <br> (Default = 0) |
| MV | Tag Name | **MV tag naem** | N/A |
| MV | Mode | **MV mode** <br> Is the MV ON or OFF? | 0 = OFF <br> Any other value = ON <br> (Default = 1) |
| MV | High | **MV high constraint** | Real number <br> (Default = 0) |
| MV | Low | **MV low constraint** | Real number <br> (Default = 0) |
| MV | Opt Mode | **MV optimizer mode** <br> Is the MV Optimizer ON or OFF? | 0 = OFF <br> Any other value = ON <br> (Default = 1) |
| MV | Opt Coeff | **MV optimizer coefficient** | Real number <br> (Default = 0) |
| MV | Max Move | **MV move upsize / downsize limit** <br> The limit of the MV can move up / down each iteration. | Real number <br> (Default = 0) |
| MV | Wind Up | **MV windup** <br> Is MV wound up (i.e., limited)? | 0 = OFF <br> Any other value = ON <br> (Default = 1) |
| DV | Tag Name | **DV tag naem** | N/A |
| DV | Mode | **DV mode** <br> Is the DV ON or OFF? | 0 = OFF <br> Any other value = ON <br> (Default = 1) |

![Figure 14](Figures/Figure%2014.png)

### Output Tab

This tab contains three main sections including processing, message log, and step response plot. All activities are logged and visualized in the message log box. By default, the step response plot is not displayed without clicking the Run button. Run button covers two activities:

* **CPM configuration file generation**
    The files include configuration XML file, model LO2 file, and tag mapping CSV file as shown in below figures. The LO2 model is generated based on the model aggregation concept in [Model Aggregation Concept](#model-aggregation-concept). However, there still have room to improve this part because the order of the aggregation result will be increased every time. In this example, it goes up to 7th order which might be not practical in the real world.
* **Step response plot visualization**
    To primary check the model result, a step response plot is one option. By engineering sense and experience in the plant, it can roughly confirm the correction of the model aggregation.

![Figure 15](Figures/Figure%2015.png)
![Figure 16](Figures/Figure%2016.png)
![Figure 17](Figures/Figure%2017.png)
![Figure 18](Figures/Figure%2018.png)
![Figure 19](Figures/Figure%2019.png)
