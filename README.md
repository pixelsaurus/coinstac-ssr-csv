## COINSTAC Singleshot Regression using CSV based data
Coinstac Code for Single-Shot Regression (ssr) on FreeSurfer Data

In COINSTAC, you can select any of the following Regions of Interest as Dependent variables in the regression.

**Tools:** Python 3.6.5, coinstac-simulator 4.2.0
### Steps

sudo npm i -g coinstac-simulator@4.2.0
git clone https://github.com/trendscenter/coinstac-ssr-fsl-2.git
cd coinstac_ssr_fsl_2
docker build -t ssr_fsl .
coinstac-simulator
