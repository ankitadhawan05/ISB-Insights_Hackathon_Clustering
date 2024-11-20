# ISB-Insights_Hackathon

This project was a part of ISB@Insights Hackathon 2021. Its problem statement was to create a  multi-dimensional poverty and deprivation Index based on the Mission Antayodya Survey 2019 (a household survey done to examine the state of facilities in terms of access to healthcare, education, banking and infrastructure facilities)

The dataset was divided into 4 parts to create sub-indices on education, banking, healthcare and infra. 
The code attached is the backend work done to cluster rural districts of India for education and healthcare. K-Means clustering was used to cluster the districts. Similar variables were dropped to remove the multi-collinearity and transformations on the raw data set was done before performing clustering.

For data transformation- based on literature scans, a threshold value of 30% was chosen to define deprivation i.e. the districts that had less than 30% of facilities were considered deprived and assigned a value of 1. rest of the values were assigned values of 0.

