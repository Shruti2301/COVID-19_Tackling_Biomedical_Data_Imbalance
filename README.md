## Tackling COVID-19 Data Imbalance and Health Disparity using Deep Transfer Learning

The COVID-19 pandemic has underscored glaring healthcare inequalities across demographic groups. Biomedical datasets frequently exhibit imbalances in representation, contributing to biased models and unequal healthcare outcomes. This research proposes a novel approach leveraging deep transfer learning to mitigate such disparities. By transferring knowledge from well-represented groups to underrepresented ones, this study aims to rectify biases and enhance healthcare equity.


# Dataset
Link : https://catalog.data.gov/dataset/provisional-weekly-deaths-by-region-race-age-997d6

Dataset Overview:
The "Provisional COVID-19 Deaths by HHS Region, Race, and Age" dataset provides insights into deaths involving COVID-19 reported to the National Center for Health Statistics (NCHS) categorized by time period, Health and Human Services (HHS) region, race and Hispanic origin, and age group. Here's a breakdown of the dataset fields:

- Data As Of: Date when the dataset was last updated.

- Start Date: Start date of the time period covered by the data.

- End Date: End date of the time period covered by the data.

- Group: Data grouping, including by month, by week, by total, or by year.

   --> (Classes : By Month, By Week, By Total, By Year)

- Year: Year of the data, ranging from 2019 to 2023 and including combined periods like 2019/2020 and 2020-2023.

   --> (Classes : 2020, 2021, 2022, 2023, 2019/2020, 2020/2021,2020-2023,2021/2022)

- Month: Month of the data, ranging from January to December.
   --> (Classes :  (Blanks),1,2,3,4,5,6,7,8,9,10,11,12)

- MMWR Week: Week according to the Morbidity and Mortality Weekly Report (MMWR) system.
   --> (Classes : (Blanks), 1,2,3 etc.. upto 53) 

- Week-Ending Date: Date when the week ended.

- HHS Region: Geographic region as defined by the United States Department of Health and Human Services.
  
   --> (Classes :  United States,
        - Region 1: Connecticut, Maine, Massachusetts, New Hampshire, Rhode Island, Vermont;
        - Region 2: New Jersey, New York, New York City, Puerto Rico;
        - Region 3: Delaware, District of Columbia, Maryland, Pennsylvania, Virginia, West Virginia;
        - Region 4: Alabama, Florida, Georgia, Kentucky, Mississippi, North Carolina, South Carolina, Tennessee;
        - Region 5: Illinois, Indiana, Michigan, Minnesota, Ohio, Wisconsin;
        - Region 6: Arkansas, Louisiana, New Mexico, Oklahoma, Texas;
        - Region 7: Iowa, Kansas, Missouri, Nebraska;
        - Region 8: Colorado, Montana, North Dakota, South Dakota, Utah, Wyoming;
        - Region 9: Arizona, California, Hawaii, Nevada;
        - Region 10: Alaska, Idaho, Oregon, Washington. )

- Race and Hispanic Origin Group: Ethnicity and race categories including
  --> (Classes : Hispanic, non-Hispanic American Indian or Alaska Native, non-Hispanic Asian, non-Hispanic Black, non-Hispanic more than one race, non-Hispanic Native Hawaiian or other Pacific Islander, non-Hispanic White, and unknown.)

- Age Group: Age categories ranging from 0-4 years to 85 years and over.
  --> (Classes : (0-4 years),(5-17 years),(18-29 years),(30-39years),(40-49 years),(50-64 years),(65-74 years), (75-84 years), (85 years and over))

- COVID-19 Deaths: Number of deaths involving COVID-19 reported for the specified demographic group and time period.

- Total Deaths: Total number of deaths reported for the specified demographic group and time period.

- Footnote: Indicates if data cells have counts suppressed due to NCHS confidentiality standards.

This dataset includes information for the United States as a whole, as well as broken down by HHS regions, providing a detailed view of COVID-19 mortality patterns across different geographical areas, racial and ethnic groups, and age cohorts.

Please note that the dataset is no longer updated as of September 27, 2023, and similar data can be accessed from wonder.cdc.gov.



## Repository Structure:
data/: Directory for storing biomedical datasets.

models/: Directory for storing trained deep transfer learning models.

notebooks/: Jupyter notebooks for data exploration, model training, and evaluation.

scripts/: Scripts for data preprocessing, model training, and deployment.

docs/: Documentation files, including READMEs, project overview, and usage instructions.

requirements.txt: File containing dependencies required to run the project.

