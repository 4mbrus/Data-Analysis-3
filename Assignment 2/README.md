README: Assignment 2
====================

This repository contains the the entire workflow for the second assignment of Data Analysis 3. The folder "reports" contains the required technical and project report. For more information on the decisions and steps behind the two reports, please consider the "assignment_2.ipynb" notebook.

Hard Facts:
------------
- Authors: Ambrus Fodor & Katharina Burtscher
- Date: 15.02.2026
- CEU - Data Analysis 3

Project Structure
-----------------

```
.
├── README.md
├── requirements.txt
├── data
│   ├── bisnode_firms_clean.csv   <- Transformed and cleaned data; used used for model building in "assignment_2.ipynb".
│   ├── cs_bisnode_panel.csv      <- Data from third party sources.
│   └── variable_table.md         <- comprehensive list of variables
├── notebooks-&-python-files
│   ├── assignment_2.ipynb        <- Model building, evaluation and selection.
│   ├── data_prep.ipynb           <- Data cleaning and preparation; based on class notebook
│   ├── make_it_smaller.py        <- Helper function to upload "cs_bisnode_panel.csv" to GitHub
│   └── py_helper_functions.py    <- Helper functions; based on class notebook
└── plots
    ├── technical_report.md       <- Technical Report as markdown
    ├── technical_report.pdf      <- Technical Report as PDF
    └── project_report.pdf        <- Project Report as PDF
```
