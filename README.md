# Datathon-Project-2026
Summary
On Feb 1, 2026, using only what the university knew that day, our model forecast 4,396 first-year students. 4,361 enrolled, a 0.8% miss. An early count matters more each year: yield fell from 39.1% to 34.8% since 2022, and students commit later.
Data and method
We predicted which admits enroll from a Feb 1 snapshot: deposits and visits count only if dated before Feb 1. We excluded later data such as credit hours, which give a meaningless perfect AUC of 1.00. Gradient boosting and logistic regression were trained on 2022–24, calibrated on 2025, and tested once on 2026. Late applicants got a separate estimate.
Results
2026 forecast (actual 4,361)	Forecast	Miss
Gradient boosting (ours)	4,396	+35 (0.8%)
Last year's yield × admits	4,431	+70 (1.6%)
Deposits-to-date rule	4,301	−60 (1.4%)
Logistic regression	4,276	−85 (2.0%)
Predicted and actual enrollment rates match across groups of admits, and a Dec 1 forecast also landed within 1%. By residency, the misses offset: bordering-state students came in 5.0% under, residents 4.4% over.
What drives enrollment
•	Deposit by Feb 1: 15× the odds of enrolling.
•	Residents: 2.3× the odds of bordering-state admits. They were 32% of admits but 52% of the class.
•	Visit by Feb 1: 1.45× the odds. Higher class rank: 0.78× per standard deviation, since top students have more options.
•	Later deposits: 61% of final deposits were in by Feb 1 in 2026, down from about 75% in 2022–24.
Recommendations
1.	Track deposit pace against the same date last year.
2.	Plan budgets and housing on expected enrollees, not admit counts.
3.	Time out-of-state aid for spring, and bring bordering-state admits to campus early.
4.	Rerun the forecast monthly.
Limitations
These results rest on one test year (noise alone is about ±34 students). Odds are associations with other factors held fixed, and the data is synthetic

