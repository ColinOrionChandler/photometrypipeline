# 2013 RN30 Find_Orb comparison

| Fit | Obs used | Latest used | RMS | Weighted RMS | U | q sigma (au) | a sigma (au) |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Historical saved fit | 38/42 | 2014-08-27T13:43:37.344Z | 0.19135 | 0.6111 | 4.6180 | 0.000513 | 0.00148 |
| Baseline rerun (42 input) | 38/42 | 2014-08-27T13:43:37.344Z | 0.19135 | 0.6111 | 4.4451 | 0.000389 | 0.00115 |
| New DECam only (15 input) | 15/15 | 2016-12-25T03:13:22.656Z | 0.05496 | 0.1099 | 5.7166 | 0.00186 | 0.0115 |
| Baseline plus DECam appended (57 input; Find_Orb removed duplicate/mismatched lines) | 55/55 | 2016-12-25T03:13:22.656Z | 0.16524 | 0.5197 | 3.6245 | 3.96e-05 | 0.000624 |
| Baseline with old flagged W84 lines replaced by new DECam (53 input) | 53/53 | 2016-12-25T03:13:22.656Z | 0.16741 | 0.5283 | 3.6286 | 3.99e-05 | 0.000628 |

Notes:
- The appended comparison preserves the prior flagged W84 lines; Find_Orb reports two exact duplicates removed and two same-date/same-obscode mismatches ignored.
- The replacement comparison removes the four old !-flagged 2014-10-18 W84 lines before adding the 15 new PP measurements, avoiding double-weighting that night.
- The clean replacement fit extends the used arc from 2014-08-27 to 2016-12-25 and improves weighted RMS from 0.6111 to 0.5283.
