# Example QC Report

This report summarizes automated quality-control metrics per subject and sensor modality.

## Sample QC Metrics

| subject_id | device | signal | missingness | flatline_fraction | out_of_range_fraction | jump_fraction | coverage_seconds | candidate_windows | kept_windows | excluded_windows | exclusion_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S2 | chest | ACC | 0.0 | 7.175366184551515e-05 | 0.0 | 0.0 | 18237.0 | nan | nan | nan | nan |
| S2 | chest | ECG | 0.0 | 0.0064282204376237 | 0.0 | 0.0 | 6079.0 | nan | nan | nan | nan |
| S2 | chest | EDA | 0.0 | 0.0124973591749956 | 0.0 | 0.0 | 6079.0 | nan | nan | nan | nan |
| S2 | chest | EMG | 0.0 | 0.0024879567804753 | 0.0 | 0.0005205274646975 | 6079.0 | nan | nan | nan | nan |
| S2 | chest | Resp | 0.0 | 0.0336824744865166 | 0.0 | 0.0 | 6079.0 | nan | nan | nan | nan |
| S2 | chest | Temp | 0.0 | 0.0594280213916812 | 0.0 | 0.0 | 6079.0 | nan | nan | nan | nan |
| S2 | wrist | ACC | 0.0 | 0.0129510283884211 | 0.0 | 0.0001764958883312 | 18237.0 | nan | nan | nan | nan |
| S2 | wrist | BVP | 0.0 | 0.0012054850856562 | 0.0 | 0.0 | 6079.0 | nan | nan | nan | nan |
| S2 | wrist | EDA | 0.0 | 0.2604153814517787 | 0.0 | 0.0 | 6079.0 | nan | nan | nan | nan |
| S2 | wrist | TEMP | 0.0 | 0.8572074850915073 | 0.0 | 0.0 | 6079.0 | nan | nan | nan | nan |
| S2 | all | window_exclusion | nan | nan | nan | nan | nan | 201.0 | 77.0 | 124.0 | 0.6169154228855721 |
| S3 | chest | ACC | 0.0 | 0.0001724934671399 | 0.0 | 1.2760996293517288e-05 | 19479.0 | nan | nan | nan | nan |

## Notes

- Missingness and flatline fractions flag unreliable segments.
- Window exclusion counts reflect mixed-label windows removed by majority-vote filtering.
- These metrics are intended for transparent screening in decentralized phenotyping workflows.
