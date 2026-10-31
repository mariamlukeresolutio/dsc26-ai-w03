# AI Usage

## Tools I used, and for what

I used ChatGPT to understand the assignment requirements, design customer history features, build the scikit-learn preprocessing pipeline, and plan the evaluation against dummy and rule-based baselines.

## AI-suggested code I rejected or fixed in add_history_features or time_split

An initial history-feature implementation needed special handling for parcels sharing the same timestamp. I reviewed the timestamp-grouping logic and ensured that parcels created at the same timestamp are not counted as prior parcels for one another. The `history_ties_are_not_prior` test is relevant to verifying this behavior.

I also reviewed the chronological splitting logic to ensure that records with tied timestamps are kept together rather than split across training and testing sets.

## How I checked that my model has no leakage

I excluded `parcel_id`, `courier_notes`, `delivered_at`, and `failed` from the model's input features. Customer history features use only information from strictly earlier timestamps, and I used a chronological train/test split to evaluate performance on later parcels.

## Testing and verification

I ran the provided public test suite, and all five tests passed. I also evaluated the models on the synthetic dataset and compared logistic regression and the rule-based baseline against the dummy baseline.