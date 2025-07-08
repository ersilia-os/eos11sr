# Embedded Morgan Fingerprints

Embedded version of Morgan binary Fingerprints (eMFP) that preserves the key structural information of the encoded molecule. The implementation of eMFP offers an improved data representation that mitigates the risk of overfitting while enhancing model performance.

This model was incorporated on 2025-07-03.

## Information
### Identifiers
- **Ersilia Identifier:** `eos11sr`
- **Slug:** `emfps`

### Domain
- **Task:** `Representation`
- **Subtask:** `Featurization`
- **Biomedical Area:** `Any`
- **Target Organism:** `Not Applicable`
- **Tags:** `Embedding`, `Descriptor`

### Input
- **Input:** `Compound`
- **Input Dimension:** `1`

### Output
- **Output Dimension:** `1024`
- **Output Consistency:** `Fixed`
- **Interpretation:** Vector representation of a molecule

Below are the **Output Columns** of the model:
| Name | Type | Direction | Description |
|------|------|-----------|-------------|
| dim_0000 | float |  | Dimension index 0 of the embedded Morgan Fingerprint |
| dim_0001 | float |  | Dimension index 1 of the embedded Morgan Fingerprint |
| dim_0002 | float |  | Dimension index 2 of the embedded Morgan Fingerprint |
| dim_0003 | float |  | Dimension index 3 of the embedded Morgan Fingerprint |
| dim_0004 | float |  | Dimension index 4 of the embedded Morgan Fingerprint |
| dim_0005 | float |  | Dimension index 5 of the embedded Morgan Fingerprint |
| dim_0006 | float |  | Dimension index 6 of the embedded Morgan Fingerprint |
| dim_0007 | float |  | Dimension index 7 of the embedded Morgan Fingerprint |
| dim_0008 | float |  | Dimension index 8 of the embedded Morgan Fingerprint |
| dim_0009 | float |  | Dimension index 9 of the embedded Morgan Fingerprint |

_10 of 1024 columns are shown_
### Source and Deployment
- **Source:** `Local`
- **Source Type:** `External`
- **DockerHub**: [https://hub.docker.com/r/ersiliaos/eos11sr](https://hub.docker.com/r/ersiliaos/eos11sr)
- **Docker Architecture:** `AMD64`, `ARM64`
- **S3 Storage**: [https://ersilia-models-zipped.s3.eu-central-1.amazonaws.com/eos11sr.zip](https://ersilia-models-zipped.s3.eu-central-1.amazonaws.com/eos11sr.zip)

### Resource Consumption
- **Model Size (Mb):** `1`
- **Environment Size (Mb):** `794`
- **Image Size (Mb):** `709.61`

**Computational Performance (seconds):**
- 10 inputs: `32.34`
- 100 inputs: `18.25`
- 10000 inputs: `131.74`

### References
- **Source Code**: [https://github.com/MMLabCodes/eMFP](https://github.com/MMLabCodes/eMFP)
- **Publication**: [https://chemrxiv.org/engage/chemrxiv/article-details/685d5791c1cb1ecda07e0680](https://chemrxiv.org/engage/chemrxiv/article-details/685d5791c1cb1ecda07e0680)
- **Publication Type:** `Preprint`
- **Publication Year:** `2025`
- **Ersilia Contributor:** [arnaucoma24](https://github.com/arnaucoma24)

### License
This package is licensed under a [GPL-3.0](https://github.com/ersilia-os/ersilia/blob/master/LICENSE) license. The model contained within this package is licensed under a [None](LICENSE) license.

**Notice**: Ersilia grants access to models _as is_, directly from the original authors, please refer to the original code repository and/or publication if you use the model in your research.


## Use
To use this model locally, you need to have the [Ersilia CLI](https://github.com/ersilia-os/ersilia) installed.
The model can be **fetched** using the following command:
```bash
# fetch model from the Ersilia Model Hub
ersilia fetch eos11sr
```
Then, you can **serve**, **run** and **close** the model as follows:
```bash
# serve the model
ersilia serve eos11sr
# generate an example file
ersilia example -n 3 -f my_input.csv
# run the model
ersilia run -i my_input.csv -o my_output.csv
# close the model
ersilia close
```

## About Ersilia
The [Ersilia Open Source Initiative](https://ersilia.io) is a tech non-profit organization fueling sustainable research in the Global South.
Please [cite](https://github.com/ersilia-os/ersilia/blob/master/CITATION.cff) the Ersilia Model Hub if you've found this model to be useful. Always [let us know](https://github.com/ersilia-os/ersilia/issues) if you experience any issues while trying to run it.
If you want to contribute to our mission, consider [donating](https://www.ersilia.io/donate) to Ersilia!
