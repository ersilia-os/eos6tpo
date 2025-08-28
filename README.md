# Neuro-Symbolic AI for Automated Chemical Classification in the ChEBI Ontology

Automated classification of chemical entities within the ChEBI ontology using a neuro-symbolic AI framework, which intelligently leverages the ontology’s structure itself to guide and shape the learning system.

This model was incorporated on 2025-08-22.


## Information
### Identifiers
- **Ersilia Identifier:** `eos6tpo`
- **Slug:** `chebifier`

### Domain
- **Task:** `Annotation`
- **Subtask:** `Activity prediction`
- **Biomedical Area:** `Any`
- **Target Organism:** `Any`
- **Tags:** `Chemical notation`

### Input
- **Input:** `Compound`
- **Input Dimension:** `1`

### Output
- **Output Dimension:** `1`
- **Output Consistency:** `Fixed`
- **Interpretation:** The output corresponds to a list of semicolon-separated ChEBI predicted parents

Below are the **Output Columns** of the model:
| Name | Type | Direction | Description |
|------|------|-----------|-------------|
| chebi_predicted_parents | string |  | Semicolon-separated ChEBI predicted parents |


### Source and Deployment
- **Source:** `Local`
- **Source Type:** `External`
- **S3 Storage**: [https://ersilia-models-zipped.s3.eu-central-1.amazonaws.com/eos6tpo.zip](https://ersilia-models-zipped.s3.eu-central-1.amazonaws.com/eos6tpo.zip)

### Resource Consumption
- **Model Size (Mb):** `854`
- **Environment Size (Mb):** `7657`


### References
- **Source Code**: [https://github.com/ChEB-AI/python-chebifier](https://github.com/ChEB-AI/python-chebifier)
- **Publication**: [https://pubs.rsc.org/en/content/articlehtml/2024/dd/d3dd00238a](https://pubs.rsc.org/en/content/articlehtml/2024/dd/d3dd00238a)
- **Publication Type:** `Peer reviewed`
- **Publication Year:** `2025`
- **Ersilia Contributor:** [arnaucoma24](https://github.com/arnaucoma24)

### License
This package is licensed under a [GPL-3.0](https://github.com/ersilia-os/ersilia/blob/master/LICENSE) license. The model contained within this package is licensed under a [GPL-3.0-only](LICENSE) license.

**Notice**: Ersilia grants access to models _as is_, directly from the original authors, please refer to the original code repository and/or publication if you use the model in your research.


## Use
To use this model locally, you need to have the [Ersilia CLI](https://github.com/ersilia-os/ersilia) installed.
The model can be **fetched** using the following command:
```bash
# fetch model from the Ersilia Model Hub
ersilia fetch eos6tpo
```
Then, you can **serve**, **run** and **close** the model as follows:
```bash
# serve the model
ersilia serve eos6tpo
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
