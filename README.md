# Neuro-Symbolic AI for Automated Chemical Classification in the ChEBI Ontology

Assigns a molecule its parent classes in ChEBI, the reference ontology for biologically interesting chemistry, returning them as one semicolon-separated list. Glauer, Hastings and colleagues built Chebifier on a neuro-symbolic approach in which the ontology's own structure shapes the learning system, so the classifier can keep pace as ChEBI grows instead of being rebuilt by hand. Ersilia serves a later release than the published one, an ensemble that combines the transformer with a graph network, rule-based classifiers and a ChEBI lookup, reconciled by weighted majority vote.

This model was incorporated on 2025-08-22.Last packaged on 2026-09-17.

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
- **Interpretation:** Semicolon-separated list of predicted ChEBI ontology parent classes for the molecule.

Below are the **Output Columns** of the model:
| Name | Type | Direction | Description |
|------|------|-----------|-------------|
| chebi_predicted_parents | string |  | Semicolon-separated ChEBI predicted parents |


### Source and Deployment
- **Source:** `Local`
- **Source Type:** `External`
- **DockerHub**: [https://hub.docker.com/r/ersiliaos/eos6tpo](https://hub.docker.com/r/ersiliaos/eos6tpo)
- **Docker Architecture:** `AMD64`, `ARM64`
- **S3 Storage**: [https://ersilia-models-zipped.s3.eu-central-1.amazonaws.com/eos6tpo.zip](https://ersilia-models-zipped.s3.eu-central-1.amazonaws.com/eos6tpo.zip)

### Resource Consumption
- **Model Size (Mb):** `595`
- **Environment Size (Mb):** `7728`
- **Image Size (Mb):** `8353.59`

**Computational Performance (seconds):**
- 10 inputs: `81.79`
- 100 inputs: `118.75`
- 10000 inputs: `-1`

### References
- **Source Code**: [https://github.com/ChEB-AI/python-chebifier](https://github.com/ChEB-AI/python-chebifier)
- **Publication**: [https://doi.org/10.1039/D3DD00238A](https://doi.org/10.1039/D3DD00238A)
- **Publication Type:** `Peer reviewed`
- **Publication Year:** `2024`
- **Ersilia Contributor:** [arnaucoma24](https://github.com/arnaucoma24)

### License
This package is licensed under a [GPL-3.0](https://github.com/ersilia-os/ersilia/blob/master/LICENSE) license. The model contained within this package is licensed under a [MIT](LICENSE) license.

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
