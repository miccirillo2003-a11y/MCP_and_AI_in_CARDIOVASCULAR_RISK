# Heart Disease Clinical Analysis System — MCP + Claude AI

> **Course Project** — Programming for Data Science | Master's Degree in Data Science and Artificial Intelligence

## Overview

Cardiovascular diseases represent one of the leading causes of mortality worldwide, and early risk prediction plays a crucial role in prevention, diagnosis, and personalized treatment planning. This project provides researchers and medical professionals with an **AI-based decision support tool** (powered by Claude AI) for cardiovascular risk assessment.

The system leverages machine learning techniques to analyze clinical datasets and extract meaningful patterns that support clinical decision-making. The architecture is built around three **Model Context Protocol (MCP) servers**, providing modular and scalable services for data processing, statistical analysis, and knowledge graph construction.

**Live demo** — [View Claude AI analysis on the UCI dataset](https://claude.ai/public/artifacts/5a4b4d18-06dc-4a17-b27c-c334bb12713c)

---

## Dataset

**Heart Disease UCI** — UCI Machine Learning Repository
- 303 patient records collected at the Cleveland Clinic
- 14 original clinical features
- Binary target variable (presence/absence of heart disease)
- Public and anonymized data

---

## System Architecture

```
ClinicalDataETL  ──►  ClinicalDataAnalyzer  ──►  MedicalKnowledgeGraph
    (MCP 1)                  (MCP 2)                     (MCP 3)
  ETL & Feature           Statistical &              Knowledge Graph &
   Engineering          Exploratory Analysis         Clinical Insights
```

---

## MCP Server 1 — ClinicalDataETL

Manages the complete data lifecycle: downloading from UCI, cleaning, transformation, and feature engineering.

### Available Tools

| Tool | Description |
|------|-------------|
| `fetch_and_process_heart_data` | Downloads and processes the dataset with missing value handling and type conversion |
| `get_dataset_info` | Returns complete metadata: columns, source URL, description |
| `analyze_heart_disease` | Descriptive statistics and correlations |
| `export_processed_data` | Exports data in JSON, CSV, or dict format |

### Feature Engineering

The server automatically enriches the dataset with derived features:

| Feature | Description |
|---------|-------------|
| `age_group` | Age categorization: young / adult / elderly |
| `bp_category` | Blood pressure: normal / elevated / high |
| `chol_category` | Cholesterol: normal / borderline / high |
| `risk_score` | Calculated cardiovascular risk score |
| `has_disease` | Binary disease presence flag |

---

## MCP Server 2 — ClinicalDataAnalyzer

Advanced statistical analysis, anomaly detection, and data quality assessment engine specifically designed for clinical data.

### Available Tools

| Tool | Description |
|------|-------------|
| `analyze_clinical_data` | Descriptive statistics: mean, median, standard deviation, quartiles |
| `exploratory_data_analysis` | Complete EDA with correlations, patterns, and feature-outcome relationships |
| `detect_data_issues` | Detection of missing values, outliers, and inconsistencies |
| `get_analyzer_info` | Supported version and operations |

### Capabilities

- Preliminary data analysis and quick overview
- Complete exploratory analysis with distributions and relationships
- Automatic data quality assurance
- Full descriptive statistics per variable
- Correlation analysis between clinical variables
- Outlier detection via statistical methods

---

## MCP Server 3 — MedicalKnowledgeGraph

Builds and manages a knowledge graph representing complex relationships between clinical variables, using graph analytics to identify hidden patterns and non-obvious connections.

### Available Tools

| Tool | Description |
|------|-------------|
| `explore_relations` | Explores how a feature connects to others (correlation strength, type, clinical interpretation) |
| `clinical_insights` | Generates top-10 strongest relationships, risk factors, and protective variables |
| `kg_status` | System status, graph size, and available tools |

### Knowledge Graph Architecture

- **Nodes**: Each clinical feature with name, data type, descriptive statistics, and clinical relevance
- **Edges**: Weighted relationships with Pearson correlation coefficient, p-value significance, and clinical interpretation

---

## Integration with Claude AI

### Configuration

The three MCP servers must be configured in Claude's MCP configuration file and made accessible through the MCP protocol.

### Typical Analysis Workflow

**Scenario 1 — Complete Exploratory Analysis**

```
1. Load data       →  ClinicalDataETL: fetch_and_process_heart_data()
2. Quality check   →  ClinicalDataAnalyzer: analyze_clinical_data() + detect_data_issues()
3. Deep EDA        →  ClinicalDataAnalyzer: exploratory_data_analysis()
4. Knowledge Graph →  MedicalKnowledgeGraph: clinical_insights()
```

**Scenario 2 — Specific Investigation**

```
# Cholesterol relationships
MedicalKnowledgeGraph: explore_relations(feature="chol")

# Age and heart disease
ClinicalDataAnalyzer: exploratory_data_analysis()
MedicalKnowledgeGraph: explore_relations(feature="age")
MedicalKnowledgeGraph: explore_relations(feature="target")
```

### Effective Prompts

```
# General analysis
"Load and analyze the heart disease dataset"
"Give me a complete overview of clinical data"
"Identify the main cardiovascular risk factors"

# Specific investigations
"Explore cholesterol relationships with other variables"
"Which features are most correlated with disease presence?"
"Analyze the impact of age on cardiovascular parameters"

# Quality assurance
"Check data quality and identify problems"
"Are there outliers or anomalous values in the dataset?"

# Cardiologist-oriented
"Analyze the Heart Disease dataset and identify which combinations of risk factors
are most impactful for detecting heart disease. Provide actionable insights that
a cardiologist can use to prioritize preventative interventions."
```

### Best Practices

1. **Start with loading**: Always call `fetch_and_process_heart_data()` first to prepare the data
2. **Check quality**: Run `detect_data_issues()` before deep analysis
3. **Leverage the Knowledge Graph**: For relationship questions, the graph provides richer insights than simple correlations
4. **Sampling**: Use `sample_size` for quick tests on data subsets
5. **Export**: Use `export_processed_data()` to retrieve processed data for external use

---

## System Advantages

| Feature | Description |
|---------|-------------|
| **Automation** | Automatic management of the complete data lifecycle |
| **Quality Control** | Automatic detection of problems and anomalies |
| **Insights** | Knowledge graph for non-obvious clinical relationships |
| **Flexibility** | Sampling and export in multiple formats |
| **Integration** | Seamless integration with Claude AI via FastMCP |

---

## Future Developments

- Support for additional UCI clinical datasets
- Integrated machine learning algorithms for predictive modeling
- Interactive knowledge graph visualizations
- Export in standard medical formats (FHIR, HL7)
- Integration with clinical decision support systems

---

## Author

**Cirillo Michele** — January/February 2026
Master's Degree in Data Science and Artificial Intelligence
